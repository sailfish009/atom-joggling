import gc
import sys
from abc import ABC, abstractmethod

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import CrossEntropyLoss, L1Loss, NLLLoss
from torch.nn.functional import softmax
from tqdm import tqdm, trange

from supercon.utils import save_checkpoint


class BaseModel(nn.Module, ABC):
    """ A base class for regression and classification models. """

    def __init__(
        self, task: str, robust: bool, epoch: int = 1, checkpoint_dir: str = None
    ) -> None:
        super().__init__()
        self.task = task
        self.robust = robust
        self.epoch = epoch
        self.best_val_score = None
        self.val_score_name = "MAE" if task == "regression" else "Acc"
        self.model_params = {}
        self.checkpoint_dir = checkpoint_dir
        if task == "classification":
            self.criterion = NLLLoss() if robust else CrossEntropyLoss()
        else:  # regression
            self.criterion = RobustL1Loss if robust else L1Loss()

    @property
    def n_params(self, trainable=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad is trainable)

    def fit(
        self,
        train_loader,
        val_loader,
        optimizer,
        epochs: int,
        writer=None,
        checkpoint: bool = True,
        verbose: bool = True,
    ) -> None:
        start_epoch = self.epoch
        for epoch in trange(
            start_epoch, start_epoch + epochs, desc="Epochs: ", disable=verbose
        ):
            self.epoch += 1
            # Training
            if verbose:
                print(f"\nEpoch: [{epoch}/{start_epoch + epochs - 1}]", flush=True)
            train_metrics = self.evaluate(train_loader, optimizer, "train", verbose)

            if verbose:
                metric_str = "\t ".join(
                    f"{key} {val:.3f}" for key, val in train_metrics.items()
                )
                print(f"Train      : {metric_str}")

            # Validation
            if val_loader is None:
                is_best = False
            else:
                with torch.no_grad():
                    # Evaluate on validation set
                    val_metrics = self.evaluate(val_loader, None, action="val")

                if verbose:
                    metric_str = "\t ".join(
                        f"{key} {val:.3f}" for key, val in val_metrics.items()
                    )
                    print(f"Validation : {metric_str}")

                if self.task == "regression":
                    val_score = val_metrics["mae"]
                    # or fallback for first epoch
                    is_best = val_score < (self.best_val_score or 1e6)
                else:  # classification
                    val_score = val_metrics["acc"]
                    is_best = val_score > (self.best_val_score or 0)

            if is_best:
                self.best_val_score = val_score

            if checkpoint:
                checkpoint_dict = {
                    "task": self.task,
                    "model_params": self.model_params,
                    "state_dict": self.state_dict(),
                    "epoch": self.epoch,
                    "best_val_score": self.best_val_score,
                    "val_score_name": self.val_score_name,
                    "optimizer": optimizer.state_dict(),
                }
                normalizer = train_loader.dataset.normalizer
                if normalizer is not None:
                    checkpoint_dict["normalizer"] = normalizer.state_dict()

                if hasattr(self, "swa"):
                    checkpoint_dict["swa"] = self.swa.copy()
                    for key in ["model", "scheduler"]:  # refers to SWA scheduler
                        # remove model as it can't and needs not be serialized
                        del checkpoint_dict["swa"][key]
                        state_dict = self.swa[key].state_dict()
                        checkpoint_dict["swa"][f"{key}_state_dict"] = state_dict

                save_checkpoint(checkpoint_dict, is_best, self.checkpoint_dir)

            if writer is not None:
                for metric, val in train_metrics.items():
                    writer.add_scalar(f"training/{metric}", val, epoch + 1)

                if val_loader is not None:
                    for metric, val in val_metrics.items():
                        writer.add_scalar(f"validation/{metric}", val, epoch + 1)

            if hasattr(self, "swa") and epoch > self.swa["start"]:
                self.swa["model"].update_parameters(self)
                self.swa["scheduler"].step()

            # catch memory leak
            gc.collect()

        # if self.swa:
        #     # handle batch norm + SWA (does nothing if model has no batch norm)
        #     currently incompatible (https://github.com/pytorch/pytorch/issues/49082)
        #     torch.optim.swa_utils.update_bn(train_loader, self.swa["model"])

    def evaluate(
        self, loader, optimizer, action: str = "train", verbose: bool = False
    ) -> dict:
        """ Evaluate the model for one epoch """

        assert action in ["train", "val"], f"action must be train or val, got {action}"

        normalizer = loader.dataset.normalizer

        self.train() if action == "train" else self.eval()

        # records both regr. and clf. metrics for an epoch to compute averages below
        metrics = {key: [] for key in ["loss", "mae", "rmse", "acc", "f1"]}

        # we do not need batch_comp or batch_ids when training
        for input_, target, *_ in tqdm(loader, disable=not verbose, file=sys.stdout):

            # compute output
            output = self(*input_)

            if self.task == "regression":
                target_norm = normalizer.norm(target)
                if self.robust:
                    mean, log_std = output.chunk(2, dim=1)
                    loss = self.criterion(mean, log_std, target_norm)

                    pred = normalizer.denorm(mean.data.cpu())
                else:
                    loss = self.criterion(output.squeeze(), target_norm)
                    pred = normalizer.denorm(output.data.cpu())

                metrics["mae"] += [(pred - target).abs().mean()]
                metrics["rmse"] += [(pred - target).pow(2).mean().sqrt()]

            else:  # classification
                if self.robust:
                    output, log_std = output.chunk(2, dim=1)
                    logits = sampled_softmax(output, log_std)
                    loss = self.criterion(torch.log(logits), target)
                else:
                    loss = self.criterion(output, target)
                    logits = softmax(output, dim=1)
                metrics["acc"] += [(logits.argmax(1) == target).float().mean().cpu()]

                # call .cpu() for automatic numpy conversion
                # since sklearn metrics need numpy arrays
                f1 = f1_score(logits.argmax(1).cpu(), target.cpu(), average="weighted")
                metrics["f1"] += [f1]

            metrics["loss"] += [loss.cpu().item()]

            if action == "train":
                # compute gradient and take an optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        metrics = {key: sum(val) / len(val) for key, val in metrics.items() if val}

        return metrics

    @torch.no_grad()
    def predict(self, loader, verbose: bool = False) -> tuple:
        """ Generate predictions """

        material_ids, formulas, targets, outputs = [], [], [], []

        # Ensure model is in evaluation mode
        self.eval()

        # iterate over mini-batches
        for features, targs, comps, ids in tqdm(loader, disable=not verbose):

            # compute output
            output = self(*features)

            # collect the model outputs
            material_ids += ids
            formulas += comps
            targets.append(targs)
            outputs.append(output)

        targets = torch.cat(targets, dim=0).cpu()
        outputs = torch.cat(outputs, dim=0).cpu()

        return material_ids, formulas, targets, outputs

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass through the model. Needs to be implemented in any derived
        model class.
        """
        raise NotImplementedError("forward() is not defined!")


def RobustL1Loss(output, log_std, target) -> float:
    """Robust L1 loss using a Lorentzian prior.
    Allows for aleatoric uncertainty estimation.
    """
    loss = 2 ** 0.5 * (output - target).abs() / log_std.exp() + log_std
    return loss.mean()


def RobustL2Loss(output, log_std, target) -> float:
    """Robust L2 loss using a gaussian prior.
    Allows for aleatoric uncertainty estimation.
    """
    loss = 0.5 * (output - target) ** 2 / (2 * log_std).exp() + log_std
    return loss.mean()


def sampled_softmax(pre_logits, log_std, samples: int = 10) -> float:
    """Draw samples from Gaussian distributed pre-logits and use these to
    estimate a mean and aleatoric uncertainty.
    """
    # NOTE here as we do not risk dividing by zero should we really be
    # predicting log_std or is there another way to deal with negative numbers?
    # This choice may have an unknown effect on the calibration of the uncertainties
    sam_std = log_std.exp().repeat_interleave(samples, dim=0)
    # TODO here we are normally distributing the samples even if the loss
    # uses a different prior?
    epsilon = torch.randn_like(sam_std)
    pre_logits = pre_logits.repeat_interleave(samples, dim=0) + epsilon * sam_std
    logits = softmax(pre_logits, dim=1).view(len(log_std), samples, -1)
    return logits.mean(dim=1)

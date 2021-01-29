import gc
import sys
from abc import ABC, abstractmethod

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.functional import softmax
from tqdm import tqdm

from supercon.core import sampled_softmax, save_checkpoint


class BaseModel(nn.Module, ABC):
    """ A base class for regression and classification models. """

    def __init__(self, task, robust, device=None, epoch=1):
        super().__init__()
        self.task = task
        self.robust = robust
        self.device = device
        self.epoch = epoch
        self.best_val_score = None
        self.val_score_name = "MAE" if task == "regression" else "Acc"
        self.model_params = {}

    @property
    def n_params(self, trainable=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad is trainable)

    def fit(
        self,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs,
        criterion,
        normalizer,
        model_dir,
        checkpoint=True,
        writer=None,
        verbose=True,
    ):
        start_epoch = self.epoch
        for epoch in range(start_epoch, start_epoch + epochs):
            self.epoch += 1
            # Training
            print(f"\nEpoch: [{epoch}/{start_epoch + epochs - 1}]", flush=True)
            train_metrics = self.evaluate(
                train_loader, criterion, optimizer, normalizer, "train", verbose
            )

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
                    val_metrics = self.evaluate(
                        val_loader, criterion, None, normalizer, action="val"
                    )

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
                    "scheduler": scheduler.state_dict(),
                }

                if self.task == "regression":
                    checkpoint_dict["normalizer"] = normalizer.state_dict()

                if hasattr(self, "swa"):
                    checkpoint_dict["swa"] = self.swa.copy()
                    for key in ["model", "scheduler"]:
                        # remove model as it can't and needs not be serialized
                        del checkpoint_dict["swa"][key]
                        state_dict = self.swa[key].state_dict()
                        checkpoint_dict["swa"][f"{key}_state_dict"] = state_dict

                save_checkpoint(checkpoint_dict, is_best, model_dir)

            if writer is not None:
                for metric, val in train_metrics.items():
                    writer.add_scalar(f"training/{metric}", val, epoch + 1)

                if val_loader is not None:
                    for metric, val in val_metrics.items():
                        writer.add_scalar(f"validation/{metric}", val, epoch + 1)

            if hasattr(self, "swa") and epoch > self.swa["start"]:
                self.swa["model"].update_parameters(self)
                self.swa["scheduler"].step()
            else:
                scheduler.step()

            # catch memory leak
            gc.collect()

        # if self.swa:
        #     # handle batch norm + SWA (does nothing if model has no batch norm)
        #     currently incompatible (https://github.com/pytorch/pytorch/issues/49082)
        #     torch.optim.swa_utils.update_bn(train_loader, self.swa["model"])

    def evaluate(
        self, loader, criterion, optimizer, normalizer, action="train", verbose=False
    ):
        """ Evaluate the model for one epoch """

        assert action in ["train", "val"], f"action must be train or val, got {action}"
        self.train() if action == "train" else self.eval()

        # records both regr. and clf. metrics for an epoch to compute averages below
        metrics = {key: [] for key in ["loss", "mae", "rmse", "acc", "f1"]}

        # we do not need batch_comp or batch_ids when training
        for input_, target, *_ in tqdm(loader, disable=not verbose, file=sys.stdout):

            # move tensors to GPU
            input_ = (tensor.to(self.device) for tensor in input_)

            # compute output
            output = self(*input_)

            if self.task == "regression":
                target_norm = normalizer.norm(target)
                target_norm = target_norm.to(self.device)
                if self.robust:
                    mean, log_std = output.chunk(2, dim=1)
                    loss = criterion(mean, log_std, target_norm)

                    pred = normalizer.denorm(mean.data.cpu())
                else:
                    loss = criterion(output, target_norm)
                    pred = normalizer.denorm(output.data.cpu())

                metrics["mae"] += [(pred - target).abs().mean()]
                metrics["rmse"] += [(pred - target).pow(2).mean().sqrt()]

            else:  # classification
                target = target.to(self.device).squeeze()
                if self.robust:
                    output, log_std = output.chunk(2, dim=1)
                    logits = sampled_softmax(output, log_std)
                    loss = criterion(torch.log(logits), target)
                else:
                    loss = criterion(output, target)
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
    def predict(self, generator, verbose=False):
        """ Generate predictions """

        material_ids, formulas, targets, outputs = [], [], [], []

        # Ensure model is in evaluation mode
        self.eval()

        # iterate over mini-batches
        for features, targs, comps, ids in tqdm(generator, disable=not verbose):

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
    def forward(self, *x):
        """
        Forward pass through the model. Needs to be implemented in any derived
        model class.
        """
        raise NotImplementedError("forward() is not defined!")

import argparse

import numpy as np
import torch

parser = argparse.ArgumentParser(description="PyTorch MixMatch Training")
# Optimization options
parser.add_argument(
    "--epochs",
    default=10,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--batch-size", default=64, type=int, metavar="N", help="train batchsize"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.002,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
# Checkpoints
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
# Method options
parser.add_argument(
    "--train-iterations", type=int, default=10, help="Number of batches per epoch"
)
parser.add_argument("--out-dir", default="runs", help="Directory to output the result")
# Mixup parameter of the Beta distribution
parser.add_argument("--alpha", default=0.75, type=float)
# weighting factor for the unlabeled loss term
parser.add_argument("--lambda-u", default=75, type=float)
# temperature used during sharpening of the pseudo-label distribution
parser.add_argument("--T", default=0.5, type=float)


args, _ = parser.parse_known_args()


np.random.seed(0)
torch.manual_seed(0)


def train(
    labeled_trainloader,
    unlabeled_trainloader,
    model,
    optimizer,
    criterion,
    epoch,
    use_cuda,
):

    losses = {"total": [], "loss_u": [], "loss_x": []}

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.train_iterations):
        try:
            inputs_x, targets_x, *_ = next(labeled_train_iter)
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, *_ = next(labeled_train_iter)

        try:
            inputs_u, *_ = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, *_ = next(unlabeled_train_iter)

        batch_size = targets_x.size(0)

        # Transform labels to one-hot
        targets_x = torch.nn.functional.one_hot(targets_x)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
            inputs_u = inputs_u.cuda()

        crys_fea_x = model.material_nn(*inputs_x)
        crys_fea_u = model.material_nn(*inputs_u)

        with torch.no_grad():
            # compute guessed labels of unlabeled samples
            outputs_u = model(*inputs_u)
            # sharpen the model's output distribution (for entropy minimization)
            proba_u = outputs_u.softmax(dim=1)
            pt = proba_u ** (1 / args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([crys_fea_x, crys_fea_u], dim=0)
        all_targets = torch.cat([targets_x, targets_u], dim=0)

        ell = np.random.beta(args.alpha, args.alpha)

        ell = max(ell, 1 - ell)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = ell * input_a + (1 - ell) * input_b
        mixed_target = ell * target_a + (1 - ell) * target_b

        # interleave labeled and unlabeled samples between batches to
        # get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model.output_nn(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model.output_nn(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(
            logits_x,
            mixed_target[:batch_size],
            logits_u,
            mixed_target[batch_size:],
            epoch + batch_idx / args.train_iterations,
        )

        loss = Lx + w * Lu

        # record loss
        losses["total"].append(loss.item())
        losses["loss_x"].append(Lx.item())
        losses["loss_u"].append(Lu.item())

        # update model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return [mean(x) for x in losses.values()]


def validate(val_loader, model, criterion):

    losses, avg_acc = [], []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for inputs, targets, *_ in val_loader:
            # compute output
            outputs = model(*inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc = (outputs.argmax(1) == targets).float().mean()
            losses.append(loss.item())
            avg_acc.append(acc.item())

    return mean(losses), mean(avg_acc)


def mean(lst):
    return sum(lst) / len(lst)


def save_checkpoint(state, is_best, checkpoint=args.out_dir, filename="/checkpoint"):
    torch.save(state, checkpoint + filename)
    if is_best:
        torch.save(state, checkpoint + "/best_model")


def linear_rampup(current, rampup_length=args.epochs):
    """Linearly ramps up the unlabeled loss with epoch count. As the
    pseudo-labels become more accurate, they should be weighted more.

    Args:
        current (float): current loss weighting factor
        rampup_length (ing, optional): number of epochs until rampup completes.
            Defaults to args.epochs.

    Returns:
        [type]: [description]
    """
    if rampup_length == 0:
        return 1.0
    else:
        current /= rampup_length
        current = max(0.0, min(current, 1.0))
        return current


class SemiLoss:
    def __call__(self, preds_x, targets_x, preds_u, targets_u, epoch):
        probs_u = preds_u.softmax(dim=1)

        Lx = -(preds_x.log_softmax(dim=1) * targets_x).sum(dim=1).mean()
        Lu = (probs_u - targets_u).pow(2).mean()  # MSE

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]
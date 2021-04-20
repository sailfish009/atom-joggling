import sys

import numpy as np
import torch
from tqdm import trange

from atom_joggling.utils import ROOT, mean, parser


# Method specific options

# Mixup parameter of the Beta distribution
parser.add_argument("--alpha", default=0.75, type=float)
# weighting factor for the unlabeled loss term
parser.add_argument("--lambda-u", default=75, type=float)
# temperature used during sharpening of the pseudo-label distribution
parser.add_argument("--T", default=0.5, type=float)

# mixmatch used a default of 1024 but CIFAR10 is a way bigger dataset so 128 seems good
parser.add_argument(
    "--train-iterations", type=int, default=128, help="Number of batches per epoch"
)

args, _ = parser.parse_known_args()

# add time stamp if no custom out_dir was specified
if not args.resume and not args.out_dir:
    out_dir = f"{ROOT}/runs/mixup/iters={args.train_iterations}_T={args.T}"
    out_dir += f"alpha={args.alpha}-lambdaU={args.lambda_u}"
    out_dir += "_robust" if args.robust else ""
    args.out_dir = out_dir


def train_with_mixup(
    labeled_loader, unlabeled_loader, model, optimizer, criterion, verbose=True
) -> tuple:
    """Train a model with mixup by randomly sampling linear combinations of
    unlabeled and unlabeled inputs as well as targets. Uses model-generated
    pseudo-labels to compute interpolated targets.

    Args:
        labeled_loader (torch.utils.data.DataLoader): labeled data
        unlabeled_loader (torch.utils.data.DataLoader): unlabeled data
        model (nn.Module): the instantiated model
        optimizer (torch.optim.Optimizer): optimizer
        criterion: loss function that computes labeled, unlabeled and combined losses

    Returns:
        [loss, Lx, Lu]: 3-tuple of total, labeled and unlabeled losses
    """
    losses = {"total": [], "loss_u": [], "loss_x": []}

    labeled_train_iter = iter(labeled_loader)
    unlabeled_train_iter = iter(unlabeled_loader)

    model.train()
    # file=sys.stdout (default stderr) prevents print order issues by using
    # same stream as print https://stackoverflow.com/a/45265707
    for _ in trange(
        args.train_iterations, desc="Batches:", file=sys.stdout, disable=not verbose
    ):
        try:
            inputs_x, targets_x, *_ = next(labeled_train_iter)
        except StopIteration:
            labeled_train_iter = iter(labeled_loader)
            inputs_x, targets_x, *_ = next(labeled_train_iter)

        try:
            inputs_u, *_ = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_loader)
            inputs_u, *_ = next(unlabeled_train_iter)

        batch_size = targets_x.size(0)

        # Transform labels to one-hot
        targets_x = torch.nn.functional.one_hot(targets_x)

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

        Lx, Lu, u_ramp = criterion(
            logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:]
        )

        loss = Lx + u_ramp * Lu

        # record loss
        losses["total"].append(loss.item())
        losses["loss_x"].append(Lx.item())
        losses["loss_u"].append(Lu.item())

        # update model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # dicts are insertion ordered as of Python 3.6
    losses = (mean(x) for x in losses.values())
    return (*losses, u_ramp)


@torch.no_grad()
def validate_mixup(val_loader, model, criterion) -> tuple:

    losses, avg_acc = [], []

    # switch to evaluate mode
    model.eval()

    for inputs, targets, *_ in val_loader:
        # compute output
        outputs = model(*inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc = (outputs.argmax(1) == targets).float().mean()
        losses.append(loss.item())
        avg_acc.append(acc.item())

    return mean(losses), mean(avg_acc)


def linear_rampup(current: int, rampup_length: int) -> float:
    """Linearly ramps up the unlabeled loss with epoch count. As the
    pseudo-labels become more accurate, they can be be weighted more.

    Args:
        current (float): current loss weighting factor
        rampup_length (ing, optional): number of epochs until rampup completes.
            Defaults to args.epochs.

    Returns:
        float: increasing weighting factor for the unlabeled loss
    """
    if rampup_length == 0:
        return 1.0
    else:
        current /= rampup_length
        current = max(0.0, min(current, 1.0))
        return current


class SemiLoss:
    def __init__(self, u_ramp_length: int, ramp_start: int = 0) -> None:
        self.u_ramp_length = u_ramp_length
        self.ramp_count = ramp_start

    def __call__(self, preds_x, targets_x, preds_u, targets_u) -> tuple:
        self.ramp_count += 1
        probs_u = preds_u.softmax(dim=1)

        Lx = -(preds_x.log_softmax(dim=1) * targets_x).sum(dim=1).mean()
        Lu = (probs_u - targets_u).pow(2).mean()  # MSE

        ramp = linear_rampup(self.ramp_count, self.u_ramp_length)
        return Lx, Lu, args.lambda_u * ramp


def interleave_offsets(batch_size: int, nu: int) -> list:
    groups = [batch_size // (nu + 1)] * (nu + 1)
    for x in range(batch_size - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch_size
    return offsets


def interleave(xy: list, batch_size: int) -> list:
    """Interleave labeled and unlabeled samples between batches to
    get correct batch normalization calculation.

    Args:
        xy (tuple): list of inputs or targets split by batch size
        batch_size (int): batch size

    Returns:
        [list]: list of interleaved inputs or targets
    """
    nu = len(xy) - 1
    offsets = interleave_offsets(batch_size, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

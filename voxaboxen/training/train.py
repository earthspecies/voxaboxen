"""
Functions required to train model
"""

import argparse
import os
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tqdm
import yaml
from einops import rearrange
from torch import Tensor

from voxaboxen.data.data import get_train_dataloader, get_val_dataloader
from voxaboxen.evaluation.evaluation import (
    evaluate_based_on_manifest,
    predict_and_generate_manifest,
)
from voxaboxen.evaluation.plotters import plot_eval
from voxaboxen.model.model import rms_and_mixup

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cpu":
    import warnings

    warnings.warn("Only using CPU! Check CUDA", stacklevel=2)


def train(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    """Train a model with the given arguments.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    args : argparse.Namespace or similar
        Training configuration containing:
        - lr: learning rate
        - n_epochs: number of epochs
        - early_stopping: whether to use early stopping
        - val_during_training: whether to validate during training
        - patience: early stopping patience
        - min_epochs: minimum epochs before validation
        - experiment_dir: directory to save outputs
        - is_test: whether this is a test run
        - bidirectional: whether model is bidirectional
        - rho: class loss weight
        - lamb: regression loss weight

    Returns
    -------
    torch.nn.Module
        The trained model
    """

    detection_loss_fn = get_detection_loss_fn(args)
    reg_loss_fn = get_reg_loss_fn(args)
    class_loss_fn = get_class_loss_fn(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.n_epochs, eta_min=0, last_epoch=-1
    )

    train_evals = []
    learning_rates = []
    val_evals = []

    if args.early_stopping:
        assert args.val_info_fp is not None

    if (args.val_info_fp is not None) and args.val_during_training:
        val_dataloader = get_val_dataloader(args)
        use_val_ = True
    else:
        use_val_ = False

    best_f1 = 0
    patience = 0
    best_loss = torch.inf
    for t in range(args.n_epochs):
        print(f"Epoch {t}\n-------------------------------")
        train_dataloader = get_train_dataloader(
            args, random_seed_shift=t
        )  # reinitialize dataloader with different negatives each epoch
        model, train_eval = train_epoch(
            model,
            t,
            train_dataloader,
            detection_loss_fn,
            reg_loss_fn,
            class_loss_fn,
            optimizer,
            args,
        )
        if train_eval["loss"] < best_loss:
            best_loss = train_eval["loss"]
            patience = 0
        else:
            patience += 1
        if patience == args.patience:
            break
        train_evals.append(train_eval.copy())
        learning_rates.append(optimizer.param_groups[0]["lr"])

        train_evals_by_epoch = {i: e for i, e in enumerate(train_evals)}
        train_evals_fp = os.path.join(args.experiment_dir, "train_history.yaml")
        with open(train_evals_fp, "w") as f:
            yaml.dump(train_evals_by_epoch, f)

        use_val = use_val_ and t >= args.min_epochs
        if use_val:
            eval_scores = val_epoch(model, t, val_dataloader, args)
            # TODO: maybe plot evals for other pred_types
            val_evals.append(eval_scores["fwd"].copy())
            # plot_eval(train_evals, learning_rates, args, val_evals=val_evals)

            val_evals_by_epoch = {i: e for i, e in enumerate(val_evals)}
            val_evals_fp = os.path.join(args.experiment_dir, "val_history.yaml")
            with open(val_evals_fp, "w") as f:
                yaml.dump(val_evals_by_epoch, f)
        else:
            plot_eval(train_evals, learning_rates, args)
        scheduler.step()

        if use_val and args.early_stopping:
            current_f1 = (
                eval_scores["comb"]["f1"]
                if model.is_bidirectional
                else eval_scores["fwd"]["f1"]
            )
            if args.is_test or (current_f1 > best_f1):
                print("found new best model")
                best_f1 = current_f1

                checkpoint_dict = {
                    "epoch": t,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_evals": train_evals,
                    "val_evals": val_evals,
                }

                torch.save(
                    checkpoint_dict,
                    os.path.join(args.experiment_dir, "model.pt"),
                )

            else:
                checkpoint_dict = {
                    "epoch": t,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_evals": train_evals,
                    "val_evals": val_evals,
                }

                torch.save(
                    checkpoint_dict,
                    os.path.join(args.experiment_dir, "model.pt"),
                )

    print("Done!")

    if best_f1 > 0:
        cp = torch.load(os.path.join(args.experiment_dir, "model.pt"))
        model.load_state_dict(cp["model_state_dict"])

    # resave validation with best model
    if use_val:
        val_epoch(model, args.n_epochs, val_dataloader, args)

    return model


def lf(
    dets: Tensor,
    det_preds: Tensor,
    regs: Tensor,
    reg_preds: Tensor,
    y: Tensor,
    y_preds: Tensor,
    args: argparse.Namespace,
    det_loss_fn: Callable[[Tensor, Tensor], Tensor],
    reg_loss_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    class_loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculate loss function with end masking.

    Parameters
    ----------
    dets : torch.Tensor
        Ground truth detections [batch, time]
    det_preds : torch.Tensor
        Predicted detections [batch, time]
    regs : torch.Tensor
        Ground truth regressions [batch, time]
    reg_preds : torch.Tensor
        Predicted regressions [batch, time]
    y : torch.Tensor
        Ground truth labels [batch, time, n_classes]
    y_preds : torch.Tensor
        Predicted labels (one-hot) [batch, time, n_classes]
    args : argparse.Namespace
        Configuration containing:
        - end_mask_perc: percentage of ends to mask
        - pos_loss_weight: weight for positive samples
        - rho: class loss weight
        - lamb: regression loss weight
    det_loss_fn : callable
        Detection loss function
    reg_loss_fn : callable
        Regression loss function
    class_loss_fn : callable
        Classification loss function

    Returns
    -------
    0-d torch.tensor(), detection_loss
    0-d torch.tensor(), regression_loss
    0-d torch.tensor(), class_loss

    Notes
    -----
    We mask out loss from each end of the clip, so the model isn't forced to
    learn to detect events that are partially cut off. This does not affect
    inference, because during inference we overlap clips at 50%
    """

    end_mask_perc = args.end_mask_perc
    end_mask_dur = int(det_preds.size(1) * end_mask_perc)

    det_preds_clipped = det_preds[:, end_mask_dur:-end_mask_dur]
    dets_clipped = dets[:, end_mask_dur:-end_mask_dur]

    reg_preds_clipped = reg_preds[:, end_mask_dur:-end_mask_dur]
    regs_clipped = regs[:, end_mask_dur:-end_mask_dur]

    y_clipped = y[:, end_mask_dur:-end_mask_dur, :]

    detection_loss = det_loss_fn(
        det_preds_clipped, dets_clipped, pos_loss_weight=args.pos_loss_weight
    )
    reg_loss = reg_loss_fn(reg_preds_clipped, regs_clipped, dets_clipped, y_clipped)
    if len(args.label_set) == 1 and not args.segmentation_based:
        class_loss = torch.tensor(0)
    else:
        y_preds_clipped = y_preds[:, end_mask_dur:-end_mask_dur, :]
        class_loss = class_loss_fn(y_preds_clipped, y_clipped, dets_clipped)
    return detection_loss, reg_loss, class_loss


def train_epoch(
    model: torch.nn.Module,
    t: int,
    dataloader: torch.utils.data.DataLoader,
    detection_loss_fn: Callable[[Tensor, Tensor], Tensor],
    reg_loss_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    class_loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> Tuple[torch.nn.Module, Dict]:
    """Train model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    t : int
        Current epoch number
    dataloader : torch.utils.data.DataLoader
        Training data loader
    detection_loss_fn : callable
        Detection loss function
    reg_loss_fn : callable
        Regression loss function
    class_loss_fn : callable
        Classification loss function
    optimizer : torch.optim.Optimizer
        Model optimizer
    args : argparse.Namespace
        Configuration containing:
        - unfreeze_encoder_epoch: epoch at which to unfreeze encoder
        - rho: class loss weight
        - lamb: regression loss weight
        - display_pbar: how often to update progress bar
        - is_test: whether this is a test run (reduced train and eval)

    Returns
    -------
    model: DetectionModel or similar
        the updated input model
    evals: dict
        training metrics
    """

    model.train()
    if t < args.unfreeze_encoder_epoch:
        model.freeze_encoder()
    else:
        model.unfreeze_encoder()

    evals = {}
    train_loss = 0
    losses = []
    detection_losses = []
    regression_losses = []
    class_losses = []
    rev_train_loss = 0
    rev_losses = []
    rev_detection_losses = []
    rev_regression_losses = []
    rev_class_losses = []

    data_iterator = tqdm.tqdm(dataloader)
    for i, batch in enumerate(data_iterator):
        num_batches_seen = i
        batch = [item.to(device, dtype=torch.float) for item in batch]
        X, d, r, y = batch[:4]
        X, d, r, y = rms_and_mixup(X, d, r, y, True, args)
        probs, regression, class_logits, rev_probs, rev_regression, rev_class_logits = (
            model(X)
        )
        detection_loss, reg_loss, class_loss = lf(
            d,
            probs,
            r,
            regression,
            y,
            class_logits,
            args=args,
            det_loss_fn=detection_loss_fn,
            reg_loss_fn=reg_loss_fn,
            class_loss_fn=class_loss_fn,
        )

        loss = args.rho * class_loss + detection_loss + args.lamb * reg_loss
        train_loss += loss
        losses.append(loss)
        detection_losses.append(detection_loss)
        regression_losses.append(args.lamb * reg_loss)
        class_losses.append(args.rho * class_loss)

        if (i + 1) % args.display_pbar == 0:
            pbar_str = f"loss {torch.tensor(losses).mean():.5f}, "
            pbar_str += f"det {torch.tensor(detection_losses).mean():.5f}, "
            pbar_str += f"reg {torch.tensor(regression_losses).mean():.5f}, "
            pbar_str += f"class {torch.tensor(class_losses).mean():.5f}"
            losses = []
            detection_losses = []
            regression_losses = []
            class_losses = []

        if model.is_bidirectional:
            assert all(
                x is not None for x in [rev_probs, rev_regression, rev_class_logits]
            )
            rev_d, rev_r, rev_y = batch[4:]
            _, rev_d, rev_r, rev_y = rms_and_mixup(X, rev_d, rev_r, rev_y, True, args)

            rev_detection_loss, rev_reg_loss, rev_class_loss = lf(
                rev_d,
                rev_probs,
                rev_r,
                rev_regression,
                rev_y,
                rev_class_logits,
                args=args,
                det_loss_fn=detection_loss_fn,
                reg_loss_fn=reg_loss_fn,
                class_loss_fn=class_loss_fn,
            )
            rev_loss = (
                args.rho * rev_class_loss
                + rev_detection_loss
                + args.lamb * rev_reg_loss
            )
            rev_train_loss += rev_loss
            rev_losses.append(rev_loss)
            rev_detection_losses.append(rev_detection_loss)
            rev_regression_losses.append(args.lamb * rev_reg_loss)
            rev_class_losses.append(args.rho * rev_class_loss)
            loss = (loss + rev_loss) / 2

            if (i + 1) % args.display_pbar == 0:
                pbar_str += f" revloss {torch.tensor(rev_losses).mean():.5f}, "
                pbar_str += f"revdet {torch.tensor(rev_detection_losses).mean():.5f}, "
                pbar_str += f"revreg {torch.tensor(rev_regression_losses).mean():.5f}, "
                pbar_str += f"revclass {torch.tensor(rev_class_losses).mean():.5f}"
                rev_train_loss = 0
                rev_losses = []
                rev_detection_losses = []
                rev_regression_losses = []
                rev_class_losses = []
        else:
            assert all(x is None for x in [rev_probs, rev_regression, rev_class_logits])

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if (i + 1) % args.display_pbar == 0:
            data_iterator.set_description(pbar_str)

        if args.is_test and i == 5:
            break

    train_loss = train_loss / num_batches_seen
    evals["loss"] = float(train_loss)

    print(f"Epoch {t} | Train loss: {train_loss:1.5f}")
    return model, evals


def val_epoch(
    model: torch.nn.Module,
    t: int,
    dataloader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics on model for one epoch on the given dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        Model to validate
    t : int
        Current epoch number
    dataloader : torch.utils.data.DataLoader
        Validation data loader
    args : argparse.Namespace
        Configuration containing:
        - detection_threshold: detection threshold
        - experiment_output_dir: output directory
        - model_selection_iou: IoU threshold for matching during evaluation
        - model_selection_class_threshold: class threshold
        - comb_discard_thresh: discard threshold
        - label_mapping: label mappings
        - unknown_label: unknown label name
        - bidirectional: whether model is bidirectional
        - label_set: set of labels

    Returns
    -------
    evals: dict
        metrics
    """

    model.eval()

    manifests = predict_and_generate_manifest(model, dataloader, args, verbose=False)
    manifest = manifests[args.detection_threshold]
    e, _ = evaluate_based_on_manifest(
        manifest,
        output_dir=args.experiment_output_dir,
        iou=args.model_selection_iou,
        det_thresh=args.detection_threshold,
        class_threshold=args.model_selection_class_threshold,
        comb_discard_threshold=args.comb_discard_thresh,
        label_mapping=args.label_mapping,
        unknown_label=args.unknown_label,
        bidirectional=args.bidirectional,
    )

    metrics_to_print = [
        "precision",
        "recall",
        "f1",
        "precision_seg",
        "recall_seg",
        "f1_seg",
    ]
    print(f"Epoch {t} | val@{args.model_selection_iou}IoU:")
    evals = {}
    for pt in e.keys():
        evals[pt] = {k: [] for k in metrics_to_print}
        for k in metrics_to_print:
            for labelname in args.label_set:
                m = e[pt]["summary"][labelname][k]
                evals[pt][k].append(m)
            evals[pt][k] = float(np.mean(evals[pt][k]))

        for m in metrics_to_print:
            score = evals[pt][m]
            print(f"{pt}-{m}: {score:1.4f}", end=" ")
        print()
    return evals


def modified_focal_loss(
    pred: Tensor, gt: Tensor, pos_loss_weight: float = 1.0
) -> Tensor:
    """Modified focal loss for detection.

    Parameters
    ----------
    pred : torch.Tensor
        Predictions [batch, time]
    gt : torch.Tensor
        Ground truth [batch, time]
    pos_loss_weight : float, optional
        Weight for positive samples (default: 1)

    Returns
    -------
    torch.Tensor
        Scalar loss value

    Notes
    -----
    Modified from https://github.com/xingyizhou/CenterNet/blob/2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/models/losses.py
    """

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * pos_loss_weight
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    loss = -1.0 * (neg_loss + pos_loss)

    loss = loss.mean()
    return loss


def masked_reg_loss(
    regression: Tensor,
    r: Tensor,
    d: Tensor,
    y: Tensor,
    class_weights: Optional[Tensor] = None,
) -> Tensor:
    """Masked regression loss.

    Parameters
    ----------
    regression : torch.Tensor
        Predicted regression [batch, time]
    r : torch.Tensor
        Ground truth regression [batch, time]
    d : torch.Tensor
        Detection mask [batch, time]
    y : torch.Tensor
        Class labels [batch, time, n_classes]
    class_weights : torch.Tensor, optional
        How to weight loss by class [n_classes]
        default: None, means no reweighting, i.e. uniform

    Returns
    -------
    torch.Tensor
        Scalar loss value
    """

    reg_loss = F.l1_loss(regression, r, reduction="none")
    mask = d.eq(1).float()

    reg_loss = reg_loss * mask

    if class_weights is not None:
        y = rearrange(y, "b t c -> b c t")

        high_prob = torch.amax(y, dim=1)
        knowns = high_prob.eq(1).float()
        unknowns = high_prob.lt(1).float()

        reg_loss_unknowns = reg_loss * unknowns

        class_weights = torch.reshape(class_weights, (1, -1, 1))
        class_weights = y * class_weights
        class_weights = torch.amax(class_weights, dim=1)

        reg_loss_knowns = reg_loss * knowns * class_weights

        reg_loss = reg_loss_unknowns + reg_loss_knowns

    reg_loss = torch.sum(reg_loss)
    n_pos = mask.sum()

    reg_loss = reg_loss / torch.clip(n_pos, min=1)
    # equivalent to the below without forcing torch synchronisation
    # if n_pos>0:
    # reg_loss2 = reg_loss2 / n_pos

    return reg_loss


def masked_classification_loss(
    class_logits: Tensor, y: Tensor, d: Tensor, class_weights: Optional[Tensor] = None
) -> Tensor:
    """Masked classification loss.

    Parameters
    ----------
    class_logits : torch.Tensor
        Class logits [batch, time, n_classes]
    y : torch.Tensor
        Ground truth labels [batch, time, n_classes]
    d : torch.Tensor
        Detection mask [batch, time]
    class_weights : torch.Tensor, optional
        Class weights [n_classes] (default: None)

    Returns
    -------
    torch.Tensor
        Scalar loss value
    """

    class_logits = rearrange(class_logits, "b t c -> b c t")
    y = rearrange(y, "b t c -> b c t")

    high_prob = torch.amax(y, dim=1)
    knowns = high_prob.eq(1).float()
    unknowns = high_prob.lt(1).float()

    mask = d.eq(1).float()  # mask out time steps where no event is present

    known_class_loss = F.cross_entropy(
        class_logits, y, weight=class_weights, reduction="none"
    )
    known_class_loss = known_class_loss * mask * knowns
    known_class_loss = torch.sum(known_class_loss)

    unknown_class_loss = F.cross_entropy(class_logits, y, weight=None, reduction="none")
    unknown_class_loss = unknown_class_loss * mask * unknowns
    unknown_class_loss = torch.sum(unknown_class_loss)

    class_loss = known_class_loss + unknown_class_loss
    n_pos = mask.sum()

    # class_loss2 = class_loss.clone()
    class_loss = class_loss / torch.clip(n_pos, min=1)
    # if n_pos>0:
    # class_loss2 = class_loss2 / n_pos
    # assert class_loss==class_loss

    return class_loss


def segmentation_loss(
    class_logits: Tensor, y: Tensor, d: Tensor, class_weights: Optional[Tensor] = None
) -> Tensor:
    """Segmentation loss using focal loss.

    Parameters
    ----------
    class_logits : torch.Tensor
        Class logits [batch, time, n_classes]
    y : torch.Tensor
        Ground truth labels [batch, time, n_classes]
    d : torch.Tensor
        Detection mask [batch, time]
    class_weights : torch.Tensor, optional
        Unused, for API compatibility (default: None)

    Returns
    -------
    torch.Tensor
        Scalar loss value
    """

    default_focal_loss = torchvision.ops.sigmoid_focal_loss(
        class_logits, y, reduction="mean"
    )
    return default_focal_loss


def get_class_loss_fn(args: argparse.Namespace) -> Callable:
    """Get classification loss function based on args.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration containing:
        - segmentation_based: whether to use segmentation
        - experiment_dir: experiment directory
        - recompute_class_weights: whether to recompute weights

    Returns
    -------
    callable
        Classification loss function
    """

    if hasattr(args, "segmentation_based") and args.segmentation_based:
        return segmentation_loss
    elif (
        os.path.exists(cache_fp := f"{args.experiment_dir}/cached_class_weights.pt")
        and not args.recompute_class_weights
    ):
        class_weights = torch.load(cache_fp).to(device)
    else:
        dataloader_temp = get_train_dataloader(args, random_seed_shift=0)
        class_proportions = dataloader_temp.dataset.get_class_proportions()
        class_weights = 1.0 / (class_proportions + 1e-6)
        class_weights = class_weights * (
            class_proportions > 0
        )  # ignore weights for unrepresented classes

        class_weights = (
            1.0 / (np.mean(class_weights) + 1e-6)
        ) * class_weights  # normalize so average weight = 1
        print(f"Using class weights {class_weights}")

        class_weights = torch.Tensor(class_weights).to(device)
    return partial(masked_classification_loss, class_weights=class_weights)


def get_reg_loss_fn(args: argparse.Namespace) -> Callable:
    """Get regression loss function based on args.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration containing:
        - segmentation_based: whether to use segmentation
        - experiment_dir: experiment directory
        - recompute_class_weights: whether to recompute weights

    Returns
    -------
    callable
        Regression loss function
    """

    if hasattr(args, "segmentation_based") and args.segmentation_based:

        def zrl(
            regression: Tensor,
            r: Tensor,
            d: Tensor,
            y: Tensor,
            class_weights: Optional[Tensor] = None,
        ) -> Tensor:
            """
            zero regression loss placeholder

            Returns
            ------
            Tensor which is just zero
            """
            return torch.tensor(0.0)

        return zrl
    elif (
        os.path.exists(cache_fp := f"{args.experiment_dir}/cached_class_weights.pt")
        and not args.recompute_class_weights
    ):
        class_weights = torch.load(cache_fp).to(device)
    else:
        dataloader_temp = get_train_dataloader(args, random_seed_shift=0)
        class_proportions = dataloader_temp.dataset.get_class_proportions()
        class_weights = 1.0 / (class_proportions + 1e-6)
        class_weights = class_weights * (
            class_proportions > 0
        )  # ignore weights for unrepresented classes

        class_weights = (
            1.0 / (np.mean(class_weights) + 1e-6)
        ) * class_weights  # normalize so average weight = 1

        class_weights = torch.Tensor(class_weights).to(device)
        torch.save(class_weights, cache_fp)

    return partial(masked_reg_loss, class_weights=class_weights)


def get_detection_loss_fn(args: argparse.Namespace) -> Callable:
    """Get detection loss function based on args.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration containing:
        - segmentation_based: whether to use segmentation

    Returns
    -------
    callable
        Detection loss function
    """
    if hasattr(args, "segmentation_based") and args.segmentation_based:

        def zdl(pred: Tensor, gt: Tensor, pos_loss_weight: float = 1.0) -> Tensor:
            """
            zero detection loss placeholder
            Returns
            --------
            Tensor which is just zero
            """
            return torch.tensor(0.0)

        return zdl
    else:
        return modified_focal_loss

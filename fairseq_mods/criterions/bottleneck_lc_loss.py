# Copyright (c) Facebook, Inc. and its affiliates.
# Bottleneck_LC: Data-Dependent Prior for NAT (Model Level).
# Paper: Widening The Bottleneck of Lexical Choice for Non-Autoregressive Translation.

import math
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch import Tensor

from . import nat_loss as nat_loss_module


@register_criterion("bottleneck_lc_loss")
class BottleneckLCCriterion(nat_loss_module.LabelSmoothedDualImitationCriterion):
    """
    NAT loss + data-dependent prior (KL with Q from raw data).
    L = (1 - λ) * L_NAT + λ * L_prior,  λ = log(I/(2(i+1)))/log(I/2) for i <= I/2 else 0.
    Q is combined WAD (word alignment distribution) + SDD (self-distilled from raw-pretrained NAT).
    """

    def __init__(self, task, label_smoothing, prior_weight_path, prior_src_vocab_path,
                 prior_tgt_vocab_path, prior_total_steps=300000, prior_temperature=2.0):
        super().__init__(task, label_smoothing)
        self.prior_weight_path = prior_weight_path
        self.prior_src_vocab_path = prior_src_vocab_path
        self.prior_tgt_vocab_path = prior_tgt_vocab_path
        self.prior_total_steps = prior_total_steps
        self.prior_temperature = prior_temperature

        assert self.prior_weight_path is not None
        assert self.prior_src_vocab_path is not None
        assert self.prior_tgt_vocab_path is not None

        import numpy as np
        import h5py
        with h5py.File(self.prior_weight_path, "r") as f:
            self.prior_weights = torch.from_numpy(np.array(f["weights"])).float()
        with open(self.prior_src_vocab_path, "r", encoding="utf-8") as f:
            self.prior_src_vocab = [line.strip() for line in f if line.strip()]
        with open(self.prior_tgt_vocab_path, "r", encoding="utf-8") as f:
            self.prior_tgt_vocab = [line.strip() for line in f if line.strip()]

        assert len(task.source_dictionary) == self.prior_weights.shape[0]
        assert len(task.target_dictionary) == self.prior_weights.shape[1]

    @staticmethod
    def add_args(parser):
        nat_loss_module.LabelSmoothedDualImitationCriterion.add_args(parser)
        parser.add_argument("--prior-weight-path", type=str, help="Path to prior weight .h5 (WAD+SDD)")
        parser.add_argument("--prior-src-vocab-path", type=str, help="Source vocab for prior (one token per line)")
        parser.add_argument("--prior-tgt-vocab-path", type=str, help="Target vocab for prior")
        parser.add_argument("--prior-total-steps", type=int, default=300000, help="I in λ decay")
        parser.add_argument("--prior-temperature", type=float, default=2.0, help="Temperature τ for WAD softmax")

    def _lambda(self, num_updates):
        """λ(i) = log(I/(2(i+1)))/log(I/2) for i <= I/2 else 0."""
        I = self.prior_total_steps
        if num_updates >= I // 2:
            return 0.0
        return max(0.0, math.log(I / (2 * (num_updates + 1))) / math.log(I / 2))

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0,
        prior_weights_batch=None, num_updates=0
    ):
        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0.0, device=outputs.device)
            prior_loss = torch.tensor(0.0, device=outputs.device)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")
                nll_loss = mean_ds(losses)
                if label_smoothing > 0:
                    loss = nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                else:
                    loss = nll_loss

                # Prior KL: L_prior = - sum_e KL(Q||P^M) => we minimize KL(Q||P), so -E_q[log P] + const
                lam = self._lambda(num_updates)
                prior_loss = torch.tensor(0.0, device=logits.device)
                if lam > 0 and prior_weights_batch is not None:
                    prior_weights_batch = prior_weights_batch.to(logits.device)
                    if prior_weights_batch.dim() == 2:
                        prior_loss = F.kl_div(logits, prior_weights_batch.clamp(min=1e-8), reduction="batchmean")
                    loss = loss * (1.0 - lam) + prior_loss * lam
            else:
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none").sum(-1)
                nll_loss = mean_ds(losses)
                if label_smoothing > 0:
                    loss = nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                else:
                    loss = nll_loss
                prior_loss = torch.tensor(0.0, device=logits.device)

        loss = loss * factor
        prior_loss_val = prior_loss if isinstance(prior_loss, Tensor) else torch.tensor(0.0, device=loss.device)
        return {
            "name": name, "loss": loss, "nll_loss": nll_loss,
            "prior_loss": prior_loss_val,
            "factor": factor,
        }

    def forward(self, model, sample, reduce=True):
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        tgt_tokens = sample["target"]
        prev_output_tokens = sample["prev_target"]

        num_updates = model.get_num_updates() if hasattr(model, "get_num_updates") else 0

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        losses, nll_losses, prior_losses = [], [], []

        # Build prior weights for this batch: linear alignment tgt_pos j -> src_pos
        prior_weights_batch = None
        if self.prior_weights.device != src_tokens.device:
            self.prior_weights = self.prior_weights.to(src_tokens.device)
        B, S, T = src_tokens.size(0), src_tokens.size(1), tgt_tokens.size(1)
        # Linear alignment: tgt pos j -> src pos = j * S // T (rough)
        src_pos = torch.arange(T, device=src_tokens.device).mul(S).div(max(T, 1)).clamp(max=S - 1)
        aligned_src = src_tokens[:, src_pos]  # [B, T]
        prior_weights_batch = self.prior_weights[aligned_src]  # [B, T, V_tgt]

        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                mask = outputs[obj].get("mask", None)
                out = outputs[obj].get("out")
                tgt = outputs[obj].get("tgt")
                if mask is not None and prior_weights_batch is not None:
                    # Flatten and select masked positions: [B,T,V] -> select mask
                    pw = prior_weights_batch[mask]  # [num_masked, V_tgt]
                    pw = pw / (pw.sum(-1, keepdim=True) + 1e-8)
                else:
                    pw = None
                _losses = self._compute_loss(
                    out, tgt, mask, outputs[obj].get("ls", 0.0),
                    name=obj + "-loss", factor=outputs[obj].get("factor", 1.0),
                    prior_weights_batch=pw, num_updates=num_updates,
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"), name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
                _losses["nll_loss"] = _losses.get("loss", torch.tensor(0.0))
                _losses["prior_loss"] = torch.tensor(0.0, device=_losses["loss"].device)
            losses.append(_losses)
            nll_losses.append(_losses.get("nll_loss", 0.0))
            prior_losses.append(_losses.get("prior_loss", 0.0))

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(nll_losses) if nll_losses else loss.new_tensor(0.0)
        prior_loss = sum(prior_losses) if prior_losses else loss.new_tensor(0.0)
        sample_size = 1
        logging_output = {
            "loss": loss.data, "nll_loss": nll_loss.data, "prior_loss": prior_loss.data,
            "ntokens": ntokens, "nsentences": nsentences, "sample_size": sample_size,
        }
        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"]) if reduce else l["loss"].data / l["factor"]
            )
        return loss, sample_size, logging_output

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor, "nll_loss": loss, "prior_loss": loss.new_tensor(0.0)}

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        nat_loss_module.LabelSmoothedDualImitationCriterion.reduce_metrics(logging_outputs)
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        prior_loss = utils.item(sum(log.get("prior_loss", 0) for log in logging_outputs))
        if sample_size > 0 and prior_loss != 0:
            metrics.log_scalar("prior_loss", prior_loss / sample_size / math.log(2), sample_size, round=3)

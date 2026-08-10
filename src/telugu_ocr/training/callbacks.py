"""Trainer callbacks shared by the training loops."""

from __future__ import annotations

import torch
from transformers import TrainerCallback

from src.telugu_ocr.metrics.errors import compute_cer

# How many images per slice the CER eval decodes, and how many to print. Generation is
# B=1 and slow, so this is a monitoring sample, not the whole eval set.
CER_EVAL_SAMPLES = 200
CER_PRINT_K = 5


class GradNormAlert(TrainerCallback):
    """Print a warning when the gradient norm spikes above `threshold`.

    A spike is the visible symptom of a bad batch, a too-high LR, or fp16 overflow --
    all of which otherwise show up only later, as a loss curve that flattened for
    reasons nobody can reconstruct. Printing at the moment it happens ties the spike to
    a step number.
    """

    def __init__(self, threshold: float = 10.0):
        self.threshold = threshold

    def on_log(self, args, state, control, logs=None, **kwargs):
        gn = (logs or {}).get("grad_norm")
        if gn is not None and gn > self.threshold:
            print(f"[ALERT] grad_norm={gn:.2f} > {self.threshold} at step {state.global_step}")


class CEREvalCallback(TrainerCallback):
    """CER during evaluate(), reported PER SLICE, for the model's TWO decoders:

      TEXT decoder (autoregressive LM), in two flavours:
        * generation CER     (``eval_<slice>_cer``)     — free-running B=1 greedy decode
        * teacher-forced CER (``eval_<slice>_tf_cer``)  — one forward with the ground-truth
                                                          tokens fed in, argmax next-token
      CTC decoder (the encoder's CTC head):
        * CTC CER            (``eval_<slice>_ctc_cer``) — per-frame argmax -> collapse
                                                          repeats -> drop blanks (no ref)

    All three are computed on the SAME up-to-CER_EVAL_SAMPLES images per slice, so they are
    directly comparable. Macro-averages ``eval_cer`` / ``eval_tf_cer`` / ``eval_ctc_cer``
    over slices are added too. All metrics go into the metrics dict and to wandb.

    Early stopping was removed, so these are purely for monitoring; everything runs on
    rank 0 only (see on_evaluate).
    """

    def __init__(self, model, eval_slices, preprocessor, tokenizer,
                 image_col="image", text_col="text",
                 report_tf=False, report_ctc=False, cer_eval_samples=CER_EVAL_SAMPLES,
                 cer_print_k=CER_PRINT_K):
        # report_tf / report_ctc default OFF, which is exactly stage 1: generation CER
        # only. Stage 2 turns both on to watch the text and CTC decoders diverge.
        self.report_tf = report_tf
        self.report_ctc = report_ctc
        self.cer_eval_samples = cer_eval_samples
        self.cer_print_k = cer_print_k
        self.model = model
        self.slices = eval_slices  # {name: list of raw rows with image/text cols}
        self.prep = preprocessor   # clean preprocessor (no augmentation)
        self.tok = tokenizer
        self.image_col = image_col
        self.text_col = text_col

    @torch.no_grad()
    def _teacher_forced_pred(self, bridged, cross_key_mask, ref_ids, device):
        """Teacher-forced through the TEXT decoder, REUSING a precomputed (bridged) encoder
        output — no re-encode. logits[t] predicts token t+1, so argmax(logits[:-1]) are the
        predictions for ref positions 1..T-1. Returns the decoded hypothesis string."""
        ids = torch.as_tensor([ref_ids], dtype=torch.long, device=device)  # (1, T)
        out = self.model.decoder_model(ids, bridged, text_padding_mask=None,
                                       img_text_padding_mask=cross_key_mask, labels=None)
        pred = out.logits[0, :-1].argmax(dim=-1)         # (T-1,)
        return self.tok.decode(pred.tolist(), skip_special_tokens=True)

    @torch.no_grad()
    def _ctc_pred(self, enc_raw):
        """CTC greedy decode from the encoder's CTC head (the second decoder), REUSING the
        precomputed RAW (pre-bridge) encoder output: per-frame argmax -> collapse consecutive
        repeats -> drop blanks -> decode. Collapse matches the standalone CTC encoder's eval."""
        logits = self.model.encoder_model.ctc_head(enc_raw)       # (1, T, C=vocab+1)
        frame_ids = logits[0].argmax(dim=-1).tolist()
        blank = self.model.encoder_model.blank_id
        collapsed, prev = [], None
        for t in frame_ids:
            if t != prev and t != blank:
                collapsed.append(t)
            prev = t
        return self.tok.decode(collapsed, skip_special_tokens=True)

    def _slice_cer(self, rows, device, ctx, image_col, text_col):
        n = min(self.cer_eval_samples, len(rows))
        gen_preds, tf_preds, ctc_preds, refs = [], [], [], []
        for i in range(n):
            ex = rows[i]
            pix = self.prep(ex[image_col]).unsqueeze(0).to(device)  # (1, 1, H, W)
            ref_ids = self.tok.encode(ex[text_col])
            cap = min(ctx - 1, int(1.5 * len(ref_ids)) + 10)        # length-aware cap

            # ---- ONE encoder forward per image, shared by all three decoders ----
            # B=1 un-padded -> input_lengths=None, so key_padding_mask (and cross_key_mask)
            # is None. enc_raw feeds the CTC head; bridged (enc_raw -> enc_to_dec) feeds the
            # text decoder's cross-attention. Mirrors EncoderDecoder.forward exactly.
            with torch.no_grad():
                enc_raw, key_padding_mask = self.model.encoder_model.encode(pix, None)  # (1, T, D)
                bridged = self.model.enc_to_dec(enc_raw)                                # (1, T, dec_dim)
            cross_key_mask = (None if key_padding_mask is None
                              else (~key_padding_mask).unsqueeze(1).unsqueeze(2))

            gen_ids = self.model.generate(                                    # text decoder (generation)
                pix, self.tok.bos_token_id, self.tok.eos_token_id,
                max_new_tokens=cap, enc_out=bridged, cross_key_mask=cross_key_mask)
            gen_preds.append(self.tok.decode(gen_ids, skip_special_tokens=True))
            if self.report_tf:                                                # text decoder (teacher-forced)
                tf_preds.append(
                    self._teacher_forced_pred(bridged, cross_key_mask, ref_ids, device))
            if self.report_ctc:                                               # CTC decoder
                ctc_preds.append(self._ctc_pred(enc_raw))
            refs.append(ex[text_col])
        out = {"gen_cer": compute_cer(gen_preds, refs), "gen": gen_preds, "refs": refs}
        if self.report_tf:
            out["tf_cer"] = compute_cer(tf_preds, refs)
            out["tf"] = tf_preds
        if self.report_ctc:
            out["ctc_cer"] = compute_cer(ctc_preds, refs)
            out["ctc"] = ctc_preds
        return out

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # RANK-0 ONLY. This CER is monitoring-only (no early stopping / best-model
        # selection reads it), and the generation is a long, collective-free stretch of
        # work. Running it independently on every rank lets the ranks desync — one
        # finishes and hits the next NCCL barrier while the other is still decoding — and
        # blows past ddp_timeout (the 30-min ALLGATHER watchdog that killed the run).
        # Doing it on rank 0 alone keeps control flow deterministic; the other ranks skip
        # to the following save/step barrier and wait (ddp_timeout is raised to cover it).
        if not state.is_world_process_zero:
            return
        try:
            self.model.eval()
            device = next(self.model.parameters()).device
            ctx = self.model.decoder_model.positional_encodings.shape[0]
            image_col, text_col = self.image_col, self.text_col
            per_gen, per_tf, per_ctc = {}, {}, {}
            for name, rows in self.slices.items():
                if not rows:
                    continue
                r = self._slice_cer(rows, device, ctx, image_col, text_col)
                per_gen[name] = r["gen_cer"]
                per_tf[name] = r["tf_cer"]
                per_ctc[name] = r["ctc_cer"]
                if metrics is not None:
                    metrics[f"eval_{name}_cer"] = r["gen_cer"]          # text decoder (generation)
                    metrics[f"eval_{name}_tf_cer"] = r["tf_cer"]        # text decoder (teacher-forced)
                    metrics[f"eval_{name}_ctc_cer"] = r["ctc_cer"]      # CTC decoder
                print(f"[eval] step {state.global_step}  {name} genCER={r['gen_cer']:.4f} "
                      f"tfCER={r['tf_cer']:.4f} ctcCER={r['ctc_cer']:.4f} "
                      f"(n={min(CER_EVAL_SAMPLES, len(rows))})", flush=True)
                for gp, tp, cp, rf in list(zip(r["gen"], r["tf"], r["ctc"], r["refs"]))[:self.cer_print_k]:
                    print(f"    [{name}] ref: {rf!r}")
                    print(f"    [{name}] gen: {gp!r}")
                    print(f"    [{name}] tf : {tp!r}")
                    print(f"    [{name}] ctc: {cp!r}")
            if per_gen:
                macro = sum(per_gen.values()) / len(per_gen)          # macro-avg text-decoder generation CER
                macro_tf = sum(per_tf.values()) / len(per_tf)         # macro-avg text-decoder teacher-forced CER
                macro_ctc = sum(per_ctc.values()) / len(per_ctc)      # macro-avg CTC-decoder CER
                if metrics is not None:
                    metrics["eval_cer"] = macro
                    metrics["eval_tf_cer"] = macro_tf
                    metrics["eval_ctc_cer"] = macro_ctc
                print(f"[eval] step {state.global_step}  macro genCER={macro:.4f} "
                      f"tfCER={macro_tf:.4f} ctcCER={macro_ctc:.4f}", flush=True)
                try:
                    import wandb
                    wandb.log({**{f"eval_{k}_cer": v for k, v in per_gen.items()},
                               **{f"eval_{k}_tf_cer": v for k, v in per_tf.items()},
                               **{f"eval_{k}_ctc_cer": v for k, v in per_ctc.items()},
                               "eval_cer": macro, "eval_tf_cer": macro_tf,
                               "eval_ctc_cer": macro_ctc}, step=state.global_step)
                except Exception:
                    pass
        except Exception as exc:
            print(f"[CER] failed at step {state.global_step}: {exc}", flush=True)
        finally:
            self.model.train()

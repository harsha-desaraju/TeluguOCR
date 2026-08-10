"""The encoder-decoder Trainer, shared by both fine-tuning stages.

There were two of these, and the difference between them was one behaviour that only
stage 1 wants: running the TRAINING forward with the model in .eval() mode.

That is not a hack. In stage 1 the encoder and the pretrained decoder layers are frozen
and only the new adapters train, so the frozen backbone's dropout and stochastic depth
would be injecting noise into the very features the adapters are trying to learn from.
`eval()` turns those off without stopping gradients, so the adapters still train.

It has to be done inside compute_loss because HF's Trainer.training_step calls
model.train() at the start of every step -- a one-off eval() before train() would not
survive the first step.

Stage 2 unfreezes everything, at which point dropout and DropPath are wanted again for
regularisation, so it leaves the flag off.
"""

from __future__ import annotations

from transformers import Trainer

from src.telugu_ocr.training.optim import build_scheduler


class EncoderDecoderTrainer(Trainer):
    """Trainer that builds the warmup -> cosine scheduler using the step count Trainer
    computes, so nothing here has to reason about DDP or epochs itself.

    force_eval_during_train  stage 1: True. See the module docstring.
    min_lr                   stage 1 passes an absolute floor; stage 2 leaves it None
                             because its optimizer attached a per-tier 'min_lr' to each
                             param group.
    """

    def __init__(self, *args, warmup_steps=1500, min_lr=None,
                 force_eval_during_train=False, **kwargs):
        self._warmup_steps = warmup_steps
        self._min_lr = min_lr
        self.force_eval_during_train = force_eval_during_train
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            self.lr_scheduler = build_scheduler(
                optimizer or self.optimizer, self._warmup_steps, num_training_steps,
                self._min_lr)
        return self.lr_scheduler

    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.force_eval_during_train:
            model.eval()
        return super().compute_loss(model, inputs, *args, **kwargs)

    def log(self, logs, *args, **kwargs):
        # Surface the CE / CTC breakdown behind the combined training loss so it reaches
        # the console and wandb alongside `loss`. Values are the last micro-batch's
        # components -- enough to watch the two terms trade off while tuning
        # ctc_loss_weight. Train logs only (gated on 'loss'), never eval logs. Inert in
        # stage 1, where the model never sets _ctc_loss.
        if "loss" in logs:
            core = self.model.module if hasattr(self.model, "module") else self.model
            ce = getattr(core, "_ce_loss", None)
            ctc = getattr(core, "_ctc_loss", None)
            if ce is not None:
                logs["ce_loss"] = float(ce)
            if ctc is not None:
                logs["ctc_loss"] = float(ctc)
        return super().log(logs, *args, **kwargs)

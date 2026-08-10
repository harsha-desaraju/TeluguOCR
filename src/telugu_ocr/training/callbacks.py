"""Trainer callbacks shared by the training loops."""

from __future__ import annotations

from transformers import TrainerCallback


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

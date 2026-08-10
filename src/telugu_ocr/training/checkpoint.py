"""Resume-point discovery, shared by the training loops."""

from __future__ import annotations

import os

from transformers.trainer_utils import get_last_checkpoint


def find_last_checkpoint(output_dir, prev_run_dir=None):
    """Newest checkpoint in `output_dir`, else in `prev_run_dir`, else None.

    Two directories because a run resumed on a fresh machine writes to a new output_dir
    while the checkpoint it must continue from sits in the previous run's. Order matters:
    the current output_dir wins, so a resumed run does not jump back to an older tree.
    """
    for d in (output_dir, prev_run_dir):
        if d and os.path.isdir(d):
            ckpt = get_last_checkpoint(d)
            if ckpt is not None:
                return ckpt
    return None

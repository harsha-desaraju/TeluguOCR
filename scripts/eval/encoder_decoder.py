"""Evaluate the stage-2 encoder-decoder on REAL scanned lines (wikisource_lines_eval).

For every line image, ONE encoder forward feeds three decoders, and each is scored
against the human-reviewed reference:

  1. CTC greedy        — per-frame argmax on the encoder's CTC head -> collapse
                         repeats -> drop blanks. No language knowledge, monotonic
                         alignment by construction.
  2. LM beam search    — KV-cached beam search over the cross-attention text decoder.
                         Picks the hypothesis with the best (optionally length-
                         normalized) attention log-prob.
  3. Joint rescoring   — the beam's n-best list re-ranked by
                             lambda * logP_ctc(y|x) + (1 - lambda) * logP_attn(y|x)
                         where logP_ctc comes from the CTC forward algorithm
                         (F.ctc_loss) over the SAME encoder frames. Swept over a
                         lambda grid; lambda=0 is pure-attention rescoring and
                         lambda=1 ranks the beam purely by CTC. This imports the CTC
                         head's monotonic-alignment discipline into the LM decode
                         (the hybrid CTC/attention recipe from ASR).

CER here is code-point Levenshtein / reference length (same convention as the
training-run CER logs), aggregated corpus-wide (sum of edits / sum of ref lengths).

Run from the repo root:  .venv/bin/python3 -m scripts.eval.encoder_decoder
All knobs live in the CONFIG block of __main__.
"""

import json
import os
import random
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.telugu_ocr.models.encoder_decoder import EncoderDecoder
from src.telugu_ocr.models.image_encoder import CTCEncoderConfig
from src.telugu_ocr.models.text_decoder import GPTConfig
from src.telugu_ocr.tokenizer.grapheme import TeluguGraphemeTokenizer


# ============================================================================
# Preprocessing — matches training's eval path: grayscale -> ToTensor ->
# Normalize(0.5, 0.5). The wikisource crops are already 64px tall; anything else
# is resized to height 64 (aspect preserved) and width is padded WITH PAPER
# (white) up to a multiple of the conv stem's downsample factor.
# ============================================================================
_TO_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def preprocess_image(img: Image.Image, image_height: int = 64, downsample: int = 8) -> torch.Tensor:
    img = img.convert("L")
    if img.height != image_height:
        new_w = max(downsample, round(img.width * image_height / img.height))
        img = img.resize((new_w, image_height), Image.LANCZOS)
    if img.width % downsample != 0:
        pad = downsample - img.width % downsample
        padded = Image.new("L", (img.width + pad, image_height), color=255)
        padded.paste(img, (0, 0))
        img = padded
    return _TO_TENSOR(img)  # (1, H, W)


# ============================================================================
# Metric — code-point Levenshtein, same convention as the training CER logs.
# ============================================================================
def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


# ============================================================================
# Decoders
# ============================================================================
@torch.no_grad()
def ctc_greedy_ids(ctc_log_probs: torch.Tensor, blank_id: int) -> list:
    """(T, C) frame log-probs -> argmax -> collapse consecutive repeats -> drop blanks."""
    frame_ids = ctc_log_probs.argmax(dim=-1).tolist()
    out, prev = [], None
    for t in frame_ids:
        if t != prev and t != blank_id:
            out.append(t)
        prev = t
    return out


def _reorder_cache(past, idx: torch.Tensor):
    """Select/duplicate the batch dimension of every cached K/V after a beam reshuffle."""
    out = []
    for self_kv, cross_kv in past:
        s = (self_kv[0].index_select(0, idx), self_kv[1].index_select(0, idx))
        c = None if cross_kv is None else (
            cross_kv[0].index_select(0, idx), cross_kv[1].index_select(0, idx))
        out.append((s, c))
    return out


@torch.no_grad()
def beam_search(model, bridged, bos_id, eos_id, beam_width, max_new_tokens):
    """KV-cached beam search over the text decoder for a single (un-padded) image.

    bridged: (1, T, dec_dim) encoder output already through enc_to_dec.
    Returns the n-best list as dicts {"ids": [token ids, no BOS/EOS], "logp": float}
    where logp is the summed attention log-prob INCLUDING the EOS step (hypotheses
    that never emitted EOS within the budget are closed as-is).
    """
    device = bridged.device
    ctx = model.decoder_model.positional_encodings.shape[0]
    max_steps = min(max_new_tokens, ctx - 1)

    # Step 1: one BOS forward seeds the beams (all beams would be identical anyway).
    bos = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
    out, past = model.decoder_model(bos, bridged, None, None, None,
                                    past_kvs=None, use_cache=True)
    logprobs = F.log_softmax(out.logits[0, -1].float(), dim=-1)
    top_scores, top_tokens = logprobs.topk(beam_width)

    finished = []
    beam_ids, beam_scores = [], []
    for s, t in zip(top_scores.tolist(), top_tokens.tolist()):
        if t == eos_id:
            finished.append({"ids": [], "logp": s})
        else:
            beam_ids.append([t])
            beam_scores.append(s)
    if not beam_ids:
        return finished
    past = _reorder_cache(past, torch.zeros(len(beam_ids), dtype=torch.long, device=device))
    beam_scores = torch.tensor(beam_scores, device=device)

    for _ in range(max_steps - 1):
        B = len(beam_ids)
        step_in = torch.tensor([[ids[-1]] for ids in beam_ids], dtype=torch.long, device=device)
        out, past = model.decoder_model(step_in, bridged.expand(B, -1, -1), None, None, None,
                                        past_kvs=past, use_cache=True)
        logprobs = F.log_softmax(out.logits[:, -1].float(), dim=-1)      # (B, V)
        cand = beam_scores.unsqueeze(1) + logprobs
        V = logprobs.shape[-1]
        # Top 2*beam_width candidates: EOS extensions retire to `finished`, the rest
        # refill the active beam until it is full again.
        flat_scores, flat_idx = cand.flatten().topk(min(2 * beam_width, cand.numel()))
        new_ids, new_scores, parents = [], [], []
        for sc, fi in zip(flat_scores.tolist(), flat_idx.tolist()):
            parent, tok = divmod(fi, V)
            if tok == eos_id:
                finished.append({"ids": beam_ids[parent][:], "logp": sc})
            else:
                new_ids.append(beam_ids[parent] + [tok])
                new_scores.append(sc)
                parents.append(parent)
            if len(new_ids) == beam_width:
                break
        if len(finished) >= beam_width or not new_ids:
            beam_ids = new_ids
            break
        beam_ids = new_ids
        beam_scores = torch.tensor(new_scores, device=device)
        past = _reorder_cache(past, torch.tensor(parents, dtype=torch.long, device=device))

    # Budget exhausted (or beam retired early): close the remaining actives without EOS.
    for ids, sc in zip(beam_ids, beam_scores.tolist()[:len(beam_ids)]):
        finished.append({"ids": ids, "logp": sc})
    return finished


@torch.no_grad()
def ctc_hyp_logprobs(ctc_log_probs: torch.Tensor, hyps: list, blank_id: int) -> torch.Tensor:
    """logP_ctc(y|x) for each hypothesis via the CTC forward algorithm (= -ctc_loss).

    ctc_log_probs: (T, C) fp32 log-softmax frame posteriors, on CPU (MPS has no
    ctc_loss kernel, and this is cheap). Hypotheses that CTC cannot emit (longer
    than the frame count, or containing ids outside the CTC class space) score -inf,
    which simply removes them from the combined ranking.
    """
    T, C = ctc_log_probs.shape
    n = len(hyps)
    scores = torch.full((n,), float("-inf"))
    valid = [i for i, h in enumerate(hyps)
             if len(h) <= T and all(0 <= t < blank_id for t in h)]
    if not valid:
        return scores
    max_len = max(1, max(len(hyps[i]) for i in valid))
    targets = torch.zeros(len(valid), max_len, dtype=torch.long)
    tlens = torch.zeros(len(valid), dtype=torch.long)
    for row, i in enumerate(valid):
        h = hyps[i]
        targets[row, :len(h)] = torch.tensor(h, dtype=torch.long)
        tlens[row] = len(h)
    lp = ctc_log_probs.unsqueeze(1).expand(T, len(valid), C).contiguous()
    loss = F.ctc_loss(lp, targets,
                      input_lengths=torch.full((len(valid),), T, dtype=torch.long),
                      target_lengths=tlens,
                      blank=blank_id, reduction="none", zero_infinity=False)
    scores[torch.tensor(valid)] = -loss
    return scores


if __name__ == "__main__":
    ROOT = "/Users/xai/Personal/Projects/TeluguOCR"

    # ------------------------------- CONFIG -------------------------------
    # <-- point at the stage-2 checkpoint downloaded from Kaggle
    #     (.pt state dict or a checkpoint dir's model.safetensors)
    CKPT_PATH = f"{ROOT}/models/encoder_decoder/results_stage_2_mid/telugu-ocr-stage2/checkpoint-34000/model.safetensors"
    VOCAB_FILE = f"{ROOT}/src/telugu_ocr/tokenizer/assets/telugu-vocab.json"

    DATA_DIR = f"{ROOT}/data/wikisource_lines_eval"
    LABELS_FILE = f"{DATA_DIR}/lines_corrected.jsonl"   # human-reviewed references

    N_SAMPLES = None          # None = all completed rows (199). Set e.g. 50 for a smoke run.
    SEED = 42                 # subsample seed when N_SAMPLES < available rows

    BEAM_WIDTH = 5
    MAX_NEW_TOKENS = 200      # decode budget per line (ctx caps it at 255 anyway)
    BEAM_LEN_ALPHA = 0.0      # length-norm for the pure-LM beam pick: logp / (len+1)^alpha.
                              # 0.0 = raw log-prob; try 0.6 if beam output truncates early.
    LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0]  # CTC weight in joint rescoring

    PRINT_K = 3               # sample transcriptions to print
    WORST_K = 5               # worst joint-decoded samples to print at the end
    OUT_JSON = f"{DATA_DIR}/eval_results_1.json"   # per-sample dump; None to disable

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    # -----------------------------------------------------------------------

    tokenizer = TeluguGraphemeTokenizer(vocab_file=VOCAB_FILE)

    # Configs must match the stage-2 checkpoint (see train_stage_2.py).
    decoder_config = GPTConfig(
        vocab_size=len(tokenizer), embed_dim=512, hidden_dim=1368,
        num_heads=8, num_layers=16, ctx_len=256, dropout=0.0)
    encoder_config = CTCEncoderConfig(max_image_width=2048, max_frames=256)

    model = EncoderDecoder(encoder_config, decoder_config, tokenizer.pad_token_id)
    if CKPT_PATH.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(CKPT_PATH)
    else:
        state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)
    blank_id = model.encoder_model.blank_id
    print(f"[setup] loaded {CKPT_PATH}\n[setup] device={DEVICE}, blank_id={blank_id}")

    # ---- Data: human-reviewed rows only ----
    with open(LABELS_FILE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r.get("status") == "completed" and r.get("original_text", "").strip()]
    if N_SAMPLES is not None and len(rows) > N_SAMPLES:
        rows = random.Random(SEED).sample(rows, N_SAMPLES)
    print(f"[data] evaluating {len(rows)} human-reviewed lines from {LABELS_FILE}")

    # ---- Accumulators: corpus CER = sum(edits) / sum(ref lens) ----
    decoders = ["ctc", "beam"] + [f"joint@{lam}" for lam in LAMBDAS]
    edits = {d: 0 for d in decoders}
    exact = {d: 0 for d in decoders}
    total_ref = 0
    per_sample, skipped = [], 0
    t_enc = t_ctc = t_beam = t_joint = 0.0
    t_start = time.time()

    for i, r in enumerate(rows):
        ref = r["text"].strip()
        img_path = os.path.join(DATA_DIR, r["line_image"])
        try:
            img = Image.open(img_path)
        except OSError:
            skipped += 1
            continue
        if img.width / img.height * 64 > encoder_config.max_image_width:
            skipped += 1
            continue
        pix = preprocess_image(img).unsqueeze(0).to(DEVICE)   # (1, 1, 64, W)

        # ---- ONE encoder forward feeds all three decoders ----
        t0 = time.time()
        with torch.no_grad():
            enc_raw, _ = model.encoder_model.encode(pix, None)         # (1, T, D_enc)
            bridged = model.enc_to_dec(enc_raw)                        # (1, T, D_dec)
            ctc_lp = model.encoder_model.ctc_head(enc_raw)[0].float().log_softmax(-1).cpu()
        t_enc += time.time() - t0

        # 1) CTC greedy
        t0 = time.time()
        ctc_text = tokenizer.decode(ctc_greedy_ids(ctc_lp, blank_id),
                                    skip_special_tokens=True).strip()
        t_ctc += time.time() - t0

        # 2) LM beam search (n-best kept for the joint rescoring)
        t0 = time.time()
        nbest = beam_search(model, bridged, tokenizer.bos_token_id, tokenizer.eos_token_id,
                            BEAM_WIDTH, MAX_NEW_TOKENS)
        t_beam += time.time() - t0
        texts = [tokenizer.decode(h["ids"], skip_special_tokens=True).strip() for h in nbest]
        attn_lp = torch.tensor([h["logp"] for h in nbest])
        norm = attn_lp / torch.tensor([(len(h["ids"]) + 1.0) ** BEAM_LEN_ALPHA for h in nbest])
        beam_text = texts[int(norm.argmax())]

        # 3) Joint rescoring of the SAME n-best across the lambda grid
        t0 = time.time()
        ctc_scores = ctc_hyp_logprobs(ctc_lp, [h["ids"] for h in nbest], blank_id)
        joint_text = {}
        for lam in LAMBDAS:
            combined = lam * ctc_scores + (1.0 - lam) * attn_lp
            if torch.isinf(combined).all():        # CTC rejected every hypothesis
                combined = attn_lp
            joint_text[lam] = texts[int(combined.argmax())]
        t_joint += time.time() - t0

        # ---- Score ----
        total_ref += max(len(ref), 1)
        outputs = {"ctc": ctc_text, "beam": beam_text,
                   **{f"joint@{lam}": joint_text[lam] for lam in LAMBDAS}}
        rec = {"line_image": r["line_image"], "ref": ref}
        for name, hyp in outputs.items():
            e = edit_distance(hyp, ref)
            edits[name] += e
            exact[name] += int(hyp == ref)
            rec[name] = hyp
            rec[f"{name}_edits"] = e
        per_sample.append(rec)

        if i < PRINT_K:
            print(f"\n[{i}] {r['line_image']}")
            print(f"  ref : {ref!r}")
            print(f"  ctc : {ctc_text!r}")
            print(f"  beam: {beam_text!r}")
            print(f"  jnt : {joint_text[0.3]!r}   (lambda=0.3)")
        if (i + 1) % 20 == 0:
            done = len(per_sample)
            print(f"[{i + 1}/{len(rows)}] running CER  "
                  f"ctc={edits['ctc'] / total_ref:.4f}  beam={edits['beam'] / total_ref:.4f}  "
                  f"joint@0.3={edits['joint@0.3'] / total_ref:.4f}  "
                  f"({(time.time() - t_start) / done:.2f}s/sample)", flush=True)

    # ---- Summary ----
    n = len(per_sample)
    print(f"\n{'=' * 64}\n[result] {n} lines scored, {skipped} skipped, "
          f"{time.time() - t_start:.0f}s total "
          f"(enc {t_enc:.0f}s, ctc {t_ctc:.0f}s, beam {t_beam:.0f}s, joint {t_joint:.0f}s)")
    print(f"{'decoder':<12} {'CER':>8} {'exact-match':>12}")
    for name in decoders:
        print(f"{name:<12} {edits[name] / total_ref:>8.4f} {exact[name] / max(n, 1):>11.1%}")
    best = min(decoders, key=lambda d: edits[d])
    print(f"\n[result] best decoder: {best}  (CER {edits[best] / total_ref:.4f})")

    worst = sorted(per_sample, key=lambda p: -p["joint@0.3_edits"])[:WORST_K]
    print(f"\n[worst {WORST_K} by joint@0.3 edits]")
    for p in worst:
        print(f"  {p['line_image']}  ({p['joint@0.3_edits']} edits)")
        print(f"    ref : {p['ref']!r}")
        print(f"    jnt : {p['joint@0.3']!r}")

    if OUT_JSON:
        summary = {"n": n, "skipped": skipped,
                   "cer": {d: edits[d] / total_ref for d in decoders},
                   "exact": {d: exact[d] / max(n, 1) for d in decoders},
                   "config": {"ckpt": CKPT_PATH, "beam_width": BEAM_WIDTH,
                              "max_new_tokens": MAX_NEW_TOKENS, "lambdas": LAMBDAS,
                              "beam_len_alpha": BEAM_LEN_ALPHA, "n_samples": N_SAMPLES}}
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "samples": per_sample}, f,
                      ensure_ascii=False, indent=1)
        print(f"\n[result] per-sample results -> {OUT_JSON}")

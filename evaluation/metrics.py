"""
Metric implementations used across the evaluation scripts.

Deliberately avoids anything that shells out to Java (pycocoevalcap's METEOR
and SPICE scorers do) since that's an extra, easy-to-forget system dependency
that reviewers can't reproduce without noticing. METEOR here is NLTK's
pure-Python implementation instead.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Captioning metrics
# ---------------------------------------------------------------------------

_nltk_ready = False


def _ensure_nltk():
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk
    for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    _nltk_ready = True


def bleu_1234(candidate, references):
    """Sentence-level BLEU-1..4 with NLTK method1 smoothing (avoids 0 scores on short captions)."""
    _ensure_nltk()
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothie = SmoothingFunction().method1
    cand_tok = candidate.lower().split()
    ref_tok = [r.lower().split() for r in references]
    scores = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n for _ in range(n)) + tuple(0.0 for _ in range(4 - n))
        scores[f"bleu{n}"] = sentence_bleu(ref_tok, cand_tok, weights=weights, smoothing_function=smoothie)
    return scores


def meteor(candidate, references):
    _ensure_nltk()
    from nltk.translate.meteor_score import meteor_score
    cand_tok = candidate.lower().split()
    ref_tok = [r.lower().split() for r in references]
    return meteor_score(ref_tok, cand_tok)


def rouge_l(candidate, references):
    """Best ROUGE-L F1 against any single reference (standard multi-ref practice)."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    best = 0.0
    for ref in references:
        score = scorer.score(ref, candidate)["rougeL"].fmeasure
        best = max(best, score)
    return best


_cider_scorer = None


def cider_corpus(candidates_by_id, references_by_id):
    """
    Corpus-level CIDEr (needs the whole corpus at once — CIDEr's IDF term is
    computed over the candidate set). candidates_by_id/references_by_id are
    {image_id: caption} / {image_id: [refs]}.
    """
    from pycocoevalcap.cider.cider import Cider
    scorer = Cider()
    gts = {k: [r for r in v] for k, v in references_by_id.items()}
    res = {k: [v] for k, v in candidates_by_id.items()}
    score, scores = scorer.compute_score(gts, res)
    return score, dict(zip(candidates_by_id.keys(), scores))


def caption_metrics_for_method(candidates_by_id, references_by_id):
    """Aggregate BLEU-1..4/METEOR/ROUGE-L (mean over sentences) + corpus CIDEr for one method."""
    per_sentence = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [], "meteor": [], "rougeL": []}
    for image_id, cand in candidates_by_id.items():
        refs = references_by_id[image_id]
        b = bleu_1234(cand, refs)
        for k in ["bleu1", "bleu2", "bleu3", "bleu4"]:
            per_sentence[k].append(b[k])
        per_sentence["meteor"].append(meteor(cand, refs))
        per_sentence["rougeL"].append(rouge_l(cand, refs))

    cider_score, _ = cider_corpus(candidates_by_id, references_by_id)

    return {
        "bleu1": float(np.mean(per_sentence["bleu1"])),
        "bleu2": float(np.mean(per_sentence["bleu2"])),
        "bleu3": float(np.mean(per_sentence["bleu3"])),
        "bleu4": float(np.mean(per_sentence["bleu4"])),
        "meteor": float(np.mean(per_sentence["meteor"])),
        "rougeL": float(np.mean(per_sentence["rougeL"])),
        "cider": float(cider_score),
        "n": len(candidates_by_id),
    }


# ---------------------------------------------------------------------------
# Enhancement metrics
# ---------------------------------------------------------------------------

def psnr_ssim(reference_rgb_uint8, output_rgb_uint8):
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
    ref = reference_rgb_uint8.astype(np.uint8)
    out = output_rgb_uint8.astype(np.uint8)
    if ref.shape != out.shape:
        raise ValueError(f"Shape mismatch for PSNR/SSIM: ref={ref.shape} out={out.shape}")
    psnr_val = psnr(ref, out, data_range=255)
    ssim_val = ssim(ref, out, data_range=255, channel_axis=-1)
    return float(psnr_val), float(ssim_val)


_lpips_model = None


def lpips_distance(reference_rgb_uint8, output_rgb_uint8, device="cpu"):
    """LPIPS (lower is better), AlexNet backbone (the standard default)."""
    global _lpips_model
    import torch
    import lpips as lpips_lib
    if _lpips_model is None:
        _lpips_model = lpips_lib.LPIPS(net="alex").to(device)
        _lpips_model.eval()

    def to_tensor(img):
        t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
        return t.to(device)

    with torch.no_grad():
        d = _lpips_model(to_tensor(reference_rgb_uint8), to_tensor(output_rgb_uint8))
    return float(d.item())


# ---------------------------------------------------------------------------
# Translation metrics
# ---------------------------------------------------------------------------

def bleu_chrf_corpus(candidates, references):
    """Corpus BLEU + chrF via sacrebleu (references: list[str], one ref per candidate)."""
    import sacrebleu
    refs_transposed = [references]
    bleu = sacrebleu.corpus_bleu(candidates, refs_transposed)
    chrf = sacrebleu.corpus_chrf(candidates, refs_transposed)
    return float(bleu.score), float(chrf.score)


def bertscore_f1(candidates, references, lang="en", device="cpu"):
    """
    XLM-R rather than mBERT: mBERT's training data doesn't reliably cover
    Kannada/Telugu (and several other Indic languages this project
    supports), which would silently degrade to near-meaningless scores for
    those rows. XLM-R's CC100 training data covers all of C4PS's Indic
    target languages plus its European/East Asian ones.
    """
    from bert_score import score as bert_score_fn
    _, _, f1 = bert_score_fn(
        candidates, references,
        model_type="xlm-roberta-base",
        num_layers=9,
        lang=lang,
        device=device,
        verbose=False,
    )
    return float(f1.mean().item())

"""
Dataset loaders for the evaluation suite.

Both loaders try several known-good HuggingFace Hub sources in order, because
dataset repos on the Hub occasionally get renamed or gated. Every attempt is
logged so the experimental-setup report can record exactly which source was
used for a given run (reproducibility requirement).
"""
import io
import os
import json
import random

from .common import REPO_ROOT, results_path

CACHE_DIR = os.path.join(REPO_ROOT, "evaluation", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _log(msg):
    print(f"[dataset] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Flickr8k (image captioning references)
# ---------------------------------------------------------------------------

FLICKR8K_SOURCES = [
    "jxie/flickr8k",
    "atasoglu/flickr8k-dataset",
    "Naveengo/flickr8k",
]


def load_flickr8k(n_samples=100, split="test", seed=42):
    """
    Returns a list of dicts: {"image_id": str, "image": PIL.Image, "references": [str, ...]}

    Tries several HF Hub mirrors of Flickr8k since none of them is an
    "official" canonical release. Falls back to the 'train' split if the
    requested split name isn't present in a given mirror.
    """
    from datasets import load_dataset
    import PIL.Image

    last_err = None
    for repo_id in FLICKR8K_SOURCES:
        try:
            _log(f"Trying Flickr8k source: {repo_id}")
            ds_dict = load_dataset(repo_id, cache_dir=os.path.join(CACHE_DIR, "hf_datasets"))
            available_splits = list(ds_dict.keys())
            use_split = split if split in ds_dict else available_splits[0]
            ds = ds_dict[use_split]
            _log(f"Loaded {repo_id}[{use_split}] with {len(ds)} rows, columns={ds.column_names}")

            cols = ds.column_names
            image_col = next((c for c in cols if c.lower() in ("image", "img", "jpg")), None)
            caption_cols = [c for c in cols if "caption" in c.lower() or c.lower().startswith("text")]
            single_caption_col = None
            if not caption_cols:
                for c in cols:
                    if c.lower() in ("sentences", "captions"):
                        single_caption_col = c
                        break
            if image_col is None:
                raise ValueError(f"No image column found among {cols}")

            rng = random.Random(seed)
            n_total = len(ds)
            indices = list(range(n_total))
            rng.shuffle(indices)
            indices = indices[:n_samples]

            samples = []
            for i, idx in enumerate(indices):
                row = ds[idx]
                img = row[image_col]
                if not isinstance(img, PIL.Image.Image):
                    img = PIL.Image.open(io.BytesIO(img["bytes"])) if isinstance(img, dict) else PIL.Image.open(img)
                img = img.convert("RGB")

                refs = []
                if caption_cols:
                    for c in caption_cols:
                        v = row[c]
                        if isinstance(v, list):
                            refs.extend([str(x) for x in v if str(x).strip()])
                        elif v:
                            refs.append(str(v))
                elif single_caption_col:
                    v = row[single_caption_col]
                    if isinstance(v, list):
                        refs = [str(x) for x in v]
                    else:
                        refs = [str(v)]
                if not refs:
                    continue

                image_id = row.get("image_id") or row.get("id") or row.get("filename") or f"{repo_id.replace('/', '_')}_{idx}"
                samples.append({
                    "image_id": str(image_id),
                    "image": img,
                    "references": refs,
                })

            if len(samples) == 0:
                raise ValueError("Loaded dataset but extracted zero usable samples")

            _log(f"Using {len(samples)} Flickr8k samples from {repo_id}[{use_split}]")
            with open(results_path("dataset_provenance_captioning.json"), "w") as f:
                json.dump({"source": repo_id, "split": use_split, "n_requested": n_samples,
                           "n_used": len(samples), "seed": seed}, f, indent=2)
            return samples

        except Exception as e:
            _log(f"Source {repo_id} failed: {e}")
            last_err = e
            continue

    raise RuntimeError(f"All Flickr8k sources failed. Last error: {last_err}")


# ---------------------------------------------------------------------------
# FLORES-200 (multilingual MT references)
# ---------------------------------------------------------------------------

# FLORES-200 (facebook/flores) language codes for the languages C4PS supports.
FLORES_LANG_CODES = {
    "hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda",
    "ml": "mal_Mlym", "bn": "ben_Beng", "mr": "mar_Deva", "gu": "guj_Gujr",
    "pa": "pan_Guru", "or": "ory_Orya", "as": "asm_Beng", "ur": "urd_Arab",
    "ko": "kor_Hang", "tr": "tur_Latn", "ja": "jpn_Jpan", "pt": "por_Latn",
    "ar": "arb_Arab", "it": "ita_Latn", "nl": "nld_Latn", "vi": "vie_Latn",
    "id": "ind_Latn", "fr": "fra_Latn", "es": "spa_Latn", "de": "deu_Latn",
    "zh": "zho_Hans", "ru": "rus_Cyrl",
}
FLORES_ENGLISH_CODE = "eng_Latn"

FLORES_ARCHIVE_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_DIR = os.path.join(CACHE_DIR, "flores200")
FLORES_EXTRACTED_DIR = os.path.join(FLORES_DIR, "flores200_dataset")


def _ensure_flores_downloaded():
    """
    facebook/flores and openlanguagedata/flores_plus on the HF Hub both
    require an authenticated + terms-accepted account (gated), which this
    unattended environment doesn't have. FLORES-200 is nonetheless a public
    dataset: Meta publishes the exact same devtest/dev split as a plain
    tarball on their own public-files CDN (this is the canonical download
    referenced by the FLORES-200 paper and used by fairseq/NLLB tooling), so
    we pull from there directly instead.
    """
    if os.path.isdir(os.path.join(FLORES_EXTRACTED_DIR, "devtest")):
        return
    os.makedirs(FLORES_DIR, exist_ok=True)
    archive_path = os.path.join(FLORES_DIR, "flores200_dataset.tar.gz")
    _log(f"Downloading FLORES-200 from {FLORES_ARCHIVE_URL} ...")
    import requests
    with requests.get(FLORES_ARCHIVE_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    import tarfile
    with tarfile.open(archive_path) as tf:
        tf.extractall(FLORES_DIR)
    _log(f"Extracted FLORES-200 to {FLORES_EXTRACTED_DIR}")


def load_flores_parallel(target_langs, n_sentences=20, seed=42, split="devtest"):
    """
    Returns {"en": [sentences...], lang_code: [reference sentences aligned to en...], ...}
    using the FLORES-200 devtest split, which is sentence-aligned across all
    languages (i.e. line i is the same underlying sentence in every language's file).
    """
    _ensure_flores_downloaded()
    split_dir = os.path.join(FLORES_EXTRACTED_DIR, split)

    def read_lines(flores_code):
        path = os.path.join(split_dir, f"{flores_code}.{split}")
        with open(path, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    en_lines = read_lines(FLORES_ENGLISH_CODE)
    n_total = len(en_lines)

    lang_lines = {}
    for lang in target_langs:
        flores_code = FLORES_LANG_CODES.get(lang)
        if flores_code is None:
            _log(f"WARNING: no FLORES-200 code mapping for '{lang}'; skipping")
            continue
        path = os.path.join(split_dir, f"{flores_code}.{split}")
        if not os.path.exists(path):
            _log(f"WARNING: no FLORES-200 file for {lang} ({flores_code}) at {path}; skipping")
            continue
        lang_lines[lang] = read_lines(flores_code)

    if not lang_lines:
        raise RuntimeError("None of the requested target languages have a FLORES-200 file")

    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    indices = indices[:n_sentences]

    out = {"en": [en_lines[i] for i in indices]}
    for lang, lines in lang_lines.items():
        out[lang] = [lines[i] for i in indices]

    with open(results_path("dataset_provenance_translation.json"), "w") as f:
        json.dump({"source": FLORES_ARCHIVE_URL, "split": split, "n_sentences": len(indices),
                   "languages": list(lang_lines.keys()), "seed": seed}, f, indent=2)
    _log(f"Using {len(indices)} FLORES-200 sentences across {len(lang_lines)} languages ({split})")
    return out

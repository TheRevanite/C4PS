# C4PS Evaluation Suite

Produces the reproducible numbers the paper's Results section needs, using
C4PS's own pipeline code (not reimplementations). All scripts write CSVs to
`evaluation/results/` and are safe to re-run (most resume/append).

## Two pre-existing bugs found and fixed while building this

These aren't evaluation-script issues -- they were silently breaking the
actual pipeline (`main.py`) before this suite existed, and would have kept
producing wrong numbers/images regardless of which experiments got run.
Mentioned here because they change what "the current code" means for
anything the paper says about image quality.

1. **`enhancement/model.py`: half-precision inference silently zeroed the
   output.** `EnhancementModel` set `half=True` for any CUDA device.
   Empirically (this dev GPU: GTX 1650, driver 550, torch 2.6+cu124),
   `RealESRGANer` with `half=True` returns a valid-shaped array that is
   *all zeros* -- no exception, no NaN, just silent numerical underflow
   through the RRDBNet forward pass. Combined with tiling, this produced
   partially-black output images (bug looked like a tiling bug at first —
   whichever tile happened to under/overflow came out black, others were
   fine) that GIT then captioned as "a black background". Verified by
   diffing `half=True` vs `half=False` output arrays on an identical input:
   only `half=False` produces a non-degenerate image. Fixed by hardcoding
   `use_half = False` in both the initial construction and the OOM-retry
   path. This means enhancement is now slower on this GPU than the
   README's benchmark table claims (those numbers were presumably measured
   on hardware where fp16 doesn't misbehave, or measured against the same
   silently-broken fast-but-wrong path) -- `benchmark_runtime.py`'s numbers
   in this run reflect the honest fp32 cost.

2. **`enhancement/enhancer.py`: `sharp_anime` and `auto_vehicle` modes had
   no `MODELS_CONFIG` entry**, so `MODELS_CONFIG.get(mode, MODELS_CONFIG['general'])`
   silently ran the general x4plus model for both. `auto_vehicle` is left as
   an explicit, documented alias of `general` (no vehicle-specific
   checkpoint exists to route to instead). `sharp_anime` now points at
   Real-ESRGAN's official `RealESRGAN_x4plus_anime_6B.pth` checkpoint,
   which is a real, distinct, officially-released model appropriate to what
   the mode's name already claimed. Before this fix, any paper claim of
   "4 distinct enhancement modes" was true of the UI, not of the model
   actually run for 2 of the 4 choices.

3. **`enhancement/model.py`: face-enhance mode double-upscaled the image,
   causing the CUDA OOM crashes this run's `runtime_raw.csv` originally
   recorded at 1024px with tile<=256.** `GFPGANer` was constructed with
   `upscale=outscale` (e.g. 4), but the image it's given is
   `RealESRGANer`'s output, which has *already* been upscaled by `outscale`.
   `GFPGANer`'s `FaceRestoreHelper.paste_faces_to_input_image` multiplies
   whatever image it receives by `upscale_factor` again when pasting
   restored faces back (confirmed by reading `facexlib`'s source), so the
   two stages compounded to an `outscale**2` total blow-up -- a 681x1024
   input at `outscale=4` came out 10896x16384 (16x linear / 256x pixels)
   instead of the intended 2724x4096 (4x). That's what was actually
   exhausting the 3.81GB GPU's memory at higher resolutions, not (only)
   allocator fragmentation. Fixed by constructing `GFPGANer(upscale=1, ...)`
   since its input is already at the target resolution; also added a
   GPU-cache-clear between the RealESRGAN and GFPGAN stages as a secondary
   mitigation. Re-running the full `benchmark_runtime.py` sweep after the
   fix: all 45 (mode, resolution, tile) configs now complete 15/15 repeats
   with zero crashes, including the two configs that crashed 3/6 times
   before. `general_x4plus_gfpgan` is also now correctly ~4-5x faster in
   its GFPGAN stage (was doing 16x the pixel work it needed to).

Run once, in this order, from the `C4PS/` repo root with the venv active:

```bash
source .venv/bin/activate
python -m evaluation.record_environment                  # setup/versions table
python -m evaluation.evaluate_captioning --n-samples 150 --no-resume  # captioning quality + adaptive-pipeline evidence
python -m evaluation.evaluate_enhancement --n-samples 100 # enhancement quality vs ground truth
python -m evaluation.evaluate_translation --languages core --n-sentences 80
python -m evaluation.benchmark_runtime --n-images 5 --repeats 3
python -m evaluation.significance                         # paired significance / CIs on top of the above
```

Sample sizes were bumped from the first pass (captioning 60->150, enhancement
20->100, translation 30->80/language) after `significance.py` showed the
original n wasn't enough to distinguish several between-method captioning
differences from noise -- see "Statistical significance" below before citing
any specific method-vs-method captioning number.

## What each script produces

| Script | Output | Answers reviewer point |
|---|---|---|
| `record_environment.py` | `experimental_setup.json` | #12 reproducibility |
| `evaluate_captioning.py` | `captions_raw.csv`, `captioning_metrics.csv` | #1, #2, #3 (adaptive pipeline) |
| `evaluate_enhancement.py` | `enhancement_raw.csv`, `enhancement_metrics.csv` | #4, #5, #6 |
| `evaluate_translation.py` | `translation_raw.csv`, `translation_metrics.csv` | #7, #8 |
| `benchmark_runtime.py` | `runtime_raw.csv`, `runtime_summary.csv` | #9, #10, #11 |
| `significance.py` | `enhancement_significance.csv`, `captioning_significance.csv`, `translation_ci.csv` | statistical backing for any table/claim above |

## Statistical significance (read before citing a specific number)

`*_metrics.csv` only report means -- not enough to tell whether two methods'
scores are actually distinguishable given n in the dozens-to-low-hundreds and
stds comparable in magnitude to the between-method gaps. `significance.py`
adds paired Wilcoxon signed-rank + paired t-test + a 10000-resample bootstrap
95% CI on the mean paired difference (enhancement modes vs `bicubic` at
matching scale; captioning modes vs `original`), and a bootstrap 95% CI on
each translation corpus metric.

At n=100/150/80 (enhancement/captioning/translation): every enhancement
mode's PSNR/SSIM/LPIPS difference from bicubic is significant (expected --
see the perception-distortion tradeoff note above). For captioning, only one
difference clears significance: `sharp_anime_x4` has significantly *lower*
ROUGE-L than `original` (diff=-0.0204, 95% CI=[-0.0385,-0.0027]). Every other
captioning mode-vs-original difference across BLEU-4/METEOR/ROUGE-L
(including both `general_x4plus`/`auto_vehicle_x4` readings, which are
identical since they're the same underlying model) has a 95% CI crossing
zero -- i.e. not distinguishable from noise at this sample size. **Do not
cite "enhancement mode X improves captioning" as a finding**; the honest
claim is "enhancement doesn't measurably hurt captioning (with one exception:
sharp_anime_x4 on ROUGE-L)," not that it helps.

## Methodology notes (read before quoting numbers in the paper)

- **Captioning**: Flickr8k images, 5 conditions per image -- `original`
  (GIT captions the raw image; this is what main.py's general/sharp_anime/
  auto_vehicle paths do) vs `fast_x2`/`sharp_anime_x4`/`general_x4plus`/
  `auto_vehicle_x4` (GIT captions the *enhanced* image; this is what
  main.py's fast path does). BLEU-1..4/METEOR/ROUGE-L are averaged per
  sentence; CIDEr is corpus-level (its IDF term needs the whole set). This
  is the direct evidence for whether enhancing before captioning helps.

- **Enhancement**: there is no real "ground truth HQ" for an arbitrary
  low-quality input, so we do the standard SR-literature thing: take a
  Flickr8k image as HQ, bicubic-downsample it (the classical degradation
  model) to produce the LQ input, run each method, and score against the
  original HQ. This is why the previous PSNR/SSIM numbers (scored against
  nothing in particular) weren't defensible and these are.

- **Translation**: FLORES-200 devtest, not the Flickr8k captions --
  Flickr8k has no reference translations, so there was nothing correct to
  compare against. FLORES-200 is sentence-aligned across 200 languages,
  which is exactly what's needed for a real BLEU/chrF/BERTScore table.
  `--languages core` covers the reviewer's example set (Hindi, Kannada,
  Tamil, Telugu, Spanish, French); `--languages all` covers every language
  C4PS's router supports, but costs one model download+load per language
  (MarianMT keeps only one model resident at a time -- see
  `translation/marian.py` -- so this reloads repeatedly and is slow).

- **Runtime**: every (mode, resolution, tile size) config is run in a
  subprocess with a timeout so one OOM/hang doesn't kill the sweep; a
  timed-out or OOM'd config is recorded as such in `runtime_raw.csv`
  rather than silently dropped, which is itself informative about which
  configs are actually usable on constrained hardware (this dev box: GTX
  1650, 3.81GB VRAM). After the double-upscale fix (bug #3 above), the full
  45-config sweep completes 15/15 repeats with zero crashes, including
  `general_x4plus_gfpgan` at every resolution/tile combo -- the earlier
  crashes at 1024px/tile<=256 were the bug, not an inherent hardware limit.
  A single-image edge case at 1920px can still OOM within GFPGAN's own face
  restoration (unrelated to the fixed bug -- some real faces just need more
  memory than others to align/restore); this now degrades gracefully to the
  RealESRGAN-only result instead of crashing the process.

## What this suite deliberately does NOT do

- No XAI experiments. Nothing in the current codebase implements
  saliency/LIME/SHAP, and bolting one on just to answer a reviewer comment
  would be exactly the kind of unsupported-claim problem that triggered
  this whole revision. Recommended fix is Path B: drop the speculative XAI
  paragraph from the paper (or explicitly move it to Future Work), not
  fabricate a XAI section to match Path A.
- No SOTA external-baseline comparison (BLIP/LLaVA for captioning,
  standalone Real-ESRGAN/ESRGAN checkpoints beyond what's already wired
  into `enhancement/model.py` for enhancement). This is deliberately out of
  scope for a first pass -- it's a separate, larger engineering effort
  (new model dependencies, likely more VRAM than this GTX 1650 has) rather
  than "run the existing code and log numbers." Worth doing once the
  measurements above are in and stable.
- No user study (#15) -- that's a human-subjects data collection effort,
  not something a script can produce.

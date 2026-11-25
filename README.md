# ming-transfer

Turns a Z-Image fine-tune into a LoRA for Ming-Image 0.1 Design in ComfyUI.

Ming-Image's DiT is Z-Image base with a light fine-tune, plus its own text encoder, VAE and input/output layers. A Z-Image fine-tune's change over its own base model can be added to Ming's DiT blocks, and Ming keeps its editing: the fine-tune's look comes with it. The script computes that change on the blocks only (`layers`, `noise_refiner`, `context_refiner`) and leaves Ming's input and output layers alone.

## Use

Run it with ComfyUI's Python environment (it needs `torch` and `safetensors`):

```bash
python ming_transfer.py --finetune <fine-tune> --base zimage|turbo --rank 64 --out <ComfyUI>/models/loras/ming/<name>.safetensors
```

- `--finetune`: the fine-tune's diffusion model. A local `.safetensors`, a URL, `owner/repo/path.safetensors` on Hugging Face, or a diffusers `.index.json`. Remote files are read by byte range, so only the block tensors are downloaded and nothing is saved besides the output. Downloads retry when the connection drops.
- `--base`: the model the fine-tune was trained from. `zimage` for Z-Image base, `turbo` for Z-Image Turbo, or a file, URL or index like `--finetune`. Using the wrong base adds the difference between the two bases, which is large.
- `--rank`: the LoRA rank (default 64). `0` saves the full difference as a `.diff` LoRA, about 12 GB. Norms and biases are always saved in full.
- `--keys`: only block keys matching a regex, for a quick test (`'^layers\.15\.'`) or a partial transfer.
- `--workers`: parallel downloads (default 4).

At the end it prints how large the change is and how much of it the rank keeps. A small, low-rank change (a light fine-tune) keeps most of it at rank 64; a full fine-tune can need rank 0.

Load the output with **LoraLoaderModelOnly** on the Ming-Image model.

## Results so far

| Fine-tune | Base | Change | Rank 64 keeps | On Ming |
|---|---|---|---|---|
| BEYOND REALITY 3.0 (`Nurburgring/BEYOND_REALITY_Z_IMAGE`) | turbo | ~0.4–2 % | 83 % | Film look at strength ~2; edits hold up to 2, at 4 the image washes out. Rank 64 ≈ full difference. |
| Z-Anime base (`SeeSee21/Z-Anime`) | zimage | 6–17 % | not measured | Full difference (rank 0): anime style at strength 1 and edits still work. Text to image wants CFG 3 at strength 1 to lose a grain; edits keep a paper-like skin texture there, so use ~0.5. |

Moving Ming's blocks back towards Z-Image base breaks it at a quarter of the way, so the method relies on the fine-tune's change going in another direction than Ming's own, which both fine-tunes above did.

## Examples

```bash
python ming_transfer.py --finetune "Nurburgring/BEYOND_REALITY_Z_IMAGE/BEYOND REALITY SUPER Z IMAGE 3.0 淡妆浓抹 BF16.safetensors" --base turbo --rank 64 --out ComfyUI/models/loras/ming/beyond_reality3_ming_r64.safetensors
python ming_transfer.py --finetune SeeSee21/Z-Anime/diffusion_models/z-anime-base-bf16.safetensors --base zimage --rank 0 --out ComfyUI/models/loras/ming/zanime_ming_diff.safetensors
```

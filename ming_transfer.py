"""Turn a Z-Image fine-tune into a LoRA for Ming-Image: the fine-tune's change over its own base model, on the DiT blocks only."""
import argparse
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors.torch import save_file

BASES = {
    "zimage": "https://huggingface.co/Tongyi-MAI/Z-Image/resolve/main/transformer/diffusion_pytorch_model.safetensors.index.json",
    "turbo": "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/main/transformer/diffusion_pytorch_model.safetensors.index.json",
}
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
BLOCKS = ("layers.", "noise_refiner.", "context_refiner.")
PREFIXES = ("model.diffusion_model.", "diffusion_model.")
# Diffusers names -> ComfyUI names; q, k and v are fused into qkv.
RENAME = {"attention.to_out.0.": "attention.out.", "attention.norm_q.": "attention.q_norm.", "attention.norm_k.": "attention.k_norm."}
QKV = ("to_q", "to_k", "to_v")


def hf_url(spec):
    """'owner/repo/path/file.safetensors' -> resolve URL; URLs and local paths pass through."""
    if spec.startswith(("http://", "https://")) or os.path.exists(spec):
        return spec
    owner, repo, path = spec.split("/", 2)
    return f"https://huggingface.co/{owner}/{repo}/resolve/main/{urllib.parse.quote(path)}"


def read_json(location):
    if location.startswith(("http://", "https://")):
        return urllib.request.urlopen(location, timeout=120).read()
    with open(location, "rb") as f:
        return f.read()


class File:
    """One .safetensors file, local or remote; tensors are read by byte range, so nothing is downloaded whole."""

    def __init__(self, location):
        self.location = location
        self.remote = location.startswith(("http://", "https://"))
        n = struct.unpack("<Q", self.read(0, 8))[0]
        self.data_start = 8 + n
        self.header = json.loads(self.read(8, n))
        self.header.pop("__metadata__", None)

    def read(self, start, size):
        if not self.remote:
            with open(self.location, "rb") as f:
                f.seek(start)
                return f.read(size)
        request = urllib.request.Request(self.location, headers={"Range": f"bytes={start}-{start + size - 1}"})
        for attempt in range(8):
            try:
                return urllib.request.urlopen(request, timeout=120).read()
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                if attempt == 7:
                    raise
                print(f"  retry {attempt + 1}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))

    def tensor(self, name):
        info = self.header[name]
        start, end = info["data_offsets"]
        data = bytearray(self.read(self.data_start + start, end - start))
        return torch.frombuffer(data, dtype=DTYPES[info["dtype"]]).reshape(info["shape"]).float()


class Model:
    """A diffusion model as one file or a sharded diffusers index, read under ComfyUI's Z-Image block names."""

    def __init__(self, spec):
        location = hf_url(spec)
        if location.endswith(".index.json"):
            index = json.loads(read_json(location))
            folder = location.rsplit("/", 1)[0]
            files = {shard: File(f"{folder}/{shard}") for shard in set(index["weight_map"].values())}
            raw = {name: files[shard] for name, shard in index["weight_map"].items()}
        else:
            file = File(location)
            raw = {name: file for name in file.header}
        # Block name without the checkpoint prefix -> (file, name in that file).
        self.raw = {}
        for original, file in raw.items():
            name = original
            for prefix in PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            self.raw[name] = (file, original)

    def comfy_keys(self):
        keys = set()
        for name in self.raw:
            if not name.startswith(BLOCKS):
                continue
            for a, b in RENAME.items():
                name = name.replace(a, b)
            for part in QKV:
                name = name.replace(f"attention.{part}.", "attention.qkv.")
            keys.add(name)
        return sorted(keys)

    def get(self, key):
        if key in self.raw:
            file, name = self.raw[key]
            return file.tensor(name)
        if ".attention.qkv." in key:
            return torch.cat([self.get(key.replace("attention.qkv.", f"attention.{p}.")) for p in QKV], dim=0)
        for a, b in RENAME.items():
            if b in key:
                return self.get(key.replace(b, a))
        raise KeyError(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetune", required=True, help="The fine-tune's diffusion model: a local .safetensors, a URL, 'owner/repo/path.safetensors' on Hugging Face, or a diffusers .index.json.")
    parser.add_argument("--base", required=True, help="The model the fine-tune was trained from: 'zimage' (Z-Image base), 'turbo' (Z-Image Turbo), or a file/URL/index like --finetune.")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank; 0 saves the full difference (about 12 GB).")
    parser.add_argument("--out", required=True, help="Output .safetensors, e.g. ComfyUI/models/loras/ming/name.safetensors.")
    parser.add_argument("--keys", default="", help="Only block keys matching this regex, e.g. '^layers\\.(1[0-9])\\.' for blocks 10-19.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    finetune = Model(args.finetune)
    base = Model(BASES.get(args.base, args.base))
    keys = [k for k in finetune.comfy_keys() if re.search(args.keys, k)]
    if not keys:
        sys.exit("No DiT block keys found in the fine-tune (or none match --keys).")
    print(f"{len(keys)} block tensors, rank {args.rank or 'full'}, SVD on {device}", flush=True)

    pool = ThreadPoolExecutor(args.workers)
    out = {}
    kept = total = change = weight = 0.0
    t0 = time.time()
    pending = [(k, pool.submit(finetune.get, k), pool.submit(base.get, k)) for k in keys[:args.workers]]
    for i, key in enumerate(keys):
        k, ft, bs = pending.pop(0)
        if i + args.workers < len(keys):
            nxt = keys[i + args.workers]
            pending.append((nxt, pool.submit(finetune.get, nxt), pool.submit(base.get, nxt)))
        ft, bs = ft.result(), bs.result()
        if ft.shape != bs.shape:
            sys.exit(f"{k}: fine-tune {tuple(ft.shape)} vs base {tuple(bs.shape)}; is --base the model it was trained from?")
        d = ft - bs
        change += d.square().sum().item()
        weight += bs.square().sum().item()
        name = "diffusion_model." + k.rsplit(".", 1)[0]
        if k.endswith(".bias"):
            out[name + ".diff_b"] = d.to(torch.bfloat16)
        elif args.rank and d.ndim == 2:
            d = d.to(device)
            U, S, V = torch.svd_lowrank(d, q=min(args.rank + 16, *d.shape), niter=6)
            U, S, V = U[:, :args.rank], S[:args.rank], V[:, :args.rank]
            kept += S.square().sum().item()
            total += d.square().sum().item()
            root = S.sqrt()
            out[name + ".lora_up.weight"] = (U * root).to(torch.bfloat16).cpu().contiguous()
            out[name + ".lora_down.weight"] = (root[:, None] * V.T).to(torch.bfloat16).cpu().contiguous()
            out[name + ".alpha"] = torch.tensor(float(args.rank))
        else:
            out[name + ".diff"] = d.to(torch.bfloat16)
        if i % 40 == 0 or i == len(keys) - 1:
            print(f"{i + 1}/{len(keys)} {k} {time.time() - t0:.0f}s", flush=True)

    save_file(out, args.out)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB).")
    ratio = (change / weight) ** 0.5
    print(f"Change over the base: {ratio:.1%} of the block weights.")
    if ratio > 0.25:
        print("Warning: this is a heavy fine-tune (or --base is not the model it was trained from). A change this large replaces Ming's blocks and gives noise at strength 1; try strength 0.05-0.15, or use a lighter fine-tune.")
    if total:
        print(f"Rank {args.rank} keeps {kept / total:.0%} of the change's energy.")


if __name__ == "__main__":
    main()

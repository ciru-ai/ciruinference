# CiruInference: Laguna S 2.1 NVFP4

This branch is the exact runtime used to validate the 67.03 GiB
`Laguna-S-2.1-NVFP4.gguf` release.

## What changed

The base is upstream `ggml-org/llama.cpp` commit:

```text
1425386fd996511e1f3295e7366c38289a92a271
```

One focused change in `src/models/laguna.cpp` passes the model's NVFP4 scale
tensors into:

- the attention output gate;
- routed MoE up, gate, and down experts;
- the always-on shared expert;
- leading dense FFN layers; and
- the output projection.

There are no NixOS paths, CPU topology masks, or machine-specific compiler
flags in the source. The fast native NVFP4 kernel path targets NVIDIA
Blackwell. The published measurements used an RTX 5090.

## Hardware

Validated configuration:

- RTX 5090 with 32 GB VRAM;
- 64 GB system RAM;
- NVIDIA driver 610.43.02;
- CUDA 12.9;
- 65,536-token context;
- Q8_0 K/V cache;
- no DFlash or speculative decoding.

CUDA 12.8 or newer is required for Blackwell. CUDA 13 is recommended for a new
installation.

## Ubuntu / Debian build

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git curl libcurl4-openssl-dev

# Install the NVIDIA driver and CUDA toolkit from NVIDIA's repository.
nvidia-smi
nvcc --version

git clone https://github.com/ciru-ai/ciruinference.git
cd ciruinference
git checkout laguna-nvfp4-b10106

cmake -S . -B build-cuda \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --config Release -j "$(nproc)" \
  --target llama-server llama-cli
```

## Fedora build

```bash
sudo dnf install -y gcc-c++ cmake ninja-build git curl libcurl-devel

# Install the NVIDIA driver and CUDA toolkit from NVIDIA's repository.
nvidia-smi
nvcc --version

git clone https://github.com/ciru-ai/ciruinference.git
cd ciruinference
git checkout laguna-nvfp4-b10106

cmake -S . -B build-cuda \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --config Release -j "$(nproc)" \
  --target llama-server llama-cli
```

To cross-compile for an RTX 50-series target, add:

```text
-DCMAKE_CUDA_ARCHITECTURES=120
```

For a normal build on the target machine, let CMake/llama.cpp detect the local
CUDA device.

## Download the GGUF

```bash
hf download jcbtc/Laguna-S-2.1-NVFP4-GGUF \
  Laguna-S-2.1-NVFP4.gguf \
  --local-dir .
```

Expected file:

```text
size    71,977,030,080 bytes
sha256  5cf866a0b1531c62a6754e811b64b8bd867b9f5cd5d6a69b2cde04077d807e87
```

## Serve the winning 64K profile

```bash
./build-cuda/bin/llama-server \
  --model ./Laguna-S-2.1-NVFP4.gguf \
  --ctx-size 65536 \
  --parallel 1 \
  --batch-size 1024 \
  --ubatch-size 1024 \
  --threads 8 \
  --threads-batch 16 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --fit on \
  --fit-target 1024 \
  --fit-ctx 65536 \
  --load-mode none \
  --jinja \
  --spec-type none \
  --no-context-shift \
  --cache-ram 0 \
  --no-cache-idle-slots \
  --metrics \
  --slots \
  --no-webui \
  --host 0.0.0.0 \
  --port 8080 \
  --alias laguna-s21-nvfp4-64k
```

The command deliberately omits CPU affinity masks. It is portable across
mainstream Linux CPUs. Adjust `--threads` and `--threads-batch` if a different
CPU or memory subsystem performs better.

## OpenAI-compatible request

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "laguna-s21-nvfp4-64k",
    "messages": [{"role": "user", "content": "Implement a parallel file indexer in Rust."}],
    "temperature": 0,
    "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## Verified performance

The batch winner used the same uncached 65,269-token prompt for every lane,
then forced 128 decode tokens:

| batch / ubatch | prefill tok/s | decode tok/s |
|---|---:|---:|
| 512 / 512 | 751.85 | 24.44 |
| 1024 / 1024 | **1,201.93** | **23.75** |
| 2048 / 2048 | 1,703.50 | 22.78 |

`1024/1024` is the balanced production setting.

## Upstream status

As of this release, current upstream `llama.cpp` and Poolside's Laguna branch
did not contain the complete scale wiring used here. Once an equivalent fix is
merged upstream, this branch can be retired in favor of stock `llama.cpp`.

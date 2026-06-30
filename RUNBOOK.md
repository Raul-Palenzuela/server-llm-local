# Runbook

Step-by-step commands to set up this project on a new machine.

## 1. WSL2 (Windows only)
```bash
wsl --install
# reboot
wsl --update
```

## 2. Docker inside WSL2
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# close and reopen the terminal
docker run hello-world
```

## 3. GPU support for Docker
```bash
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu24.04 nvidia-smi
```
If this shows your GPU, everything below works without friction.

## 4. Clone and start
```bash
git clone https://github.com/Raul-Palenzuela/servidor-llm-local.git
cd servidor-llm-local
docker compose up
```
Wait for "Application startup complete".

## 5. Sanity check
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "messages": [{"role": "user", "content": "hi"}]}'
```

## 6. Backend
```bash
python3 -m venv backend-env
source backend-env/bin/activate
pip install fastapi uvicorn httpx
python3 backend.py
```

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: vllm._C_stable_libtorch` | running vLLM natively on Windows | use WSL2 |
| `Could not find nvcc` | missing CUDA Toolkit | install `cuda-toolkit-12-9`+ |
| `SM 12.x requires CUDA >= 12.9` | GPU too new for installed CUDA | upgrade CUDA Toolkit |
| `unsupported GNU version` | host GCC too new for CUDA | install GCC 13, set via `update-alternatives` |
| `No available memory for cache blocks` | unquantized model doesn't fit in VRAM | use an `-AWQ` model build |
| `"auto" tool choice requires --enable-auto-tool-choice` | tool-calling flags missing | add `--enable-auto-tool-choice --tool-call-parser hermes` |
| model emits tool calls as plain text instead of `tool_calls` | model/parser/chat-template mismatch | switch model (see README) |

Using Docker (recommended) resolves the CUDA/GCC/Python rows above out of
the box — they only matter if running vLLM natively.

# Servidor LLM local con vLLM (Docker)

Servidor de inferencia propio, reproducible en cualquier máquina con GPU NVIDIA.
Sirve un modelo cuantizado (AWQ) vía API compatible OpenAI. Sin depender de APIs de pago.

## Por qué este proyecto existe

Montar vLLM "a pelo" (sin Docker) requiere hacer coincidir manualmente: versión de CUDA,
versión de GCC, versión de Python, y arquitectura de GPU. Un solo desajuste rompe todo
el arranque. Este repo usa la imagen oficial de vLLM, que ya trae todo precompilado y
correcto, para que levantar el servidor sea un único comando, siempre.

## Requisitos del host (una vez por máquina nueva)

- GPU NVIDIA con al menos 8GB VRAM
- Docker instalado
- NVIDIA Container Toolkit instalado (ver instalación abajo)
- Si es Windows: WSL2 con Ubuntu

### Instalar Docker (dentro de WSL2 o Linux nativo)
\`\`\`bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# cierra y reabre la terminal
\`\`\`

### Instalar NVIDIA Container Toolkit (permite que Docker vea la GPU)
\`\`\`bash
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
\`\`\`

### Verificar que Docker ve la GPU
\`\`\`bash
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu24.04 nvidia-smi
\`\`\`
Si muestra tu GPU, todo lo de abajo funciona sin más pasos.

## Uso

\`\`\`bash
git clone <este-repo>
cd mi-servidor-llm
docker compose up
\`\`\`

Servidor disponible en `http://localhost:8000/v1` (API compatible OpenAI).
La primera vez descarga el modelo (~5GB AWQ), tarda unos minutos. Las siguientes
veces es instantáneo gracias al volumen `huggingface-cache`.

Para pararlo:
\`\`\`bash
docker compose down
\`\`\`

## Probar que funciona

\`\`\`bash
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "di hola"}]
  }'
\`\`\`

## Conectar herramientas de código (Aider)

\`\`\`bash
pip install aider-chat
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="no-hace-falta"
aider --model openai/Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
\`\`\`

## Cambiar de modelo

Edita `command:` en `docker-compose.yml`, cambia el nombre del modelo (cualquier
repo de Hugging Face compatible con vLLM), y `docker compose up` de nuevo.

## Por qué AWQ y no el modelo completo

Un modelo de 7B sin cuantizar ocupa ~14GB en VRAM. La versión AWQ ocupa ~5GB,
permitiendo correr en GPUs de consumo (8-12GB) con margen de sobra para el
KV cache de varias peticiones concurrentes (continuous batching).

## Arquitectura

\`\`\`
[Tu app / Aider / curl] → http://localhost:8000/v1 → [contenedor vLLM] → GPU
\`\`\`

El contenedor aísla CUDA/GCC/Python del sistema operativo del host. El host solo
necesita drivers NVIDIA + Docker + NVIDIA Container Toolkit.

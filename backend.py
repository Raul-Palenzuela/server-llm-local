"""
Backend orquestador: intermediario entre el frontend web y el modelo vLLM.
Decide si una petición es conversación normal o requiere acción sobre archivos,
y ejecuta esa acción él mismo (sin terminal, sin Aider).

Instalación:
    pip install fastapi uvicorn httpx

Ejecución:
    python backend.py
    (sirve en http://localhost:8001)
"""

import json
import os
import subprocess
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Configuración ---
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
WORKSPACE = Path.home() / "test-vllm"
WORKSPACE.mkdir(exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: list = []


# --- Definición de la herramienta que el modelo puede pedir usar ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Crea o sobrescribe un archivo con el contenido dado, dentro del workspace del usuario.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Nombre del archivo, ej: fib.py"},
                    "content": {"type": "string", "description": "Contenido completo del archivo"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Ejecuta un comando de shell dentro del workspace (por ejemplo, ejecutar un script Python ya creado).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Comando a ejecutar, ej: python3 fib.py"},
                },
                "required": ["command"],
            },
        },
    },
]


def write_file(filename: str, content: str) -> str:
    """Ejecuta la acción real: escribe el archivo en disco."""
    safe_name = os.path.basename(filename)  # evita salir del workspace
    path = WORKSPACE / safe_name
    path.write_text(content)
    return f"Archivo '{safe_name}' creado en {WORKSPACE}"


def run_command(command: str) -> str:
    """Ejecuta un comando real dentro del workspace, con límite de tiempo."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout + result.stderr
        return output[:2000] if output else "(sin salida)"
    except subprocess.TimeoutExpired:
        return "Error: el comando tardó demasiado y se canceló."


ACTIONS = {
    "write_file": write_file,
    "run_command": run_command,
}


@app.get("/health")
def health():
    return {"status": "ok", "workspace": str(WORKSPACE)}


@app.post("/chat")
async def chat(req: ChatRequest):
    messages = req.history + [{"role": "user", "content": req.message}]

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            VLLM_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
            },
        )
        data = response.json()

    choice = data["choices"][0]
    message = choice["message"]

    # Si el modelo no pidió ninguna herramienta, es conversación normal
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return {
            "reply": message["content"],
            "actions": [],
        }

    # El modelo pidió ejecutar una o varias acciones: las ejecutamos de verdad
    actions_log = []
    for call in tool_calls:
        fn_name = call["function"]["name"]
        try:
            args = json.loads(call["function"]["arguments"])
        except json.JSONDecodeError:
            args = {}

        if fn_name in ACTIONS:
            result = ACTIONS[fn_name](**args)
            actions_log.append({"action": fn_name, "args": args, "result": result})

    # Devolvemos al modelo el resultado de las acciones para que redacte la respuesta final
    messages.append(message)
    for call, log in zip(tool_calls, actions_log):
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": log["result"],
        })

    async with httpx.AsyncClient(timeout=120) as client:
        final_response = await client.post(
            VLLM_URL,
            json={"model": MODEL_NAME, "messages": messages},
        )
        final_data = final_response.json()

    final_text = final_data["choices"][0]["message"]["content"]

    return {
        "reply": final_text,
        "actions": actions_log,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

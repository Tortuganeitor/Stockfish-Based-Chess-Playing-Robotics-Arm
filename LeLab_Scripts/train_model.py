#!/usr/bin/env python3
"""
Entrenamiento (fine-tuning) de SmolVLA con un dataset grabado con LeRobot.

Uso (con el entorno (lerobot) activo):
    python3 train_V01.py

Parte del modelo preentrenado lerobot/smolvla_base y lo ajusta a tu dataset local
(carpetas data/, meta/ y videos/). Para ejecutar: lerobot-train, como en
https://huggingface.co/docs/lerobot/smolvla

Requisitos: pip install -e ".[smolvla]"  (desde ~/lerobot) y conexión a internet la primera
vez, para descargar el modelo base de Hugging Face.
"""

import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ============================================================
# CONFIGURACIÓN (cambia aquí lo que necesites)
# ============================================================

# --- Dataset (la carpeta que contiene data/, meta/ y videos/) ---
DATASET_ROOT = Path("/home/agirre/lerobot/ourprograms/datasets")
DATASET_REPO_ID = "local/datasets"   # solo es una etiqueta: con DATASET_ROOT no se descarga nada

# --- Modelo ---
BASE_POLICY = "lerobot/smolvla_base"  # modelo preentrenado que se va a ajustar

# --- Entrenamiento ---
STEPS = 20000          # pasos de entrenamiento (la documentación usa 20000 como referencia)
BATCH_SIZE = 8         # si da "CUDA out of memory", bájalo (4, 2...); si te sobra memoria, súbelo
SAVE_FREQ = 5000       # guarda un checkpoint cada tantos pasos
NUM_WORKERS = 4        # procesos que leen los datos
DEVICE = "auto"        # "auto" | "cuda" | "cpu" | "mps"
TRACKIO = False        # True = guarda curvas de entrenamiento (ver con: trackio show --project lerobot)

# --- Salida ---
OUTPUT_BASE = Path("/home/agirre/lerobot/ourprograms/outputs/train")
JOB_NAME = "smolvla_so101"           # cada ejecución crea: OUTPUT_BASE/<JOB_NAME>_<fecha_hora>

# --- Subir el modelo a Hugging Face (requiere login y un repo propio) ---
PUSH_TO_HUB = False
POLICY_REPO_ID = ""                  # p. ej. "tu_usuario/mi_smolvla" (solo si PUSH_TO_HUB = True)

# --- Cámaras ---
# smolvla_base espera las cámaras con los nombres camera1, camera2 y camera3.
# None = automático: tus cámaras se asignan por orden alfabético (p. ej. top -> camera1,
# wrist -> camera2). Si quieres otro reparto, escríbelo aquí, por ejemplo:
# CAMERA_MAP = {"observation.images.wrist": "observation.images.camera1"}
CAMERA_MAP = None

# Pide confirmación (Enter) antes de empezar un entrenamiento que puede durar horas.
CONFIRMAR = True

# ============================================================

MIN_EPISODIOS_RECOMENDADOS = 50  # referencia de la documentación de SmolVLA


def leer_dataset() -> dict:
    """Comprueba que la carpeta tiene data/, meta/ y videos/ y devuelve meta/info.json."""
    if not DATASET_ROOT.is_dir():
        sys.exit(f"No existe la carpeta del dataset: {DATASET_ROOT}")
    faltan = [c for c in ("data", "meta", "videos") if not (DATASET_ROOT / c).is_dir()]
    if faltan:
        sys.exit(f"Faltan carpetas en {DATASET_ROOT}: {', '.join(faltan)}")
    info_path = DATASET_ROOT / "meta" / "info.json"
    if not info_path.exists():
        sys.exit(f"No se encuentra {info_path}: no parece un dataset de LeRobot.")
    with open(info_path, encoding="utf-8") as f:
        return json.load(f)


def camaras_del_dataset(info: dict) -> list[str]:
    return sorted(k for k in info.get("features", {}) if k.startswith("observation.images."))


def construir_mapa_camaras(camaras: list[str]) -> dict[str, str]:
    if CAMERA_MAP is not None:
        return dict(CAMERA_MAP)
    if len(camaras) > 3:
        sys.exit(f"El dataset tiene {len(camaras)} cámaras y smolvla_base admite 3. Define CAMERA_MAP.")
    mapa = {c: f"observation.images.camera{i}" for i, c in enumerate(camaras, start=1)}
    return {origen: destino for origen, destino in mapa.items() if origen != destino}


def elegir_dispositivo() -> str:
    try:
        import torch
    except ImportError:
        print("Aviso: no se pudo importar torch; se usará CPU para el cálculo.")
        return "cpu" if DEVICE == "auto" else DEVICE

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU detectada: {props.name} ({props.total_memory / 1024**3:.1f} GB de memoria)")
    if DEVICE != "auto":
        return DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def construir_comando(output_dir: Path, device: str, mapa_camaras: dict[str, str]) -> list[str]:
    comando = [
        "lerobot-train",
        f"--policy.path={BASE_POLICY}",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--dataset.root={DATASET_ROOT}",
        f"--batch_size={BATCH_SIZE}",
        f"--steps={STEPS}",
        f"--save_freq={SAVE_FREQ}",
        f"--num_workers={NUM_WORKERS}",
        f"--output_dir={output_dir}",
        f"--job_name={JOB_NAME}",
        f"--policy.device={device}",
    ]
    if PUSH_TO_HUB:
        if not POLICY_REPO_ID:
            sys.exit("PUSH_TO_HUB = True requiere rellenar POLICY_REPO_ID.")
        comando += ["--policy.push_to_hub=true", f"--policy.repo_id={POLICY_REPO_ID}"]
    else:
        comando.append("--policy.push_to_hub=false")
    if mapa_camaras:
        # Equivale a --rename_map='{"observation.images.wrist": "observation.images.camera1"}'
        comando.append("--rename_map=" + json.dumps(mapa_camaras))
    if TRACKIO:
        comando.append("--trackio.enable=true")
    return comando


def main() -> int:
    if shutil.which("lerobot-train") is None:
        sys.exit("No se encuentra 'lerobot-train'. Activa el entorno: conda activate lerobot")

    info = leer_dataset()
    episodios = info.get("total_episodes", "?")
    frames = info.get("total_frames", "?")
    camaras = camaras_del_dataset(info)
    mapa_camaras = construir_mapa_camaras(camaras)

    print(f"Dataset: {DATASET_ROOT}")
    print(f"  Episodios: {episodios} | Frames: {frames} | FPS: {info.get('fps', '?')}")
    print(f"  Cámaras: {', '.join(c.removeprefix('observation.images.') for c in camaras) or 'ninguna'}")
    if mapa_camaras:
        print("  Renombrado de cámaras para SmolVLA:")
        for origen, destino in mapa_camaras.items():
            print(f"    {origen} -> {destino}")
    if isinstance(episodios, int) and episodios < MIN_EPISODIOS_RECOMENDADOS:
        print(
            f"\nAviso: hay {episodios} episodios. La documentación de SmolVLA recomienda unos "
            f"{MIN_EPISODIOS_RECOMENDADOS} como punto de partida; con pocos, el resultado será limitado."
        )
    if not camaras:
        sys.exit("El dataset no tiene cámaras (observation.images.*): SmolVLA las necesita.")

    device = elegir_dispositivo()
    if device == "cpu":
        print("Aviso: entrenar SmolVLA en CPU es extremadamente lento. Se recomienda una GPU.")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    output_dir = OUTPUT_BASE / f"{JOB_NAME}_{datetime.now():%Y%m%d_%H%M%S}"

    comando = construir_comando(output_dir, device, mapa_camaras)
    print("\nComando que se va a ejecutar:\n  " + " ".join(shlex.quote(c) for c in comando))
    print(f"\nPasos: {STEPS} | batch: {BATCH_SIZE} | dispositivo: {device}")
    print(f"Resultados en: {output_dir}\n")

    if CONFIRMAR:
        try:
            input("Pulsa Enter para empezar el entrenamiento (Ctrl+C para cancelar)... ")
        except (KeyboardInterrupt, EOFError):
            print("\nCancelado.")
            return 130

    proc = subprocess.Popen(comando)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C: deteniendo el entrenamiento... (otro Ctrl+C para forzar)")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()

    if proc.returncode == 0:
        modelo = output_dir / "checkpoints" / "last" / "pretrained_model"
        print(f"\nEntrenamiento terminado. Modelo final en:\n  {modelo}")
        if mapa_camaras:
            print("\nAl usar el modelo con el robot, pasa el mismo --rename_map:")
            print(f"  --rename_map='{json.dumps(mapa_camaras)}'")
    else:
        print(f"\nlerobot-train terminó con código {proc.returncode}.")
        print('Si ves "ModuleNotFoundError", instala las dependencias de SmolVLA:')
        print('  cd ~/lerobot && pip install -e ".[smolvla]"')
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())

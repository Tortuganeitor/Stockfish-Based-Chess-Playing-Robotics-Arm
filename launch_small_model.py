#!/usr/bin/env python3
"""
ROBOCHESS - SO101 ACT launcher (versión Python)

Equivalente al script bash original. Uso:

    python3 robochase_launcher.py [MODEL]

Variables de entorno soportadas:
    ROBOT_PORT, ROBOT_ID, TOP_URL, TOP_CAM, WRIST_CAM,
    TASK, DURATION, FPS, WIDTH, HEIGHT, CREATE_TOP_VIRTUAL

Ejemplo:
    TASK='Move the chess piece from e2 to e4' DURATION=10 python3 robochase_launcher.py
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración (equivalente a las variables del script bash)
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Tortuganeitor/SO-101_Training_For_Stockfish_Robotics_Arm"

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL", DEFAULT_MODEL)
ROBOT_PORT = os.environ.get("ROBOT_PORT", "/dev/ttyACM0")
ROBOT_ID = os.environ.get("ROBOT_ID", "my_follower_arm")
TOP_URL = os.environ.get("TOP_URL", "http://127.0.0.1:5000/video")
TOP_CAM = os.environ.get("TOP_CAM", "/dev/video20")
WRIST_CAM = os.environ.get("WRIST_CAM", "/dev/video2")
TASK = os.environ.get("TASK", "")
DURATION = os.environ.get("DURATION", "10")
FPS = os.environ.get("FPS", "30")
WIDTH = os.environ.get("WIDTH", "640")
HEIGHT = os.environ.get("HEIGHT", "480")
CREATE_TOP_VIRTUAL = os.environ.get("CREATE_TOP_VIRTUAL", "1")

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
ffmpeg_proc: subprocess.Popen | None = None
tmp_dir = Path(tempfile.mkdtemp(prefix="robochase_"))


def cleanup() -> None:
    global ffmpeg_proc
    if ffmpeg_proc is not None:
        try:
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=5)
        except Exception:
            try:
                ffmpeg_proc.kill()
            except Exception:
                pass
        ffmpeg_proc = None
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _signal_handler(signum, frame):  # noqa: ARG001
    cleanup()
    sys.exit(1)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    cleanup()
    sys.exit(1)


def which_or_fail(cmd: str) -> None:
    if shutil.which(cmd) is None:
        fail(f"Falta el comando {cmd}")


# ---------------------------------------------------------------------------
# 1/5 - Descarga y verificación del checkpoint
# ---------------------------------------------------------------------------
def download_and_check_checkpoint(model: str, out_dir: Path) -> list[str]:
    print("[1/5] Descargando y comprobando el checkpoint...")
    # Comprobación de dependencias Python (equivalente al `python3 - <<PY`)
    try:
        import cv2  # noqa: F401
        import huggingface_hub  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError as e:
        fail(f"Falta un módulo Python: {e}")

    from huggingface_hub import snapshot_download

    p = Path(snapshot_download(repo_id=model))
    cfg_file = p / "config.json"
    if not cfg_file.is_file():
        fail("No existe config.json en el repositorio del modelo.")
    cfg = json.loads(cfg_file.read_text())
    if cfg.get("type") != "act":
        fail(f"El repositorio no declara ACT: type={cfg.get('type')!r}")

    weights = list(p.rglob("*.safetensors"))
    if not weights:
        fail("No se encontraron pesos .safetensors.")

    features = cfg.get("input_features") or {}
    keys = [
        k
        for k, v in features.items()
        if k.startswith("observation.images.")
        and str(v.get("type", "")).upper().endswith("VISUAL")
    ]
    if not keys:
        fail("El checkpoint no declara entradas observation.images.*.")

    names = [k.split("observation.images.", 1)[1] for k in keys]
    (out_dir / "camera_names.txt").write_text("\n".join(names) + "\n")
    print("ACT verificado. Cámaras que exige el checkpoint:", ", ".join(names))
    print("Pesos encontrados:", len(weights))
    return names


# ---------------------------------------------------------------------------
# 2/5 - Comprobación de brazo y cámaras
# ---------------------------------------------------------------------------
def setup_cameras(model_cameras: list[str]) -> str:
    print("[2/5] Comprobando brazo y cámaras...")
    if not Path(ROBOT_PORT).exists():
        fail(f"No se encuentra el brazo en {ROBOT_PORT}")

    needs_top = 0
    camera_args_parts: list[str] = []

    for name in model_cameras:
        if name == "top":
            device = TOP_CAM
            needs_top = 1
        elif name == "wrist":
            device = WRIST_CAM
        else:
            fail(
                f"El modelo requiere la cámara '{name}'. "
                "Configura su dispositivo antes de ejecutar."
            )
        camera_args_parts.append(
            f"{name}: {{type: opencv, index_or_path: {device}, "
            f"width: {WIDTH}, height: {HEIGHT}, fps: {FPS}}}"
        )

    if needs_top and not Path(TOP_CAM).exists():
        if CREATE_TOP_VIRTUAL != "1":
            fail(f"Falta {TOP_CAM} y CREATE_TOP_VIRTUAL=0.")
        subprocess.run(
            [
                "sudo", "modprobe", "v4l2loopback",
                "devices=1", "video_nr=20", "card_label=SO101_TOP",
                "exclusive_caps=1",
            ],
            check=True,
        )
        if not Path(TOP_CAM).exists():
            fail(f"No se pudo crear {TOP_CAM}.")

        global ffmpeg_proc
        ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", TOP_URL,
                "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}",
                "-pix_fmt", "yuyv422",
                "-f", "v4l2", TOP_CAM,
            ]
        )
        time.sleep(3)
        if ffmpeg_proc.poll() is not None:
            fail("No se pudo iniciar la cámara TOP.")

    return ", ".join(camera_args_parts)


# ---------------------------------------------------------------------------
# 3/5 - Comprobación de que llegan imágenes
# ---------------------------------------------------------------------------
def verify_frames(camera_names_file: Path) -> None:
    print("[3/5] Comprobando que llegan imágenes...")
    import cv2

    names = camera_names_file.read_text().splitlines()
    devices = {"top": TOP_CAM, "wrist": WRIST_CAM}
    for name in names:
        dev = devices[name]
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if not ok or frame is None:
            fail(f"No llega imagen desde {name}: {dev}")
        print(f"OK {name}: {dev}, {frame.shape[1]}x{frame.shape[0]}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== ROBOCHESS ACT / SO-101 ===")

    for cmd in ("python3", "ffmpeg", "lerobot-rollout", "v4l2-ctl"):
        which_or_fail(cmd)

    if not TASK:
        fail(
            "Define TASK con el texto EXACTO usado durante el entrenamiento. "
            "Ejemplo: TASK='Move the chess piece from e2 to e4' "
            "python3 robochase_launcher.py"
        )
    if not (DURATION.isdigit() and int(DURATION) >= 1):
        fail("DURATION debe ser un entero >= 1.")

    model_cameras = download_and_check_checkpoint(MODEL, tmp_dir)
    if not model_cameras:
        fail("No se pudieron leer las cámaras del modelo.")

    camera_args = setup_cameras(model_cameras)
    verify_frames(tmp_dir / "camera_names.txt")

    print("[4/5] Configuración:")
    print(f"Modelo: {MODEL}")
    print(f"Robot: {ROBOT_ID} ({ROBOT_PORT})")
    print(f"Cámaras del modelo: {' '.join(model_cameras)}")
    print(f"Tarea: {TASK}")
    print(f"Duración: {DURATION}s")
    print()
    print("SEGURIDAD: robot calibrado, zona despejada y corte de alimentación accesible.")
    print(f"La primera ejecución está limitada a {DURATION}s y requiere supervisión.")

    answer = input("Escribe ACTIVAR para iniciar: ")
    if answer != "ACTIVAR":
        print("Cancelado.")
        cleanup()
        return

    print("[5/5] Iniciando lerobot-rollout...")
    cmd = [
        "lerobot-rollout",
        "--strategy.type=base",
        f"--policy.path={MODEL}",
        "--robot.type=so101_follower",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.cameras={{{camera_args}}}",
        f"--task={TASK}",
        f"--duration={DURATION}",
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        cleanup()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        fail(f"Comando falló con código {e.returncode}: {e.cmd}")
    except KeyboardInterrupt:
        cleanup()
        sys.exit(1)
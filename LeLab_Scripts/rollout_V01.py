#!/usr/bin/env python3
"""
Ejecuta en el robot (SO-101 follower) un modelo entrenado con LeRobot (SmolVLA).

Uso (con el entorno (lerobot) activo):
    python3 rollout_V01.py

El robot se mueve SOLO, guiado por el modelo y por las cámaras. El brazo líder no se usa.
Para parar: Ctrl+C (el robot vuelve a su posición inicial; espera a que termine).

Si usas la cámara de red "top", lerobot_ipcam.py debe estar en la misma carpeta que este programa.
"""

import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# ============================================================
# CONFIGURACIÓN (cambia aquí lo que necesites)
# ============================================================

# --- Modelo ---
# None = usa el entrenamiento más reciente de TRAIN_OUTPUT_BASE (el que crea train_V01.py).
# O pon la ruta exacta, p. ej.: ".../smolvla_so101_20261121_150000/checkpoints/last/pretrained_model"
MODEL_PATH = None
TRAIN_OUTPUT_BASE = Path("/home/agirre/lerobot/ourprograms/outputs/train")

# --- Tarea (la instrucción que recibe el modelo; usa la misma que al grabar los datos) ---
TASK = "Describe aquí la tarea que hace el robot"
DURATION_S = 60          # segundos que se ejecuta el modelo (0 = sin límite, se para con Ctrl+C)
FPS = 30                 # debe coincidir con los FPS del dataset con el que se entrenó

# --- Brazo follower (el brazo líder no se usa) ---
FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_NAME = "my_follower_arm"
# Límite de movimiento por paso, en grados. None = sin límite. Para las primeras pruebas se
# recomienda un valor pequeño (p. ej. 10): el robot se mueve más despacio y es más seguro.
MAX_RELATIVE_TARGET = None
RETURN_TO_INITIAL_POSITION = True   # al terminar, el robot vuelve a la posición del inicio

# --- Cámaras (las mismas que al grabar los datos) ---
WRIST_CAM = 2                                        # webcam USB: Id que da lerobot-find-cameras
TOP_CAM_URL = "http://192.168.188.56:5000/video"     # cámara de red "top"; "" = no usarla
CAM_WIDTH = 640
CAM_HEIGHT = 480

# --- Ejecución del modelo ---
INFERENCE = "sync"       # "sync" = una predicción por paso | "rtc" = Real-Time Chunking (movimiento
                         # más fluido con modelos lentos como SmolVLA; pruébalo si va a tirones)
DEVICE = None            # None = el del modelo ("cuda" si hay GPU); o "cuda" / "cpu" / "mps"
DISPLAY_DATA = True      # abre Rerun para ver las cámaras en directo

# Pide confirmación (Enter) antes de mover el robot.
CONFIRMAR = True

# ============================================================

LANZADOR = Path(__file__).resolve().with_name("lerobot_ipcam.py")
PREFIJO_CAM = "observation.images."


def buscar_modelo() -> Path:
    """Devuelve la carpeta pretrained_model del modelo a ejecutar."""
    if MODEL_PATH:
        ruta = Path(MODEL_PATH).expanduser()
    else:
        candidatos = sorted(
            TRAIN_OUTPUT_BASE.glob("*/checkpoints/last/pretrained_model"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidatos:
            sys.exit(f"No hay modelos entrenados en {TRAIN_OUTPUT_BASE}. Entrena uno con train_V01.py.")
        ruta = candidatos[-1]
    if not (ruta / "config.json").exists():
        sys.exit(f"{ruta} no parece un modelo de LeRobot (falta config.json).")
    return ruta.resolve()


def leer_rename_map(modelo: Path) -> dict[str, str] | None:
    """Lee el renombrado de cámaras con el que se entrenó (train_config.json). None = no disponible."""
    try:
        with open(modelo / "train_config.json", encoding="utf-8") as f:
            return dict(json.load(f).get("rename_map") or {})
    except (OSError, ValueError):
        return None


def elegir_camaras(rename_map: dict[str, str] | None) -> list[str]:
    """Qué cámaras configurar: las que el modelo vio al entrenar (si se sabe) y que tenemos."""
    disponibles = ["wrist"] + (["top"] if TOP_CAM_URL else [])
    if not rename_map:
        return disponibles
    necesarias = [k.removeprefix(PREFIJO_CAM) for k in rename_map]
    faltan = [c for c in necesarias if c not in disponibles]
    if faltan:
        sys.exit(
            f"El modelo se entrenó con las cámaras {necesarias}, pero aquí no hay: {faltan}. "
            "Revisa WRIST_CAM y TOP_CAM_URL."
        )
    return [c for c in disponibles if c in necesarias]


def camaras_cli(camaras: list[str]) -> str:
    """Valor de --robot.cameras. Las llaves se duplican ({{ }}) porque son f-strings."""
    comunes = f"width: {CAM_WIDTH}, height: {CAM_HEIGHT}, fps: {FPS}"
    partes = []
    if "top" in camaras:
        partes.append(f'top: {{type: ipcam, url: "{TOP_CAM_URL}", {comunes}}}')
    if "wrist" in camaras:
        partes.append(f"wrist: {{type: opencv, index_or_path: {WRIST_CAM}, {comunes}}}")
    return "{ " + ", ".join(partes) + " }"


def comprobar_stream(url: str) -> bool:
    """Comprueba que la cámara de red responde antes de mover el robot (falla rápido, en 5 s)."""
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            tipo = r.headers.get("Content-Type", "desconocido")
            r.read(1024)
    except Exception as e:
        print(f"No se puede leer la cámara de red {url}\n  Motivo: {e}")
        print("  Comprueba que el stream está en marcha y que estás en la misma red.")
        return False
    print(f"Cámara de red OK: {url} ({tipo})")
    return True


def bool_cli(valor: bool) -> str:
    return "true" if valor else "false"


def construir_comando(modelo: Path, rename_map: dict[str, str] | None, camaras: list[str]) -> list[str]:
    # Con cámara de red se usa el lanzador (añade el tipo "ipcam" a lerobot-rollout).
    base = [sys.executable, str(LANZADOR), "rollout"] if "top" in camaras else ["lerobot-rollout"]
    comando = base + [
        "--strategy.type=base",                 # ejecuta el modelo sin grabar datos
        f"--policy.path={modelo}",
        "--robot.type=so101_follower",
        f"--robot.port={FOLLOWER_PORT}",
        f"--robot.id={FOLLOWER_NAME}",
        f"--robot.cameras={camaras_cli(camaras)}",
        f"--task={TASK}",
        f"--duration={DURATION_S}",
        f"--fps={FPS}",
        f"--inference.type={INFERENCE}",
        f"--return_to_initial_position={bool_cli(RETURN_TO_INITIAL_POSITION)}",
        f"--display_data={bool_cli(DISPLAY_DATA)}",
    ]
    if MAX_RELATIVE_TARGET is not None:
        comando.append(f"--robot.max_relative_target={MAX_RELATIVE_TARGET}")
    if DEVICE:
        comando.append(f"--device={DEVICE}")
    if rename_map:
        # El mismo renombrado de cámaras que se usó al entrenar
        comando.append("--rename_map=" + json.dumps(rename_map))
    return comando


# ------------------------------------------------------------
# Barra de estado (tiempo restante)
# ------------------------------------------------------------
salida_lock = threading.Lock()  # evita que la barra y los mensajes de lerobot se mezclen


def mmss(segundos: int) -> str:
    return f"{segundos // 60:02d}:{segundos % 60:02d}"


class Estado:
    def __init__(self) -> None:
        self.fase = None      # None | "ejecutando"
        self.inicio = 0.0

    def poner(self, fase) -> None:
        self.inicio = time.monotonic()
        self.fase = fase

    def texto(self) -> str:
        if self.fase != "ejecutando":
            return ""
        transcurrido = time.monotonic() - self.inicio
        if DURATION_S <= 0:
            return f"● EJECUTANDO el modelo · tiempo transcurrido {mmss(int(transcurrido))} · Ctrl+C para parar"
        restante = max(0, math.ceil(DURATION_S - transcurrido))
        if restante == 0:
            return ">>> Tiempo cumplido · finalizando..."
        return f"● EJECUTANDO el modelo · tiempo restante {mmss(restante)} · Ctrl+C para parar"


def actualizar_estado(linea: str, estado: Estado) -> None:
    if "starting rollout" in linea:
        estado.poner("ejecutando")
    elif any(t in linea for t in ("Rollout finished", "Interrupted by user", "terminal fault")):
        estado.fase = None


def pintar(estado: Estado, parar: threading.Event) -> None:
    """Hilo que redibuja la barra de estado 5 veces por segundo."""
    hay_barra = False
    while not parar.is_set():
        texto = estado.texto()
        with salida_lock:
            if texto:
                sys.stdout.write("\r\033[K" + texto)
                sys.stdout.flush()
                hay_barra = True
            elif hay_barra:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                hay_barra = False
        parar.wait(0.2)
    with salida_lock:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


def ejecutar(comando: list[str], estado: Estado) -> int:
    # La salida de lerobot pasa por aquí para saber cuándo empieza el movimiento.
    proc = subprocess.Popen(
        comando,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    interrumpido = False
    while True:
        try:
            for linea in proc.stdout:
                actualizar_estado(linea, estado)
                with salida_lock:
                    sys.stdout.write("\r\033[K" + linea)  # borra la barra y escribe el mensaje
                    sys.stdout.flush()
            break
        except KeyboardInterrupt:
            estado.fase = None
            if interrumpido:
                print("\nSegundo Ctrl+C: forzando la salida (el robot puede quedar en su última pose).")
                proc.terminate()
            else:
                print("\nCtrl+C: parando; el robot vuelve a su posición inicial... (otro Ctrl+C para forzar)")
            interrumpido = True
    estado.fase = None
    proc.wait()
    return proc.returncode


def main() -> int:
    modelo = buscar_modelo()
    rename_map = leer_rename_map(modelo)
    camaras = elegir_camaras(rename_map)

    print(f"Modelo: {modelo}")
    if rename_map is None:
        print("Aviso: no se encontró train_config.json; no se aplica renombrado de cámaras.")
    elif rename_map:
        print("Cámaras que ve el modelo (renombrado usado al entrenar):")
        for origen, destino in rename_map.items():
            print(f"  {origen} -> {destino}")
    print(f"Cámaras en uso: {', '.join(camaras)}")

    if "top" in camaras:
        if not LANZADOR.exists():
            print(f"Falta {LANZADOR.name} junto a este programa (se usa para la cámara de red).")
            return 1
        if not comprobar_stream(TOP_CAM_URL):
            return 1

    comando = construir_comando(modelo, rename_map, camaras)
    print("\nComando que se va a ejecutar:\n  " + " ".join(shlex.quote(c) for c in comando))
    limite = f"{DURATION_S} s" if DURATION_S > 0 else "sin límite"
    print(f"\nTarea: {TASK!r} | duración: {limite} | inferencia: {INFERENCE}")
    print(
        "\nATENCIÓN: el robot se moverá solo. Despeja la zona de trabajo, mantén la mano cerca de la\n"
        "alimentación del brazo y no te acerques. El movimiento empieza cuando el modelo termina de\n"
        "cargarse (puede tardar); verás '● EJECUTANDO' en pantalla.\n"
    )
    if CONFIRMAR:
        try:
            input("Pulsa Enter para empezar (Ctrl+C para cancelar)... ")
        except (KeyboardInterrupt, EOFError):
            print("\nCancelado.")
            return 130

    estado = Estado()
    parar = threading.Event()
    hilo = threading.Thread(target=pintar, args=(estado, parar), daemon=True)
    hilo.start()
    try:
        codigo = ejecutar(comando, estado)
    finally:
        parar.set()
        hilo.join(timeout=1)

    if codigo == 0:
        print("\nEjecución terminada.")
    else:
        print(f"\nlerobot-rollout terminó con código {codigo}.")
        print('Si ves "ModuleNotFoundError", instala las dependencias de SmolVLA:')
        print('  cd ~/lerobot && pip install -e ".[smolvla]"')
    return codigo


if __name__ == "__main__":
    sys.exit(main())

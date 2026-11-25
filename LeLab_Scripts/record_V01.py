#!/usr/bin/env python3
"""
Grabación de datasets con LeRobot (SO-101) con barra de estado y cuenta atrás.

Uso:
    python3 record_V01.py

Mientras grabas verás en una línea: el episodio actual, cuántos faltan y el tiempo
restante del episodio. Entre episodios verás la cuenta atrás de recolocación.

Teclas durante la grabación (con la terminal enfocada):
    →  (flecha derecha)  termina antes el episodio actual o salta la espera
    ←  (flecha izquierda) descarta y repite el episodio anterior
    Esc                  termina la sesión y guarda lo grabado
"""

import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# CONFIGURACIÓN (cambia aquí lo que necesites)
# ============================================================

# --- Brazos ---
LEADER_PORT = "/dev/ttyACM1"
LEADER_NAME = "my_leader_arm"
FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_NAME = "my_follower_arm"

# --- Cámaras (el Id que muestra lerobot-find-cameras) ---
WRIST_CAM = 2

# --- Dataset ---
DATASETS_DIR = Path("/home/agirre/lerobot/ourprograms/datasets")  # cada dataset será una carpeta aquí dentro
HF_USER = "local"                  # tu usuario de Hugging Face si vas a subirlo; "local" si no
DATASET_NAME = "so101_prueba"      # nombre del dataset (en cada ejecución se le añade fecha y hora)
TASK = "Describe aquí la tarea que hace el robot"  # texto que se guarda con cada episodio
NUM_EPISODES = 200                   # episodios que se graban en ESTE dataset
EPISODE_TIME_S = 8                # duración de cada episodio (s)
RESET_TIME_S = 15                  # espera entre episodios para recolocar todo (s)
START_DELAY_S = 5                  # cuenta atrás antes del primer episodio (0 = sin cuenta atrás)
FPS = 30
PUSH_TO_HUB = False                # True = subir a Hugging Face al terminar (requiere login)
PLAY_SOUNDS = True                 # lerobot anuncia por voz "Reset the environment", etc.

# Para AÑADIR episodios a un dataset que ya existe, pon RESUME = True y en DATASET_NAME
# el nombre EXACTO de su carpeta, con la fecha incluida (el programa lista los disponibles).
RESUME = False

# ============================================================


def es_dataset(carpeta: Path) -> bool:
    return (carpeta / "meta" / "dataset.json").exists()


def listar_datasets() -> None:
    print(f"\nDatasets en {DATASETS_DIR}:")
    nombres = (
        sorted(p.name for p in DATASETS_DIR.iterdir() if p.is_dir() and es_dataset(p))
        if DATASETS_DIR.exists()
        else []
    )
    for nombre in nombres:
        print(f"  - {nombre}")
    if not nombres:
        print("  (ninguno)")


def bool_cli(valor: bool) -> str:
    return "true" if valor else "false"


def construir_comando(nombre: str, carpeta: Path) -> list[str]:
    # Las llaves del diccionario se duplican ({{ }}) porque es una f-string.
    camaras = (
        f"{{ wrist: {{type: opencv, index_or_path: {WRIST_CAM}, "
        f"width: 640, height: 480, fps: 30}} }}"
    )
    comando = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={FOLLOWER_PORT}",
        f"--robot.id={FOLLOWER_NAME}",
        f"--robot.cameras={camaras}",
        "--teleop.type=so101_leader",
        f"--teleop.port={LEADER_PORT}",
        f"--teleop.id={LEADER_NAME}",
        "--display_data=true",
        f"--dataset.repo_id={HF_USER}/{nombre}",
        f"--dataset.root={carpeta}",           # carpeta exacta donde se guarda este dataset
        "--dataset.no_stamp=true",             # el nombre ya lleva fecha; lerobot no añade otra
        f"--dataset.single_task={TASK}",
        f"--dataset.num_episodes={NUM_EPISODES}",
        f"--dataset.episode_time_s={EPISODE_TIME_S}",
        f"--dataset.reset_time_s={RESET_TIME_S}",
        f"--dataset.fps={FPS}",
        "--dataset.streaming_encoding=true",   # guarda cada episodio casi al instante
        "--dataset.encoder_threads=2",
        f"--dataset.push_to_hub={bool_cli(PUSH_TO_HUB)}",
        f"--play_sounds={bool_cli(PLAY_SOUNDS)}",
    ]
    if RESUME:
        comando.append("--resume=true")
    return comando


# ------------------------------------------------------------
# Barra de estado (episodio, episodios restantes y cuenta atrás)
# ------------------------------------------------------------
salida_lock = threading.Lock()  # evita que la barra y los mensajes de lerobot se mezclen
PATRON_EPISODIO = re.compile(r"Recording episode (\d+)")


class Estado:
    """Qué está pasando ahora mismo; de aquí sale el texto de la barra."""

    def __init__(self) -> None:
        self.fase = None          # None | "inicio" | "grabando" | "reset"
        self.fin = 0.0            # instante en que acaba la fase actual
        self.rep = 0              # número de episodio de ESTA sesión (1, 2, 3...)
        self.primer_idx = None    # índice del primer episodio que anuncia lerobot
        self.repetir = False      # True si se va a repetir el episodio (flecha izquierda)

    def poner(self, fase: str, segundos: float = 0.0) -> None:
        self.fin = time.monotonic() + segundos
        self.fase = fase

    def texto(self) -> str:
        fase = self.fase
        if fase is None:
            return ""
        restante = max(0, math.ceil(self.fin - time.monotonic()))
        t = f"{restante // 60:02d}:{restante % 60:02d}"
        faltan = NUM_EPISODES - self.rep
        if fase == "inicio":
            return f">>> Preparados · la grabación empieza en {t}"
        if fase == "grabando":
            if restante == 0:
                return f">>> Episodio {self.rep}/{NUM_EPISODES} terminado · guardando..."
            return f"● GRABANDO episodio {self.rep}/{NUM_EPISODES} · faltan {faltan} · tiempo restante {t}"
        if self.repetir:
            return f">>> Se repetirá el episodio {self.rep}/{NUM_EPISODES} · recoloca el robot · empieza en {t}"
        return f">>> Episodio {self.rep}/{NUM_EPISODES} terminado · recoloca el robot · siguiente en {t}"


def actualizar_estado(linea: str, estado: Estado) -> None:
    """Interpreta lo que imprime lerobot para saber en qué fase estamos."""
    m = PATRON_EPISODIO.search(linea)
    if m:
        idx = int(m.group(1))
        if estado.primer_idx is None:
            estado.primer_idx = idx
        estado.rep = idx - estado.primer_idx + 1
        estado.repetir = False
        estado.poner("grabando", EPISODE_TIME_S)
    elif "Reset the environment" in linea:
        estado.poner("reset", RESET_TIME_S)
    elif "key pressed" in linea:  # has pulsado →, ← o Esc: la fase actual termina
        if "Left arrow" in linea:
            estado.repetir = True
        estado.fase = None
    elif "Re-record episode" in linea or "Stop recording" in linea:
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
    # La salida de lerobot pasa por aquí para saber en qué episodio estamos.
    # stdin NO se redirige: así las teclas de control siguen funcionando.
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
            break  # lerobot cerró su salida: ha terminado
        except KeyboardInterrupt:
            estado.fase = None
            if interrumpido:
                print("\nSegundo Ctrl+C: forzando la salida.")
                proc.terminate()
            else:
                print("\nCtrl+C: cerrando y guardando lo grabado... (otro Ctrl+C para forzar)")
            interrumpido = True

    estado.fase = None
    proc.wait()
    return proc.returncode


def main() -> int:
    if RESUME:
        nombre = DATASET_NAME
        carpeta = DATASETS_DIR / nombre
        if not es_dataset(carpeta):
            print(f"RESUME=True, pero no existe el dataset '{carpeta}'.")
            listar_datasets()
            return 1
    else:
        nombre = f"{DATASET_NAME}_{datetime.now():%Y%m%d_%H%M%S}"
        carpeta = DATASETS_DIR / nombre

    comando = construir_comando(nombre, carpeta)
    print("Comando que se va a ejecutar:\n  " + " ".join(shlex.quote(c) for c in comando))
    print(
        f"\nDataset: {carpeta}\n{NUM_EPISODES} episodios de {EPISODE_TIME_S} s | "
        f"espera entre episodios: {RESET_TIME_S} s\n"
    )

    estado = Estado()
    parar = threading.Event()
    hilo = threading.Thread(target=pintar, args=(estado, parar), daemon=True)
    hilo.start()

    try:
        if START_DELAY_S > 0:
            estado.poner("inicio", START_DELAY_S)
            time.sleep(START_DELAY_S)
            estado.fase = None
        codigo = ejecutar(comando, estado)
    except KeyboardInterrupt:
        parar.set()
        hilo.join(timeout=1)
        print("\nCancelado antes de empezar a grabar.")
        return 130
    finally:
        parar.set()
        hilo.join(timeout=1)

    if codigo == 0:
        print(f"\nGrabación terminada. Dataset guardado en: {carpeta}")
    else:
        print(f"\nlerobot-record terminó con código {codigo}.")
    listar_datasets()
    return codigo


if __name__ == "__main__":
    sys.exit(main())
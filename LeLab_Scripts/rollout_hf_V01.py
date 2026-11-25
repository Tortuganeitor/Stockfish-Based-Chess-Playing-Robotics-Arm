#!/usr/bin/env python3
"""
Ejecuta en el robot (SO-101 follower) un modelo entrenado que está en Hugging Face.

Uso (con el entorno (lerobot) activo y la sesión iniciada: hf auth login):
    python3 rollout_hf_V01.py          # ejecuta el modelo indicado en MODEL_REPO
    python3 rollout_hf_V01.py --list   # solo muestra tus modelos disponibles

Hace falta conexión a internet al lanzarlo: lerobot lee el modelo del Hub (la primera vez lo
descarga y lo guarda en la caché). El brazo líder no se usa: el robot se mueve SOLO.
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

# --- Modelo en Hugging Face ---
# Repositorio del modelo: https://huggingface.co/Tortuganeitor/SO-101_Training_For_Stockfish_Robotics_Arm
# (pon None para usar automáticamente el modelo más reciente de tu cuenta cuyo nombre empiece por
# MODEL_PREFIX; train_hf_V01.py los crea como "<usuario>/act_so101_<fecha_hora>").
MODEL_REPO = "Tortuganeitor/SO-101_Training_For_Stockfish_Robotics_Arm"
MODEL_PREFIX = "act_so101"

# --- Tarea (ACT no usa texto; solo hace falta para modelos con instrucciones, como SmolVLA) ---
TASK = ""
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
TOP_CAM_URL = "http://127.0.0.1:5000/video"     # cámara de red "top"; "" = no usarla
CAM_WIDTH = 640
CAM_HEIGHT = 480

# --- Otros ---
DEVICE = None            # None = el del modelo (si no hay GPU, lerobot usa la CPU); o "cuda" / "cpu"
DISPLAY_DATA = True      # abre Rerun para ver las cámaras en directo
CONFIRMAR = True         # pide Enter antes de mover el robot

# ============================================================

LANZADOR = Path(__file__).resolve().with_name("lerobot_ipcam.py")
PREFIJO_CAM = "observation.images."


# ------------------------------------------------------------
# Hugging Face: sesión y búsqueda del modelo
# ------------------------------------------------------------
def sesion_hf() -> tuple[str, str]:
    """Devuelve (token, usuario) de la sesión iniciada con `hf auth login`."""
    from huggingface_hub import HfApi, get_token

    token = get_token()
    if not token:
        sys.exit("No has iniciado sesión en Hugging Face. Ejecuta:  hf auth login")
    try:
        usuario = HfApi(token=token).whoami(token=token)["name"]
    except Exception as e:
        sys.exit(f"No se pudo comprobar la sesión de Hugging Face (¿hay internet?): {e}")
    return token, usuario


def token_opcional() -> str | None:
    """Token de la sesión si existe. Un modelo público se puede leer sin sesión; uno privado, no."""
    from huggingface_hub import get_token

    return get_token()


def listar_modelos(token: str, usuario: str) -> list[str]:
    """Repos de modelos de tu cuenta que empiezan por MODEL_PREFIX, del más reciente al más antiguo."""
    from huggingface_hub import HfApi

    modelos = HfApi(token=token).list_models(author=usuario, search=MODEL_PREFIX, limit=200)
    ids = [m.id for m in modelos if m.id.split("/")[-1].startswith(MODEL_PREFIX)]
    return sorted(ids, key=lambda i: i.split("/")[-1], reverse=True)  # el nombre acaba en fecha_hora


def modelo_completo(token: str, repo_id: str) -> bool:
    """Un entrenamiento terminado tiene config.json y model.safetensors (uno en curso o cortado, no)."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        return api.file_exists(repo_id, "config.json") and api.file_exists(repo_id, "model.safetensors")
    except Exception:
        return False


def leer_json_hub(token: str, repo_id: str, archivo: str) -> dict | None:
    """Descarga (o toma de la caché) un JSON del repo del modelo. None si no existe."""
    from huggingface_hub import hf_hub_download

    try:
        ruta = hf_hub_download(repo_id, archivo, token=token)
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def elegir_modelo(token: str, usuario: str) -> str:
    if MODEL_REPO:
        if not modelo_completo(token, MODEL_REPO):
            sys.exit(
                f"No se puede leer {MODEL_REPO}: no existe, no tiene un modelo terminado (config.json y "
                "model.safetensors) o es privado y no has iniciado sesión (hf auth login)."
            )
        return MODEL_REPO
    for repo in listar_modelos(token, usuario):
        if modelo_completo(token, repo):
            return repo
    sys.exit(
        f"No hay modelos terminados en tu cuenta que empiecen por '{MODEL_PREFIX}'.\n"
        "Comprueba que el entrenamiento terminó, o indica el repositorio en MODEL_REPO."
    )


# ------------------------------------------------------------
# Cámaras y comando
# ------------------------------------------------------------
def camaras_necesarias(config: dict | None, rename_map: dict | None) -> list[str]:
    """Qué cámaras configurar: las que el modelo vio al entrenar, si están disponibles."""
    disponibles = ["wrist"] + (["top"] if TOP_CAM_URL else [])
    if rename_map:  # p. ej. SmolVLA: las claves son los nombres de cámara del robot
        necesarias = [k.removeprefix(PREFIJO_CAM) for k in rename_map]
    else:           # p. ej. ACT: los nombres vienen en las entradas de visión del modelo
        entradas = (config or {}).get("input_features", {})
        necesarias = [k.removeprefix(PREFIJO_CAM) for k in entradas if k.startswith(PREFIJO_CAM)]
    if not necesarias:
        return disponibles
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


def construir_comando(repo_id: str, rename_map: dict | None, camaras: list[str]) -> list[str]:
    # Con cámara de red se usa el lanzador (añade el tipo "ipcam" a lerobot-rollout).
    base = [sys.executable, str(LANZADOR), "rollout"] if "top" in camaras else ["lerobot-rollout"]
    comando = base + [
        "--strategy.type=base",                 # ejecuta el modelo sin grabar datos
        f"--policy.path={repo_id}",             # lerobot lo lee directamente del Hub
        "--robot.type=so101_follower",
        f"--robot.port={FOLLOWER_PORT}",
        f"--robot.id={FOLLOWER_NAME}",
        f"--robot.cameras={camaras_cli(camaras)}",
        f"--duration={DURATION_S}",
        f"--fps={FPS}",
        f"--return_to_initial_position={bool_cli(RETURN_TO_INITIAL_POSITION)}",
        f"--display_data={bool_cli(DISPLAY_DATA)}",
    ]
    if TASK:
        comando.append(f"--task={TASK}")
    if MAX_RELATIVE_TARGET is not None:
        comando.append(f"--robot.max_relative_target={MAX_RELATIVE_TARGET}")
    if DEVICE:
        comando.append(f"--device={DEVICE}")
    if rename_map:
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
    if MODEL_REPO and "--list" not in sys.argv[1:]:
        token, usuario = token_opcional(), None   # con el repo indicado no hace falta saber el usuario
    else:
        token, usuario = sesion_hf()

    if "--list" in sys.argv[1:]:
        repos = listar_modelos(token, usuario)
        print(f"Modelos de {usuario} que empiezan por '{MODEL_PREFIX}' (del más reciente al más antiguo):")
        for repo in repos:
            print(f"  - {repo}  [{'terminado' if modelo_completo(token, repo) else 'incompleto'}]")
        if not repos:
            print("  (ninguno)")
        return 0

    repo_id = elegir_modelo(token, usuario)
    config = leer_json_hub(token, repo_id, "config.json")
    train_cfg = leer_json_hub(token, repo_id, "train_config.json")
    rename_map = dict((train_cfg or {}).get("rename_map") or {})
    camaras = camaras_necesarias(config, rename_map)

    print(f"Modelo: https://huggingface.co/{repo_id}")
    if rename_map:
        print("Renombrado de cámaras usado al entrenar:")
        for origen, destino in rename_map.items():
            print(f"  {origen} -> {destino}")
    print(f"Cámaras en uso: {', '.join(camaras)}")

    if "top" in camaras:
        if not LANZADOR.exists():
            print(f"Falta {LANZADOR.name} junto a este programa (se usa para la cámara de red).")
            return 1
        if not comprobar_stream(TOP_CAM_URL):
            return 1

    comando = construir_comando(repo_id, rename_map, camaras)
    print("\nComando que se va a ejecutar:\n  " + " ".join(shlex.quote(c) for c in comando))
    limite = f"{DURATION_S} s" if DURATION_S > 0 else "sin límite"
    print(f"\nDuración: {limite}")
    print(
        "\nATENCIÓN: el robot se moverá solo. Despeja la zona de trabajo, mantén la mano cerca de la\n"
        "alimentación del brazo y no te acerques. El movimiento empieza cuando el modelo termina de\n"
        "cargarse (la primera vez se descarga del Hub); verás '● EJECUTANDO' en pantalla.\n"
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
    return codigo


if __name__ == "__main__":
    sys.exit(main())

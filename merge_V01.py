#!/usr/bin/env python3
"""
Une varios datasets de LeRobot (los de una carpeta) en uno nuevo, con lerobot-edit-dataset.

Uso (con el entorno (lerobot) activo):
    python3 merge_V01.py

- Busca en SOURCE_DIR las carpetas que son datasets (tienen meta/info.json).
- Comprueba que sean compatibles (mismos FPS, robot y cámaras), igual que lo hace lerobot.
- Crea un dataset NUEVO en OUTPUT_BASE; los originales no se tocan.
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

SOURCE_DIR = Path("/home/agirre/lerobot/ourprograms/true_datasets")  # carpeta con los datasets
ONLY = None      # None = todos los de SOURCE_DIR; o una lista de nombres: ["dataset_a", "dataset_b"]
EXCLUDE = []     # nombres de carpetas que NO se quieren unir

OUTPUT_BASE = Path("/home/agirre/lerobot/ourprograms/merged")  # el resultado va dentro de esta carpeta
OUTPUT_NAME = "so101_unido"      # nombre del dataset resultante (se le añade fecha y hora)
HF_USER = "local"                # usuario de Hugging Face; "local" si no se va a subir

PUSH_TO_HUB = False  # True = subir el resultado a Hugging Face (requiere login y HF_USER real)
CONFIRMAR = True     # pide Enter antes de unir

# ============================================================

PREFIJO_CAM = "observation.images."


def leer_info(carpeta: Path) -> dict | None:
    try:
        with open(carpeta / "meta" / "info.json", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def firma(info: dict) -> str:
    """Lo que lerobot exige que sea idéntico entre datasets: fps, robot y características."""
    return json.dumps(
        {"fps": info.get("fps"), "robot_type": info.get("robot_type"), "features": info.get("features")},
        sort_keys=True,
    )


def describir(info: dict) -> str:
    camaras = sorted(k.removeprefix(PREFIJO_CAM) for k in info.get("features", {}) if k.startswith(PREFIJO_CAM))
    return f"cámaras: {', '.join(camaras) or 'ninguna'} | fps: {info.get('fps')} | robot: {info.get('robot_type')}"


def main() -> int:
    if shutil.which("lerobot-edit-dataset") is None:
        sys.exit("No se encuentra 'lerobot-edit-dataset'. Activa el entorno: conda activate lerobot")
    if not SOURCE_DIR.is_dir():
        sys.exit(f"No existe la carpeta: {SOURCE_DIR}")

    # --- Buscar datasets ---
    datasets: list[tuple[str, dict]] = []
    for carpeta in sorted(p for p in SOURCE_DIR.iterdir() if p.is_dir()):
        nombre = carpeta.name
        if ONLY is not None and nombre not in ONLY:
            continue
        if nombre in EXCLUDE:
            continue
        info = leer_info(carpeta)
        if info is None:
            print(f"  (se ignora '{nombre}': no es un dataset de LeRobot)")
            continue
        if not info.get("total_episodes"):
            print(f"  (se ignora '{nombre}': no tiene episodios)")
            continue
        datasets.append((nombre, info))

    if len(datasets) < 2:
        sys.exit(f"Hacen falta al menos 2 datasets en {SOURCE_DIR}; se han encontrado {len(datasets)}.")

    print(f"\nDatasets encontrados en {SOURCE_DIR}:")
    for nombre, info in datasets:
        print(f"  - {nombre}: {info['total_episodes']} episodios")

    # --- Comprobar compatibilidad ---
    grupos: dict[str, list[tuple[str, dict]]] = {}
    for nombre, info in datasets:
        grupos.setdefault(firma(info), []).append((nombre, info))
    if len(grupos) > 1:
        print("\nNo se pueden unir: los datasets no son compatibles entre sí.")
        print("Hay estos grupos (dentro de cada grupo SÍ se pueden unir):")
        for i, miembros in enumerate(grupos.values(), start=1):
            print(f"\n  Grupo {i} ({len(miembros)} dataset{'s' if len(miembros) != 1 else ''}) - {describir(miembros[0][1])}")
            for nombre, _ in miembros:
                print(f"    - {nombre}")
        print("\nUsa ONLY o EXCLUDE al principio del programa para quedarte con un solo grupo.")
        return 1

    total = sum(info["total_episodes"] for _, info in datasets)
    print(f"\nTodos son compatibles ({describir(datasets[0][1])}). Total: {total} episodios.")

    # --- Construir el comando ---
    nombre_salida = f"{OUTPUT_NAME}_{datetime.now():%Y%m%d_%H%M%S}"
    salida = OUTPUT_BASE / nombre_salida
    if PUSH_TO_HUB and HF_USER == "local":
        sys.exit("Para subir a Hugging Face pon tu usuario real en HF_USER (no 'local').")

    repo_ids = [f"{HF_USER}/{nombre}" for nombre, _ in datasets]
    roots = [str(SOURCE_DIR / nombre) for nombre, _ in datasets]
    comando = [
        "lerobot-edit-dataset",
        f"--new_repo_id={HF_USER}/{nombre_salida}",
        f"--new_root={salida}",
        "--operation.type=merge",
        f"--operation.repo_ids={repo_ids}",
        f"--operation.roots={roots}",
    ]
    if PUSH_TO_HUB:
        comando.append("--push_to_hub=true")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    print("\nComando que se va a ejecutar:\n  " + " ".join(shlex.quote(c) for c in comando))
    print(f"\nEl dataset unido se guardará en: {salida}\n")

    if CONFIRMAR:
        try:
            input("Pulsa Enter para unir los datasets (Ctrl+C para cancelar)... ")
        except (KeyboardInterrupt, EOFError):
            print("\nCancelado.")
            return 130

    try:
        codigo = subprocess.run(comando).returncode
    except KeyboardInterrupt:
        print("\nInterrumpido.")
        return 130

    if codigo == 0:
        print(f"\nDatasets unidos ({total} episodios). Resultado en:\n  {salida}")
        print("\nPara entrenar con él, pon en train_V01.py:")
        print(f'  DATASET_ROOT = Path("{salida}")')
    else:
        print(f"\nlerobot-edit-dataset terminó con código {codigo}.")
        print("Si falla por dependencias, prueba: pip install 'lerobot[dataset]'")
    return codigo


if __name__ == "__main__":
    sys.exit(main())

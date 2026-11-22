#!/usr/bin/env python3
"""
Cámara de red (stream de vídeo por HTTP) para LeRobot + lanzador.

Por qué hace falta: el tipo "opencv" de LeRobot convierte index_or_path en una ruta de archivo
(Path) y eso estropea las URL ("http://..." pasa a "http:/..."). Además exige poder cambiar la
resolución y los FPS de la cámara, algo que un stream de red no permite. Este archivo añade un
tipo de cámara nuevo, "ipcam", que abre la URL tal cual, la acepta con la resolución y los FPS
que traiga el stream y reescala cada fotograma al tamaño pedido.

Uso como lanzador (hace lo mismo que lerobot-record / lerobot-teleoperate, pero con "ipcam"
disponible):
    python3 lerobot_ipcam.py record      --robot.cameras="{ top: {type: ipcam, url: ...} }" ...
    python3 lerobot_ipcam.py teleoperate --robot.cameras="{ top: {type: ipcam, url: ...} }" ...
    python3 lerobot_ipcam.py rollout     --robot.cameras="{ top: {type: ipcam, url: ...} }" ...

Configuración de la cámara (dentro de --robot.cameras):
    top: {type: ipcam, url: "http://192.168.188.56:5000/video", width: 640, height: 480, fps: 30}
"""

import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.utils.errors import DeviceNotConnectedError


@CameraConfig.register_subclass("ipcam")
@dataclass
class IPCameraConfig(OpenCVCameraConfig):
    url: str = ""
    # Heredado de OpenCVCameraConfig; aquí no se usa (la cámara se abre con `url`).
    index_or_path: int | Path = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.url:
            raise ValueError("La cámara 'ipcam' necesita 'url', p. ej. http://192.168.1.10:5000/video")
        if self.rotation != Cv2Rotation.NO_ROTATION:
            raise ValueError("La cámara 'ipcam' no admite 'rotation'.")


class IPCamera(OpenCVCamera):
    """Cámara de red: abre la URL con OpenCV, lee en segundo plano y reescala al tamaño pedido."""

    def __init__(self, config: IPCameraConfig):
        super().__init__(config)
        self.url = config.url
        self.index_or_path = config.url  # connect() abre el stream con este valor, como cadena

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.url})"

    def _configure_capture_settings(self) -> None:
        # Un stream de red no permite cambiar resolución ni FPS: se aceptan los del stream
        # y cada fotograma se reescala después a (width, height).
        cap = self.videocapture
        if cap is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        if self.width is None or self.height is None:
            ok, frame = cap.read()
            if not ok:
                raise ConnectionError(f"{self} no ha recibido ningún fotograma de {self.url}")
            self.height, self.width = frame.shape[:2]
        self.capture_width, self.capture_height = self.width, self.height

        if self.fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if fps and fps > 0 else 30

    def _postprocess_image(self, image):
        h, w = image.shape[:2]
        if (w, h) != (self.width, self.height):
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return super()._postprocess_image(image)

    def _read_from_hardware(self):
        cap = self.videocapture
        if cap is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")
        ok, frame = cap.read()
        if ok:
            return frame
        self._reabrir_stream()  # corte de red: se intenta reconectar antes de dar error
        raise RuntimeError(f"{self} no recibió fotograma; reconectando...")

    def _reabrir_stream(self) -> None:
        nuevo = None
        try:
            nuevo = cv2.VideoCapture(self.url)
            parar = self.stop_event is None or self.stop_event.is_set()
            if nuevo.isOpened() and not parar:
                viejo, self.videocapture = self.videocapture, nuevo
                nuevo = None
                if viejo is not None:
                    viejo.release()
                return
        except Exception:  # nosec B110
            pass
        finally:
            if nuevo is not None:
                nuevo.release()
        time.sleep(0.5)


# ------------------------------------------------------------
# Lanzador
# ------------------------------------------------------------
COMANDOS = {
    "record": "lerobot.scripts.lerobot_record",
    "teleoperate": "lerobot.scripts.lerobot_teleoperate",
    "rollout": "lerobot.scripts.lerobot_rollout",  # ejecutar un modelo entrenado en el robot
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMANDOS:
        print(f"Uso: python3 {Path(__file__).name} {{{'|'.join(COMANDOS)}}} [opciones de lerobot...]")
        return 2
    comando = sys.argv[1]
    modulo = importlib.import_module(COMANDOS[comando])
    sys.argv = [f"lerobot-{comando}", *sys.argv[2:]]
    return modulo.main() or 0


if __name__ == "__main__":
    sys.exit(main())

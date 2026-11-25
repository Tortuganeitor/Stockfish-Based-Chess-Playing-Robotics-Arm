# Stockfish Based Chess Playing Robotics Arm

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/50c4cbd1-f957-451b-9054-360bd8e05d13" />


Prototipo desarrollado durante la **Hackathon de Bilbao (noviembre de 2026)** que combina visión artificial, un motor de ajedrez y un brazo robótico para jugar partidas sencillas sobre un tablero de ajedrez de **4×4**.

El sistema utiliza un robot **SO-101** controlado mediante **LeLab** desde terminal. A través de distintos scripts, el brazo es capaz de mover piezas de ajedrez representadas por bloques cúbicos con códigos **ARUCO**. Una cámara captura una imagen del tablero y, mediante **OpenCV**, se procesa la imagen para rectificarla usando cuatro puntos de referencia: dos verdes y dos rojos. Gracias a la lectura de los códigos ARUCO, el software identifica qué pieza del ajedrez corresponde a cada bloque.

<img width="800" height="533" alt="image" src="https://github.com/user-attachments/assets/59644251-9708-40cb-b2b5-e48c86029cb5" />

A partir de la posición detectada, el motor de ajedrez de código abierto **Stockfish** calcula la mejor jugada posible. OpenCV muestra esa jugada sobre una representación virtual del tablero. Finalmente, usando grabaciones de un dataset de *pick and place* realizadas con el robot mediante LeLab, el brazo robótico mueve la pieza y la coloca en la casilla indicada por Stockfish.

## Características principales

- Control del robot **SO-101** mediante **LeLab** y scripts de terminal.
- Visión artificial con **OpenCV** para captura, rectificación y análisis del tablero.
- Reconocimiento de piezas mediante **códigos ARUCO** en bloques cúbicos.
- Integración con **Stockfish** para la predicción de la mejor jugada.
- Visualización de la jugada sugerida sobre una imagen virtual del tablero.
- Ejecución de movimientos *pick and place* mediante datos grabados con LeLab.
- Prototipo funcional capaz de realizar jugadas sencillas en un tablero de **4×4**.

## Tecnologías utilizadas

- **SO-101** + **LeLab**
- **OpenCV**
- **ArUco**
- **Stockfish**
- **Python** / scripts de control
- Grabación y reproducción de trayectorias *pick and place*

## Contexto

Proyecto realizado en la **Hackathon de Bilbao, noviembre de 2026**, con el objetivo de explorar la integración de robótica de bajo coste, visión por computador y motores de ajedrez en un sistema capaz de interactuar físicamente con un tablero.

> Estado del proyecto: **prototipo hackathon**. El sistema demuestra el flujo completo de percepción, decisión y actuación robótica para jugar pequeñas jugadas de ajedrez sobre un tablero reducido.

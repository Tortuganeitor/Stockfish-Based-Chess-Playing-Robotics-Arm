# Stockfish Based Chess Playing Robotics Arm

Prototipo desarrollado durante la **Hackathon de Bilbao (noviembre de 2026)** que combina visión artificial, un motor de ajedrez y un brazo robótico para jugar partidas sencillas sobre un tablero de ajedrez de **4×4**.

El sistema utiliza un robot **SO-101** controlado mediante **LeLab** desde terminal. A través de distintos scripts, el brazo es capaz de mover piezas de ajedrez representadas por bloques cúbicos con códigos **ARUCO**. Una cámara captura una imagen del tablero y, mediante **OpenCV**, se procesa la imagen para rectificarla usando cuatro puntos de referencia: dos verdes y dos rojos. Gracias a la lectura de los códigos ARUCO, el software identifica qué pieza del ajedrez corresponde a cada bloque.

A partir de la posición detectada, el motor de ajedrez de código abierto **Stockfish** calcula la mejor jugada posible. OpenCV muestra esa jugada sobre una representación virtual del tablero. Finalmente, usando grabaciones de un dataset de *pick and place* realizadas con el robot mediante LeLab, el brazo robótico mueve la pieza y la coloca en la casilla indicada por Stockfish.

## Características principales

- Control del robot **SO-101** mediante **LeLab** y scripts de terminal.
- Visión artificial con **OpenCV** para captura, rectificación y análisis del tablero.
- Reconocimiento de piezas mediante **códigos ARUCO** en bloques cúbicos.
- Integración con **Stockfish** para la predicción de la mejor jugada.
- Visualización de la jugada sugerida sobre una imagen virtual del tablero.
- Ejecución de movimientos *pick and place* mediante datos grabados con LeLab.
- Prototipo funcional capaz de realizar jugadas sencillas en un tablero de **4×4**.


## Contexto

Proyecto realizado en la **Hackathon de Bilbao, noviembre de 2026**, con el objetivo de explorar la integración de robótica de bajo coste, visión por computador y motores de ajedrez en un sistema capaz de interactuar físicamente con un tablero.

> Estado del proyecto: **prototipo hackathon**. El sistema demuestra el flujo completo de percepción, decisión y actuación robótica para jugar pequeñas jugadas de ajedrez sobre un tablero reducido.

## Funcionamiento del sistema

El flujo del prototipo conecta visión por computador, motor de ajedrez y control robótico en un ciclo de percepción, decisión y ejecución:

    1. Captura y rectificación del tablero (OpenCV):

        Una cámara captura una imagen de la escena.

        Mediante OpenCV, la imagen se transforma y recorta para aislar únicamente el tablero de 4×4 usando cuatro puntos de referencia (dos marcadores verdes y dos rojos).

    2. Detección de piezas con códigos ArUco:

        Cada pieza del ajedrez tiene forma de cubo e incorpora un código ArUco.

        El software de OpenCV lee los códigos ArUco para identificar qué pieza representa cada bloque y su posición en el tablero.

    3. Cálculo de la mejor jugada (Stockfish):

        Con el estado del tablero detectado, el motor de ajedrez de código abierto Stockfish evalúa la posición y predice la mejor jugada posible.

        OpenCV superpone y muestra la jugada recomendada sobre la imagen virtual del tablero.

    4. Ejecución de pick and place (SO-101 + LeLab):

        A partir de grabaciones de datasets de pick and place capturadas previamente con el robot mediante LeLab, el brazo robótico SO-101 (controlado por scripts desde la terminal) recoge la pieza indicada y la coloca en la casilla correspondiente según la jugada sugerida por Stockfish.

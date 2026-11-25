# Stockfish-Based Chess-Playing Robotic Arm

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/50c4cbd1-f957-451b-9054-360bd8e05d13" />

Prototype developed during the **Bilbao Hackathon (November 2026)** that combines computer vision, a chess engine, and a robotic arm to play simple games on a **4×4** chessboard.

The system uses an **SO-101** robot controlled via **LeLab** from the terminal. Through different scripts, the arm can move chess pieces represented by cubic blocks with **ARUCO** codes. A camera captures an image of the board and, using **OpenCV**, the image is processed to rectify it using four reference points: two green and two red. Thanks to ARUCO code reading, the software identifies which chess piece corresponds to each block.

<img width="800" height="533" alt="image" src="https://github.com/user-attachments/assets/59644251-9708-40cb-b2b5-e48c86029cb5" />

Based on the detected position, the open-source chess engine **Stockfish** calculates the best possible move. OpenCV displays that move on a virtual representation of the board. Finally, using recordings from a *pick and place* dataset created with the robot via LeLab, the robotic arm moves the piece and places it on the square indicated by Stockfish.

## Main Features

- Control of the **SO-101** robot via **LeLab** and terminal scripts.
- Computer vision with **OpenCV** for board capture, rectification, and analysis.
- Piece recognition using **ARUCO codes** on cubic blocks.
- Integration with **Stockfish** for best-move prediction.
- Visualization of the suggested move on a virtual board image.
- Execution of *pick and place* movements using data recorded with LeLab.
- Functional prototype capable of making simple moves on a **4×4** board.

## Technologies Used

- **SO-101** + **LeLab**
- **OpenCV**
- **ArUco**
- **Stockfish**
- **Python** / control scripts
- Recording and playback of *pick and place* trajectories

## Context

Project developed at the **Bilbao Hackathon, November 2026**, with the goal of exploring the integration of low-cost robotics, computer vision, and chess engines in a system capable of physically interacting with a board.

> Project status: **hackathon prototype**. The system demonstrates the complete pipeline of perception, decision-making, and robotic actuation to play small chess moves on a reduced board.

# Hugging Face Repositories: 🤗
### Dataset and training data for the model:
  · https://huggingface.co/datasets/Tortuganeitor/SO101_ACT_Merged_Dataset
### Trained model applicable for making the robotic arm move:
  · https://huggingface.co/Tortuganeitor/SO-101_Training_For_Stockfish_Robotics_Arm

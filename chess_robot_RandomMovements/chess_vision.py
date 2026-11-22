import cv2
import numpy as np
import time
import json
import os
import threading
import random

from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# ============================================================
# CONFIGURATION
# ============================================================

SIZE = 640
CELL = SIZE // 4

STABILITY_SECONDS = 0.5
MOVE_LOCK_SECONDS = 10.0
PICKUP_CONFIRM_SECONDS = 0.6
FINAL_BOARD_TIMEOUT = 3.0

# Random moves only to empty squares.
# No captures: easier for the robotic arm.
ALLOW_CAPTURES = False

# Allow demonstration even if kings are missing.
REQUIRE_BOTH_KINGS = False

RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 220, 255)

BLUE_OPACITY = 0.85
GREEN_OPACITY = 0.65
GREEN_PADDING = 15

STREAM_HOST = "0.0.0.0"
STREAM_PORT = 5000
STREAM_FPS = 20
STREAM_WIDTH = 640
STREAM_HEIGHT = 480

_stream_frame = None
_stream_lock = threading.Lock()


# ============================================================
# ARUCO PIECES
# ============================================================

PIECES = {
    1: ("black", "K"),
    3: ("black", "Q"),
    4: ("black", "P"),
    5: ("black", "N"),
    6: ("black", "B"),
    8: ("black", "R"),

    0: ("white", "P"),
    2: ("white", "B"),
    7: ("white", "K"),
    9: ("white", "N"),
    10: ("white", "Q"),
    11: ("white", "R"),
}


# ============================================================
# NETWORK VIDEO STREAM
# ============================================================

def publish_vision_frame(frame):
    global _stream_frame

    square = cv2.resize(frame, (480, 480))

    canvas = np.zeros(
        (STREAM_HEIGHT, STREAM_WIDTH, 3),
        dtype=np.uint8
    )

    canvas[:, 80:560] = square

    ok, encoded = cv2.imencode(
        ".jpg",
        canvas,
        [cv2.IMWRITE_JPEG_QUALITY, 80]
    )

    if ok:
        with _stream_lock:
            _stream_frame = encoded.tobytes()


class VisionStreamHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path not in ("/", "/video"):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            while True:
                with _stream_lock:
                    jpeg = _stream_frame

                if jpeg is not None:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(
                        b"Content-Type: image/jpeg\r\n"
                    )
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()

                time.sleep(1.0 / STREAM_FPS)

        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            OSError
        ):
            pass

    def log_message(self, format, *args):
        pass


def start_vision_server():
    server = ThreadingHTTPServer(
        (STREAM_HOST, STREAM_PORT),
        VisionStreamHandler
    )

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True
    )
    thread.start()

    print("[NETWORK] Vision server started")
    print(f"[NETWORK] Port: {STREAM_PORT}")
    print("[NETWORK] URL: http://YOUR_IP:5000/video")

    return server


# ============================================================
# CAMERA
# ============================================================

def open_camera():
    backends = [
        cv2.CAP_DSHOW,
        cv2.CAP_MSMF,
        cv2.CAP_ANY
    ]

    for index in [1, 0, 2, 3]:
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)

            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            ok, frame = cap.read()

            if ok and frame is not None:
                print("[CAMERA] Connected:", index)
                return cap

            cap.release()

    raise RuntimeError("Camera USB introuvable")


# ============================================================
# COLOR CALIBRATION
# ============================================================

def red_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    m1 = cv2.inRange(
        hsv, (0, 90, 50), (13, 255, 255)
    )
    m2 = cv2.inRange(
        hsv, (170, 90, 50), (179, 255, 255)
    )

    mask = cv2.bitwise_or(m1, m2)
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    return cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )


def green_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv, (35, 60, 45), (90, 255, 255)
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    return cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )


def find_color_centers(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 80:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 8 or h < 8:
            continue

        if not 0.45 <= w / float(h) <= 2.2:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        candidates.append((area, (cx, cy)))

    candidates.sort(
        reverse=True,
        key=lambda item: item[0]
    )

    return [
        point for _, point in candidates[:2]
    ]


def validate_quad(quad, shape):
    if quad is None:
        return False

    quad = np.asarray(quad, dtype=np.float32)

    if quad.shape != (4, 2):
        return False

    h, w = shape[:2]

    if not np.isfinite(quad).all():
        return False

    if np.any(quad[:, 0] < 0):
        return False

    if np.any(quad[:, 0] >= w):
        return False

    if np.any(quad[:, 1] < 0):
        return False

    if np.any(quad[:, 1] >= h):
        return False

    polygon = quad.reshape(-1, 1, 2)

    if not cv2.isContourConvex(polygon):
        return False

    if abs(cv2.contourArea(polygon)) < w * h * 0.08:
        return False

    lengths = [
        np.linalg.norm(
            quad[(i + 1) % 4] - quad[i]
        )
        for i in range(4)
    ]

    if min(lengths) < 40:
        return False

    if (
        max(lengths[0], lengths[2])
        / min(lengths[0], lengths[2])
    ) > 3:
        return False

    if (
        max(lengths[1], lengths[3])
        / min(lengths[1], lengths[3])
    ) > 3:
        return False

    return True


def detect_colored_corners(frame):
    reds = find_color_centers(red_mask(frame))
    greens = find_color_centers(green_mask(frame))

    if len(reds) != 2 or len(greens) != 2:
        return None, reds, greens

    red = np.asarray(reds, dtype=np.float32)
    green = np.asarray(greens, dtype=np.float32)

    direction = (
        green.mean(axis=0)
        - red.mean(axis=0)
    )

    distance = np.linalg.norm(direction)

    if distance < 40:
        return None, reds, greens

    direction /= distance

    right = np.array(
        [direction[1], -direction[0]],
        dtype=np.float32
    )

    red = sorted(
        red,
        key=lambda p: np.dot(p, right)
    )

    green = sorted(
        green,
        key=lambda p: np.dot(p, right)
    )

    quad = np.float32([
        red[0],
        red[1],
        green[1],
        green[0]
    ])

    if not validate_quad(quad, frame.shape):
        return None, reds, greens

    return quad, reds, greens


def homography(quad):
    target = np.float32([
        [0, 0],
        [SIZE - 1, 0],
        [SIZE - 1, SIZE - 1],
        [0, SIZE - 1]
    ])

    return cv2.getPerspectiveTransform(
        np.asarray(quad, dtype=np.float32),
        target
    )


# ============================================================
# ARUCO DETECTION
# ============================================================

aruco = cv2.aruco

dictionary = aruco.getPredefinedDictionary(
    aruco.DICT_4X4_50
)

params = aruco.DetectorParameters()
params.minMarkerPerimeterRate = 0.015

detector = (
    aruco.ArucoDetector(dictionary, params)
    if hasattr(aruco, "ArucoDetector")
    else None
)

clahe = cv2.createCLAHE(2.0, (8, 8))


def detect_aruco(frame):
    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    variants = [
        (gray, 1.0),
        (clahe.apply(gray), 1.0),
        (
            cv2.resize(
                gray,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC
            ),
            2.0
        )
    ]

    result = {}

    for image, scale in variants:
        if detector is not None:
            corners, ids, _ = detector.detectMarkers(image)
        else:
            corners, ids, _ = aruco.detectMarkers(
                image,
                dictionary,
                parameters=params
            )

        if ids is None:
            continue

        for ident, pts in zip(
            ids.flatten(),
            corners
        ):
            ident = int(ident)

            pts = np.asarray(
                pts,
                dtype=np.float32
            ).reshape(4, 2) / scale

            area = cv2.contourArea(pts)

            if (
                ident not in result
                or area > result[ident][0]
            ):
                result[ident] = (area, pts)

    return [
        (ident, data[1])
        for ident, data in result.items()
    ]


def get_square(pts):
    x, y = pts.mean(axis=0)

    if not (
        0 <= x < SIZE
        and 0 <= y < SIZE
    ):
        return None

    return (
        int(y // CELL),
        int(x // CELL)
    )


def square_name(pos):
    row, col = pos
    return "ABCD"[col] + str(4 - row)


# ============================================================
# CHESS MOVEMENT RULES
# NO STOCKFISH - NO MINIMAX
# ============================================================

def enemy_of(side):
    return (
        "white"
        if side == "black"
        else "black"
    )


def find_king(board, side):
    for pos, piece in board.items():
        if (
            piece[0] == side
            and piece[1] == "K"
        ):
            return pos

    return None


def validate_board(board):
    if not board:
        return False

    if not REQUIRE_BOTH_KINGS:
        return True

    return (
        find_king(board, "white") is not None
        and find_king(board, "black") is not None
    )


def pseudo_moves(
    pos,
    board,
    attacks_only=False
):
    if pos not in board:
        return []

    row, col = pos
    side, kind, _ = board[pos]

    result = []

    def add(r, c):
        if not (
            0 <= r < 4
            and 0 <= c < 4
        ):
            return False

        target = board.get((r, c))

        if attacks_only:
            result.append((r, c))
            return target is None

        if target is None:
            result.append((r, c))
            return True

        if (
            ALLOW_CAPTURES
            and target[0] != side
            and target[1] != "K"
        ):
            result.append((r, c))

        return False

    if kind == "P":
        direction = (
            1 if side == "black" else -1
        )

        nr = row + direction

        if (
            not attacks_only
            and 0 <= nr < 4
            and (nr, col) not in board
        ):
            result.append((nr, col))

        for dc in (-1, 1):
            nc = col + dc

            if (
                0 <= nr < 4
                and 0 <= nc < 4
            ):
                target = board.get((nr, nc))

                if attacks_only:
                    result.append((nr, nc))

                elif (
                    ALLOW_CAPTURES
                    and target is not None
                    and target[0] != side
                    and target[1] != "K"
                ):
                    result.append((nr, nc))

    elif kind == "N":
        for dr, dc in [
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1)
        ]:
            add(row + dr, col + dc)

    elif kind == "K":
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr != 0 or dc != 0:
                    add(row + dr, col + dc)

    else:
        directions = []

        if kind in ("R", "Q"):
            directions += [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1)
            ]

        if kind in ("B", "Q"):
            directions += [
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1)
            ]

        for dr, dc in directions:
            r = row + dr
            c = col + dc

            while (
                0 <= r < 4
                and 0 <= c < 4
            ):
                if not add(r, c):
                    break

                r += dr
                c += dc

    return result


def is_attacked(square, board, by_side):
    if square is None:
        return False

    for pos, piece in board.items():
        if piece[0] != by_side:
            continue

        if square in pseudo_moves(
            pos,
            board,
            attacks_only=True
        ):
            return True

    return False


def apply_move(board, start, end):
    future = board.copy()
    piece = future.pop(start)

    side, kind, ident = piece

    if kind == "P":
        if (
            (side == "black" and end[0] == 3)
            or
            (side == "white" and end[0] == 0)
        ):
            piece = (side, "Q", ident)

    future[end] = piece
    return future


def legal_moves(pos, board):
    if pos not in board:
        return []

    side = board[pos][0]
    king_before = find_king(board, side)

    result = []

    for end in pseudo_moves(pos, board):
        future = apply_move(
            board,
            pos,
            end
        )

        king_after = find_king(
            future,
            side
        )

        if king_before is None:
            result.append(end)
            continue

        if (
            king_after is not None
            and not is_attacked(
                king_after,
                future,
                enemy_of(side)
            )
        ):
            result.append(end)

    return result


def all_moves(board, side):
    return [
        (pos, end)
        for pos, piece in board.items()
        if piece[0] == side
        for end in legal_moves(pos, board)
    ]


# ============================================================
# RANDOM BLACK MOVE GENERATOR
# ============================================================

def random_black_move(board):
    if not validate_board(board):
        print("[RANDOM] Board not ready")
        return None

    available = all_moves(board, "black")

    if not available:
        print("[RANDOM] No accessible moves")
        return None

    # Select ONE random move.
    # No evaluation, no minimax, no Stockfish.
    move = random.choice(available)

    print(
        "[RANDOM MOVE]",
        square_name(move[0]),
        "->",
        square_name(move[1])
    )

    return move


def analyze_board(board):
    if not validate_board(board):
        return {
            "status": "INVALID_BOARD",
            "move": None,
            "uci": None,
            "score_cp": None,
            "score_mate": None
        }

    candidate = random_black_move(board)

    status = (
        "PLAYING"
        if candidate is not None
        else "NO_MOVES"
    )

    uci = None

    if candidate is not None:
        uci = (
            square_name(candidate[0])
            + square_name(candidate[1])
        ).lower()

    return {
        "status": status,
        "move": candidate,
        "uci": uci,
        "score_cp": None,
        "score_mate": None
    }


# ============================================================
# DRAW GRID
# ============================================================

def draw_grid(frame):
    for i in range(5):
        p = min(i * CELL, SIZE - 1)

        cv2.line(
            frame,
            (p, 0),
            (p, SIZE - 1),
            YELLOW,
            2
        )

        cv2.line(
            frame,
            (0, p),
            (SIZE - 1, p),
            YELLOW,
            2
        )

    for r in range(4):
        for c in range(4):
            cv2.putText(
                frame,
                f"{r * 4 + c + 1} {square_name((r, c))}",
                (c * CELL + 8, r * CELL + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                YELLOW,
                2
            )


# ============================================================
# BLUE PLACE SQUARE ONLY
# ============================================================

def draw_move_zones(frame, best):
    if best is None:
        return

    _, end = best
    row, col = end

    x1 = col * CELL
    y1 = row * CELL
    x2 = (col + 1) * CELL - 1
    y2 = (row + 1) * CELL - 1

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        BLUE,
        -1
    )

    cv2.addWeighted(
        overlay,
        BLUE_OPACITY,
        frame,
        1.0 - BLUE_OPACITY,
        0,
        frame
    )

    cv2.rectangle(
        frame,
        (x1 + 3, y1 + 3),
        (x2 - 3, y2 - 3),
        BLUE,
        5
    )

    cv2.putText(
        frame,
        "PLACE",
        (x1 + 12, y2 - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


# ============================================================
# GREEN SELECTED ARUCO PIECE
# ============================================================

def draw_selected_piece(
    frame,
    detections,
    board,
    best,
    selected_piece_id=None
):
    moving_id = selected_piece_id

    if (
        moving_id is None
        and best is not None
        and best[0] in board
    ):
        moving_id = board[best[0]][2]

    for ident, pts in detections:
        pts_int = pts.astype(np.int32)

        if ident == moving_id:
            x, y, w, h = cv2.boundingRect(
                pts_int
            )

            x1 = max(0, x - GREEN_PADDING)
            y1 = max(0, y - GREEN_PADDING)
            x2 = min(SIZE - 1, x + w + GREEN_PADDING)
            y2 = min(SIZE - 1, y + h + GREEN_PADDING)

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                GREEN,
                -1
            )

            cv2.addWeighted(
                overlay,
                GREEN_OPACITY,
                frame,
                1 - GREEN_OPACITY,
                0,
                frame
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                GREEN,
                4
            )

        else:
            cv2.polylines(
                frame,
                [pts_int],
                True,
                GREEN,
                2
            )

        cx, cy = map(
            int,
            pts.mean(axis=0)
        )

        cv2.putText(
            frame,
            f"ID {ident}",
            (max(5, cx - 35), max(18, cy - 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2
        )


# ============================================================
# DIGITAL TWIN
# ONLY ONE BLUE ARROW
# ============================================================

def draw_twin(
    board,
    selected,
    best,
    status="PLAYING"
):
    twin = np.full(
        (SIZE + 110, SIZE, 3),
        245,
        dtype=np.uint8
    )

    def center(pos):
        r, c = pos
        return (
            c * CELL + CELL // 2,
            r * CELL + CELL // 2
        )

    # Draw board
    for r in range(4):
        for c in range(4):
            color = (
                (235, 235, 235)
                if (r + c) % 2 == 0
                else (175, 195, 215)
            )

            cv2.rectangle(
                twin,
                (c * CELL, r * CELL),
                ((c + 1) * CELL, (r + 1) * CELL),
                color,
                -1
            )

            cv2.putText(
                twin,
                f"{r * 4 + c + 1} {square_name((r, c))}",
                (c * CELL + 8, r * CELL + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (55, 55, 55),
                2
            )

    # Draw pieces
    for pos, piece in board.items():
        side, kind, ident = piece
        x, y = center(pos)

        fill = (
            (30, 30, 30)
            if side == "black"
            else (255, 255, 255)
        )

        fg = (
            (255, 255, 255)
            if side == "black"
            else (25, 25, 25)
        )

        cv2.circle(
            twin,
            (x, y),
            42,
            fill,
            -1
        )

        cv2.circle(
            twin,
            (x, y),
            42,
            (100, 100, 100),
            2
        )

        cv2.putText(
            twin,
            kind,
            (x - 18, y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            fg,
            3
        )

        cv2.putText(
            twin,
            f"ID {ident}",
            (x - 22, y + 61),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (60, 60, 60),
            1
        )

    # ONLY the chosen random move.
    # No arrows showing other legal moves.
    if best is not None:
        start, end = best

        p1 = center(start)
        p2 = center(end)

        # Black outline for visibility
        cv2.arrowedLine(
            twin,
            p1,
            p2,
            (0, 0, 0),
            13,
            tipLength=0.28
        )

        # One BLUE arrow
        cv2.arrowedLine(
            twin,
            p1,
            p2,
            BLUE,
            8,
            tipLength=0.28
        )

        # Blue destination border
        r, c = end

        cv2.rectangle(
            twin,
            (c * CELL + 4, r * CELL + 4),
            (
                (c + 1) * CELL - 4,
                (r + 1) * CELL - 4
            ),
            BLUE,
            5
        )

    # Information panel
    cv2.rectangle(
        twin,
        (0, SIZE),
        (SIZE, SIZE + 110),
        (30, 30, 30),
        -1
    )

    if best is not None:
        message = (
            f"RANDOM MOVE: "
            f"{square_name(best[0])} -> "
            f"{square_name(best[1])}"
        )

    elif status == "INVALID_BOARD":
        message = "BOARD NOT READY"

    elif status == "NO_MOVES":
        message = "NO ACCESSIBLE MOVES"

    else:
        message = "WAITING FOR RANDOM MOVE"

    cv2.putText(
        twin,
        message,
        (12, SIZE + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 220, 100),
        2
    )

    cv2.putText(
        twin,
        "MODE: RANDOM - BLACK PIECES ONLY",
        (12, SIZE + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        twin,
        "BLUE: RANDOM MOVE | GREEN: SELECTED PIECE",
        (12, SIZE + 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    return twin


# ============================================================
# EXPORT MOVE.JSON
# ============================================================

_last_exported_move = None


def export_best_move(
    board,
    best,
    filename="move.json"
):
    global _last_exported_move

    if best is None:
        return

    start, end = best

    if start not in board:
        return

    move_key = (
        tuple(
            sorted(
                (pos, piece[0], piece[1], piece[2])
                for pos, piece in board.items()
            )
        ),
        start,
        end
    )

    if move_key == _last_exported_move:
        return

    piece = board[start]
    captured = board.get(end)

    def image_center(pos):
        row, col = pos

        return {
            "x": int(col * CELL + CELL / 2),
            "y": int(row * CELL + CELL / 2)
        }

    data = {
        "status": "proposed",
        "from": square_name(start),
        "to": square_name(end),
        "piece": piece[1],
        "piece_id": piece[2],
        "capture": captured is not None,
        "captured_piece_id": (
            captured[2] if captured else None
        ),
        "from_image": image_center(start),
        "to_image": image_center(end),
        "coordinate_system": "warped_board_pixels",
        "board_size": [4, 4],
        "image_size": [SIZE, SIZE]
    }

    temporary = filename + ".tmp"

    with open(
        temporary,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(data, f, indent=4)

    os.replace(temporary, filename)

    _last_exported_move = move_key

    print(
        "[ROBOT EXPORT]",
        data["from"],
        "->",
        data["to"]
    )


# ============================================================
# MOVE LOCK CONTROLLER
# ============================================================

class MoveLockController:

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "IDLE"
        self.move = None
        self.piece_id = None
        self.piece_kind = None
        self.missing_since = None
        self.hold_until = None
        self.final_wait_since = None

    def propose(self, board, move, now):
        if move is None:
            return False

        start, end = move

        if start not in board:
            return False

        self.move = move
        self.piece_id = board[start][2]
        self.piece_kind = board[start][1]

        self.state = "WAIT_PICKUP"
        self.missing_since = None
        self.hold_until = None
        self.final_wait_since = None

        print(
            "[MOVE LOCK] Proposed:",
            square_name(start),
            "->",
            square_name(end)
        )

        print("[MOVE LOCK] Waiting for pickup")
        return True

    def update(self, detections, now):
        if self.move is None:
            return

        if self.state == "WAIT_PICKUP":
            start = self.move[0]
            at_start = False

            for ident, pts in detections:
                if ident != self.piece_id:
                    continue

                if get_square(pts) == start:
                    at_start = True
                    break

            if at_start:
                self.missing_since = None

            else:
                if self.missing_since is None:
                    self.missing_since = now

                elif (
                    now - self.missing_since
                    >= PICKUP_CONFIRM_SECONDS
                ):
                    self.state = "HOLD"
                    self.hold_until = (
                        now + MOVE_LOCK_SECONDS
                    )

                    print("[MOVE LOCK] PICKUP DETECTED")
                    print("[MOVE LOCK] Holding 15 seconds")

        elif self.state == "HOLD":
            if now >= self.hold_until:
                self.state = "WAIT_STABLE"
                self.final_wait_since = now

                print("[MOVE LOCK] Hold finished")
                print("[MOVE LOCK] Checking final board")

    def complete_if_stable(
        self,
        board,
        position_key,
        stable,
        now=None
    ):
        if self.state != "WAIT_STABLE":
            return False

        if not stable:
            return False

        found_at_destination = False

        if self.move is not None:
            destination = self.move[1]
            piece = board.get(destination)

            found_at_destination = (
                piece is not None
                and piece[2] == self.piece_id
            )

        if found_at_destination:
            print("[MOVE LOCK] Move confirmed")

        else:
            if (
                now is None
                or self.final_wait_since is None
                or now - self.final_wait_since
                < FINAL_BOARD_TIMEOUT
            ):
                return False

            print(
                "[MOVE LOCK] Timeout: "
                "final move not confirmed"
            )

        self.reset()
        print("[MOVE LOCK] Ready")
        return True

    @property
    def display_move(self):
        if self.state in (
            "WAIT_PICKUP",
            "HOLD",
            "WAIT_STABLE"
        ):
            return self.move

        return None

    @property
    def busy(self):
        return self.state != "IDLE"

    @property
    def moving_piece_id(self):
        return self.piece_id

    def remaining(self, now):
        if (
            self.state == "HOLD"
            and self.hold_until is not None
        ):
            return max(
                0.0,
                self.hold_until - now
            )

        return 0.0


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    cap = open_camera()
    stream_server = None

    game_status = "PLAYING"

    state = {
        "board": {},
        "selected_id": None,
        "mode": "auto",
        "clicks": []
    }

    locked_quad = None
    histories = {}
    seen = {}

    best = None
    last_position = None
    pending_position = None
    position_since = time.monotonic()

    move_lock = MoveLockController()

    def click_twin(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not (
            0 <= x < SIZE
            and 0 <= y < SIZE
        ):
            return

        pos = (
            y // CELL,
            x // CELL
        )

        piece = state["board"].get(pos)

        if piece and piece[0] == "black":
            state["selected_id"] = piece[2]
        else:
            state["selected_id"] = None

    def click_raw(event, x, y, flags, param):
        if (
            event == cv2.EVENT_LBUTTONDOWN
            and state["mode"] == "manual"
            and len(state["clicks"]) < 4
        ):
            state["clicks"].append((x, y))

    try:
        stream_server = start_vision_server()

        cv2.namedWindow(
            "Camera RAW",
            cv2.WINDOW_AUTOSIZE
        )
        cv2.namedWindow(
            "Chess Camera",
            cv2.WINDOW_AUTOSIZE
        )
        cv2.namedWindow(
            "Chess Coach 4x4",
            cv2.WINDOW_AUTOSIZE
        )

        cv2.setMouseCallback(
            "Chess Coach 4x4",
            click_twin
        )
        cv2.setMouseCallback(
            "Camera RAW",
            click_raw
        )

        print("================================")
        print("CHESS ROBOT 4x4")
        print("RANDOM BLACK MOVES")
        print("NO STOCKFISH - NO MINIMAX")
        print("ONE BLUE ARROW ONLY")
        print("BLUE PLACE + GREEN PIECE")
        print("15-SECOND MOVE LOCK")
        print("================================")
        print("A: Auto calibration")
        print("C: Manual calibration")
        print("X: Reset calibration")
        print("R: Reset selection")
        print("Q: Quit")
        print("================================")

        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                print("[ERROR] Camera unavailable")
                break

            now = time.monotonic()
            raw_preview = frame.copy()

            # ========================================
            # CALIBRATION
            # ========================================

            if locked_quad is None:
                if state["mode"] == "auto":
                    quad, reds, greens = (
                        detect_colored_corners(frame)
                    )

                    for x, y in reds:
                        cv2.circle(
                            raw_preview,
                            (int(x), int(y)),
                            10,
                            RED,
                            3
                        )

                    for x, y in greens:
                        cv2.circle(
                            raw_preview,
                            (int(x), int(y)),
                            10,
                            GREEN,
                            3
                        )

                    if quad is not None:
                        locked_quad = quad
                        print("[CALIBRATION] AUTO OK")

                elif len(state["clicks"]) == 4:
                    candidate = np.float32(
                        state["clicks"]
                    )

                    if validate_quad(
                        candidate,
                        frame.shape
                    ):
                        locked_quad = candidate
                        print("[CALIBRATION] MANUAL OK")

                    state["clicks"] = []

            if locked_quad is None:
                message = (
                    "AUTO: 2 RED + 2 GREEN CUBES"
                    if state["mode"] == "auto"
                    else (
                        f"MANUAL: "
                        f"{len(state['clicks'])}/4"
                    )
                )

                cv2.putText(
                    raw_preview,
                    message,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    YELLOW,
                    2
                )

                for x, y in state["clicks"]:
                    cv2.circle(
                        raw_preview,
                        (int(x), int(y)),
                        7,
                        YELLOW,
                        -1
                    )

                cv2.imshow(
                    "Camera RAW",
                    raw_preview
                )

                cv2.imshow(
                    "Chess Coach 4x4",
                    draw_twin({}, None, None)
                )

                waiting = np.zeros(
                    (SIZE, SIZE, 3),
                    dtype=np.uint8
                )

                cv2.putText(
                    waiting,
                    "WAITING FOR CALIBRATION",
                    (75, SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    YELLOW,
                    2
                )

                publish_vision_frame(waiting)

                key = cv2.waitKey(1) & 255

                if key in (ord("q"), 27):
                    break

                if key == ord("a"):
                    state["mode"] = "auto"
                    state["clicks"] = []

                if key == ord("c"):
                    state["mode"] = "manual"
                    state["clicks"] = []

                if key == ord("x"):
                    state["clicks"] = []

                continue

            # ========================================
            # RAW CAMERA
            # ========================================

            quad_int = locked_quad.astype(
                np.int32
            )

            cv2.polylines(
                raw_preview,
                [quad_int],
                True,
                GREEN,
                2
            )

            cv2.imshow(
                "Camera RAW",
                raw_preview
            )

            # ========================================
            # PERSPECTIVE TRANSFORMATION
            # ========================================

            H = homography(locked_quad)

            cropped = cv2.warpPerspective(
                frame,
                H,
                (SIZE, SIZE)
            )

            # ========================================
            # ARUCO DETECTION
            # ========================================

            detections = detect_aruco(cropped)

            for ident, pts in detections:
                if ident not in PIECES:
                    continue

                pos = get_square(pts)

                if pos is None:
                    continue

                if ident not in histories:
                    histories[ident] = deque(maxlen=5)

                histories[ident].append(pos)
                seen[ident] = now

            board = {}

            for ident, history in list(
                histories.items()
            ):
                if (
                    now - seen.get(ident, 0)
                    > 0.8
                ):
                    histories.pop(ident, None)
                    seen.pop(ident, None)
                    continue

                pos = Counter(
                    history
                ).most_common(1)[0][0]

                if pos not in board:
                    board[pos] = (
                        *PIECES[ident],
                        ident
                    )

            # ========================================
            # POSITION STABILITY
            # ========================================

            position_key = tuple(
                sorted(
                    (pos, *piece)
                    for pos, piece in board.items()
                )
            )

            if position_key != pending_position:
                pending_position = position_key
                position_since = now

            position_stable = (
                now - position_since
                >= STABILITY_SECONDS
            )

            # ========================================
            # MOVE LOCK
            # ========================================

            move_lock.update(
                detections,
                now
            )

            if move_lock.state == "WAIT_STABLE":
                if move_lock.complete_if_stable(
                    board,
                    position_key,
                    position_stable,
                    now
                ):
                    last_position = position_key
                    best = None

            # ========================================
            # RANDOM MOVE ANALYSIS
            # ========================================

            if (
                not move_lock.busy
                and position_stable
                and position_key != last_position
            ):
                result = analyze_board(board)

                game_status = result["status"]
                candidate = result.get("move")

                print("[RANDOM SYSTEM]", game_status)
                print("[RANDOM MOVE]", result.get("uci"))

                last_position = position_key

                if candidate is not None:
                    if move_lock.propose(
                        board,
                        candidate,
                        now
                    ):
                        best = candidate

                        export_best_move(
                            board,
                            best
                        )

            # ========================================
            # DRAW CAMERA
            # ========================================

            display_best = move_lock.display_move

            draw_move_zones(
                cropped,
                display_best
            )

            draw_grid(cropped)

            draw_selected_piece(
                cropped,
                detections,
                board,
                display_best,
                selected_piece_id=(
                    move_lock.moving_piece_id
                    if display_best is not None
                    else None
                )
            )

            cv2.rectangle(
                cropped,
                (1, 1),
                (SIZE - 2, SIZE - 2),
                GREEN,
                3
            )

            # ========================================
            # STATUS
            # ========================================

            if move_lock.state == "WAIT_PICKUP":
                status_text = "WAITING FOR PICKUP"

            elif move_lock.state == "HOLD":
                remaining = move_lock.remaining(now)
                status_text = (
                    f"MOVE LOCK: {remaining:.1f}s"
                )

            elif move_lock.state == "WAIT_STABLE":
                status_text = "FINALIZING MOVE"

            elif game_status == "INVALID_BOARD":
                status_text = "BOARD NOT READY"

            elif game_status == "NO_MOVES":
                status_text = "NO ACCESSIBLE MOVES"

            else:
                status_text = "RANDOM READY"

            cv2.rectangle(
                cropped,
                (5, SIZE - 40),
                (SIZE - 5, SIZE - 5),
                (25, 25, 25),
                -1
            )

            cv2.putText(
                cropped,
                status_text,
                (15, SIZE - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            publish_vision_frame(cropped)

            # ========================================
            # DIGITAL TWIN
            # ========================================

            state["board"] = board

            selected = None

            for pos, piece in board.items():
                if piece[2] == state["selected_id"]:
                    selected = pos
                    break

            cv2.imshow(
                "Chess Camera",
                cropped
            )

            cv2.imshow(
                "Chess Coach 4x4",
                draw_twin(
                    board,
                    selected,
                    display_best,
                    game_status
                )
            )

            # ========================================
            # KEYBOARD
            # ========================================

            key = cv2.waitKey(1) & 255

            if key in (ord("q"), 27):
                break

            if key == ord("r"):
                state["selected_id"] = None

            if key in (
                ord("x"),
                ord("c"),
                ord("a")
            ):
                locked_quad = None

                histories.clear()
                seen.clear()

                state["board"] = {}
                state["selected_id"] = None
                state["clicks"] = []

                best = None
                last_position = None
                pending_position = None

                position_since = time.monotonic()

                move_lock.reset()
                game_status = "PLAYING"

                state["mode"] = (
                    "manual"
                    if key == ord("c")
                    else "auto"
                )

                print("[CALIBRATION] RESET")
                print("[MOVE LOCK] RESET")

    finally:
        if stream_server is not None:
            stream_server.shutdown()
            stream_server.server_close()

        cap.release()
        cv2.destroyAllWindows()

        print("[SYSTEM] Closed successfully")


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
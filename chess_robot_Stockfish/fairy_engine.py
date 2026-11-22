import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ENGINE_PATH = Path("/usr/games/stockfish")
VARIANT_PATH = BASE_DIR / "variants.ini"

SEARCH_DEPTH = 18
SEARCH_TIME_MS = 500


class FairyChessEngine:

    def __init__(self):

        if not ENGINE_PATH.is_file():
            raise FileNotFoundError(
                f"Engine not found: {ENGINE_PATH}"
            )

        if not VARIANT_PATH.is_file():
            raise FileNotFoundError(
                f"Variants file not found: {VARIANT_PATH}"
            )

        self.process = subprocess.Popen(
            [str(ENGINE_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE_DIR)
        )

        self.send("uci")
        self.wait_for("uciok")

        self.send(
            f"setoption name VariantPath value {VARIANT_PATH}"
        )

        self.send(
            "setoption name UCI_Variant value chess4x4"
        )

        self.send(
            "setoption name Use NNUE value false"
        )

        self.send(
            "setoption name UCI_LimitStrength value false"
        )

        self.send("setoption name Threads value 4")
        self.send("setoption name Hash value 128")

        self.send("isready")
        self.wait_for("readyok")

        print("[ENGINE] Engine ready", flush=True)
        print("[ENGINE] Chess 4x4 requested", flush=True)

    def send(self, command):

        if self.process.poll() is not None:
            raise RuntimeError("Engine stopped")

        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read(self):

        line = self.process.stdout.readline()

        if not line:
            raise RuntimeError("Engine connection lost")

        return line.strip()

    def wait_for(self, keyword):

        while True:
            line = self.read()

            if line == keyword:
                return

    def board_to_fen(self, board, side="black"):

        rows = []

        for r in range(4):

            row = ""
            empty = 0

            for c in range(4):

                piece = board.get((r, c))

                if piece is None:
                    empty += 1
                    continue

                if empty:
                    row += str(empty)
                    empty = 0

                color, kind, ident = piece

                if color == "white":
                    row += kind.upper()
                else:
                    row += kind.lower()

            if empty:
                row += str(empty)

            rows.append(row)

        turn = "b" if side == "black" else "w"

        fen = "/".join(rows)

        return f"{fen} {turn} - - 0 1"

    def validate_board(self, board):

        white_kings = 0
        black_kings = 0

        for color, kind, ident in board.values():

            if kind == "K":

                if color == "white":
                    white_kings += 1
                else:
                    black_kings += 1

        return (
            white_kings == 1
            and black_kings == 1
        )

    def set_position(self, fen):

        self.send(f"position fen {fen}")

    def is_check(self):

        self.send("d")
        self.send("isready")

        in_check = None

        while True:

            line = self.read()

            if line.startswith("Checkers:"):
                checkers = line.split(":", 1)[1].strip()
                in_check = bool(checkers)

            if line == "readyok":
                break

        return in_check

    def analyze_fen(self, fen):

        print("\n==============================")
        print("[AI] ANALYZING POSITION")
        print("[AI] FEN:", fen, flush=True)

        self.set_position(fen)

        check = self.is_check()

        self.send(
            f"go depth {SEARCH_DEPTH} "
            f"movetime {SEARCH_TIME_MS}"
        )

        print("[AI] Search started", flush=True)

        uci_move = None
        score_cp = None
        score_mate = None

        while True:

            line = self.read()
            parts = line.split()

            if "score" in parts:

                i = parts.index("score")

                if i + 2 < len(parts):

                    score_type = parts[i + 1]

                    try:
                        value = int(parts[i + 2])

                        if score_type == "cp":
                            score_cp = value
                            score_mate = None

                        elif score_type == "mate":
                            score_mate = value
                            score_cp = None

                    except ValueError:
                        pass

            if line.startswith("bestmove"):

                if len(parts) >= 2:
                    uci_move = parts[1]

                break

        print(
            "[AI] RAW BEST MOVE:",
            repr(uci_move),
            flush=True
        )

        if uci_move in (None, "(none)", "0000"):

            if check is True:
                status = "CHECKMATE"
            elif check is False:
                status = "STALEMATE"
            else:
                status = "NO_LEGAL_MOVE"

            print("[AI] STATUS:", status)

            return {
                "status": status,
                "move": None,
                "uci": uci_move,
                "promotion": None,
                "score_cp": score_cp,
                "score_mate": score_mate
            }

        if (
            not isinstance(uci_move, str)
            or len(uci_move) < 4
            or uci_move[0] not in "abcd"
            or uci_move[1] not in "1234"
            or uci_move[2] not in "abcd"
            or uci_move[3] not in "1234"
        ):

            print(
                "[AI ERROR] INVALID BEST MOVE:",
                repr(uci_move),
                flush=True
            )

            return {
                "status": "INVALID_MOVE",
                "move": None,
                "uci": uci_move,
                "promotion": None,
                "score_cp": score_cp,
                "score_mate": score_mate
            }

        start = self.square_to_position(
            uci_move[:2]
        )

        end = self.square_to_position(
            uci_move[2:4]
        )

        if start is None or end is None:

            print("[AI ERROR] INVALID COORDINATES")

            return {
                "status": "INVALID_MOVE",
                "move": None,
                "uci": uci_move,
                "promotion": None,
                "score_cp": score_cp,
                "score_mate": score_mate
            }

        promotion = (
            uci_move[4].upper()
            if len(uci_move) > 4
            else None
        )

        print("[AI] START:", start)
        print("[AI] END:", end)
        print("[AI] MOVE:", uci_move)
        print("[AI] SCORE CP:", score_cp)
        print("[AI] SCORE MATE:", score_mate)

        status = "CHECK" if check else "PLAYING"

        return {
            "status": status,
            "move": (start, end),
            "uci": uci_move,
            "promotion": promotion,
            "score_cp": score_cp,
            "score_mate": score_mate
        }

    def analyze_board(self, board, side="black"):

        if not self.validate_board(board):

            print(
                "[AI ERROR] INVALID BOARD",
                flush=True
            )

            return {
                "status": "INVALID_BOARD",
                "move": None,
                "uci": None,
                "promotion": None,
                "score_cp": None,
                "score_mate": None
            }

        fen = self.board_to_fen(board, side)

        return self.analyze_fen(fen)

    @staticmethod
    def square_to_position(square):

        if not isinstance(square, str):
            return None

        if len(square) != 2:
            return None

        if square[0] not in "abcd":
            return None

        if square[1] not in "1234":
            return None

        col = "abcd".index(square[0])
        row = 4 - int(square[1])

        return row, col

    def close(self):

        if self.process.poll() is None:

            self.send("quit")

            try:
                self.process.wait(timeout=3)

            except subprocess.TimeoutExpired:
                self.process.kill()

        print("[ENGINE] Closed")


if __name__ == "__main__":

    engine = FairyChessEngine()

    tests = {
        "NORMAL GAME": "k3/4/4/2RK b - - 0 1",
        "CHECKMATE": "k3/1Q2/2K1/4 b - - 0 1",
        "STALEMATE": "k3/2Q1/2K1/4 b - - 0 1",
        "CHECK": "k3/4/R3/3K b - - 0 1"
    }

    try:

        for name, fen in tests.items():

            print("\n" + "=" * 40)
            print("TEST:", name)
            print("FEN:", fen)

            result = engine.analyze_fen(fen)

            for key, value in result.items():
                print(f"{key}: {value}")

    finally:
        engine.close()

from functools import lru_cache

SIZE = 4
WIN = 10000

def name(pos):
    r, c = pos
    return "ABCD"[c] + str(4 - r)

def knight_moves(pos):
    r, c = pos
    offsets = [
        (-2, -1), (-2, 1),
        (-1, -2), (-1, 2),
        (1, -2), (1, 2),
        (2, -1), (2, 1)
    ]
    return [
        (r + dr, c + dc)
        for dr, dc in offsets
        if 0 <= r + dr < SIZE
        and 0 <= c + dc < SIZE
    ]

def rook_moves(pos):
    r, c = pos
    return [
        (rr, c) for rr in range(SIZE) if rr != r
    ] + [
        (r, cc) for cc in range(SIZE) if cc != c
    ]

def evaluate(knight, rook):
    # Evaluation from the knight's perspective
    if knight == rook:
        return 0

    knight_threat = rook in knight_moves(knight)
    rook_threat = knight in rook_moves(rook)

    score = 0
    if knight_threat:
        score += 150
    if rook_threat:
        score -= 180

    # Prefer central squares
    r, c = knight
    score -= 5 * (abs(r - 1.5) + abs(c - 1.5))

    return score

@lru_cache(maxsize=100000)
def search(knight, rook, turn, depth, alpha, beta):
    if depth == 0:
        return evaluate(knight, rook)

    if turn == "knight":
        best = -float("inf")

        for move in knight_moves(knight):
            if move == rook:
                return WIN + depth

            value = search(
                move, rook, "rook",
                depth - 1, alpha, beta
            )
            best = max(best, value)
            alpha = max(alpha, best)

            if alpha >= beta:
                break

        return best if best != -float("inf") else -WIN

    else:
        best = float("inf")

        for move in rook_moves(rook):
            if move == knight:
                return -WIN - depth

            value = search(
                knight, move, "knight",
                depth - 1, alpha, beta
            )
            best = min(best, value)
            beta = min(beta, best)

            if alpha >= beta:
                break

        return best if best != float("inf") else WIN

def best_move(knight, rook, max_depth=6):
    chosen = None
    score = -float("inf")

    # Iterative deepening
    for depth in range(1, max_depth + 1):
        current_best = None
        current_score = -float("inf")

        for move in knight_moves(knight):
            if move == rook:
                value = WIN + depth
            else:
                value = search(
                    move, rook, "rook",
                    depth - 1,
                    -float("inf"),
                    float("inf")
                )

            if value > current_score:
                current_score = value
                current_best = move

        chosen = current_best
        score = current_score

    return chosen, score

if __name__ == "__main__":
    knight = (2, 2)  # C2
    rook = (0, 0)    # A4

    move, score = best_move(knight, rook)

    print("DARK KNIGHT:", name(knight))
    print("WHITE ROOK:", name(rook))
    print("BEST MOVE:", name(move) if move else "NONE")
    print("EVALUATION:", score)

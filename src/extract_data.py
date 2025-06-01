import chess.pgn
import os
from tqdm import tqdm
import numpy as np

def board_to_matrix(board):
    # 8x8x12 binary planes (6 piece types x 2 colors)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    board_matrix = np.zeros((8, 8, 12), dtype=np.uint8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - square // 8
            col = square % 8
            idx = piece_map[piece.symbol()]
            board_matrix[row, col, idx] = 1
    return board_matrix


def extract_positions_and_labels(pgn_path, max_games=1000, min_avg_elo=0):
    positions = []
    labels = []

    with open(pgn_path) as f:
        for _ in tqdm(range(max_games), desc="Processing games"):
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Get player Elos and compute average
            white_elo = game.headers.get("WhiteElo")
            black_elo = game.headers.get("BlackElo")
            try:
                avg_elo = (int(white_elo) + int(black_elo)) / 2
            except (TypeError, ValueError):
                continue  # Skip games with missing or invalid Elo

            if avg_elo < min_avg_elo:
                continue

            result = game.headers.get("Result")
            if result == "1-0":
                label = 1
            elif result == "0-1":
                label = 0
            elif result == "1/2-1/2":
                label = 0.5
            else:
                continue

            board = game.board()
            for move in game.mainline_moves():
                positions.append(board_to_matrix(board))
                labels.append(label)
                board.push(move)

    return positions, labels


pgn_path = "../data/lichess_db_standard_rated_2015-03.pgn"
positions, labels = extract_positions_and_labels(pgn_path, max_games=500, min_avg_elo=2500 )

print(f"Extracted {len(positions)} positions.")

np.save("positions.npy", positions)
np.save("labels.npy", labels)
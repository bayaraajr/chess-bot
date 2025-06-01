import chess.pgn
import numpy as np
from tqdm import tqdm

def board_to_tensor(board):
    """Convert board to 8x8x12 binary tensor (1 plane per piece type per color)."""
    tensor = np.zeros((8, 8, 12), dtype=np.int8)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        index = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        tensor[row][col][index] = 1
    return tensor

def move_to_index(move):
    """Convert a move into an integer index."""
    return move.from_square * 64 + move.to_square

def index_to_move(index):
    """Convert index back to move."""
    return chess.Move(from_square=index // 64, to_square=index % 64)

def load_data_from_pgn(pgn_file_path, max_games=1000):
    X, y = [], []
    with open(pgn_file_path) as pgn:
        for _ in tqdm(range(max_games)):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                input_tensor = board_to_tensor(board)
                output_label = move_to_index(move)
                X.append(input_tensor)
                y.append(output_label)
                board.push(move)
    return np.array(X), np.array(y)

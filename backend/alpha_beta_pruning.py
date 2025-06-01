import chess
import numpy as np
import tensorflow as tf

# Load trained model
model_medium = tf.keras.models.load_model("../models/chess_move_cnn_medium.h5")
model_hard = tf.keras.models.load_model("../models/chess_move_cnn_hard.h5")
model_easy = tf.keras.models.load_model("../models/chess_move_cnn.h5")


def board_to_tensor(board):
    tensor = np.zeros((8, 8, 12), dtype=np.int8)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        index = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        tensor[row][col][index] = 1
    return tensor


def move_to_index(move):
    return move.from_square * 64 + move.to_square


def cnn_move_score(board, move, difficulty="easy"):
    print(difficulty)
    if difficulty == "easy":
        model = model_easy
    elif difficulty == "medium":
        model = model_medium
    else:
        model = model_hard

    tensor = board_to_tensor(board)
    tensor = np.expand_dims(tensor, axis=0)  # Shape (1, 8, 8, 12)
    prediction = model.predict(tensor, verbose=0)[0]  # Shape (4096,)
    return prediction[move_to_index(move)]


def evaluate_board_simple(board):
    """Simple material evaluation for leaf nodes."""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3.2,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    score = 0
    for piece_type in piece_values:
        score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    return score


def alpha_beta(board, depth, alpha, beta, maximizing_player, difficulty="easy"):
    print(difficulty)
    if depth == 0 or board.is_game_over():
        return evaluate_board_simple(board)

    legal_moves = list(board.legal_moves)

    # Order moves using CNN scores
    legal_moves.sort(
        key=lambda move: cnn_move_score(board, move, difficulty=difficulty),
        reverse=maximizing_player,
    )

    if maximizing_player:
        max_eval = -float("inf")
        for move in legal_moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = float("inf")
        for move in legal_moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval


def choose_best_move(board, depth=2, difficulty="easy"):
    print(difficulty)
    best_move = None
    best_score = -float("inf") if board.turn == chess.WHITE else float("inf")
    for move in board.legal_moves:
        board.push(move)
        score = alpha_beta(
            board,
            depth - 1,
            -float("inf"),
            float("inf"),
            not board.turn,
            difficulty=difficulty,
        )
        board.pop()

        if (board.turn == chess.WHITE and score > best_score) or (
            board.turn == chess.BLACK and score < best_score
        ):
            best_score = score
            best_move = move

    return best_move

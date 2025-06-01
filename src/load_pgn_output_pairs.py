from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

# Load models
model = tf.keras.models.load_model("../models/chess_move_cnn.h5")
model_medium = tf.keras.models.load_model("../models/chess_move_cnn_medium.h5")
model_hard = tf.keras.models.load_model("../models/chess_move_cnn_hard.h5")


def convert_board_to_model_input(board):
    tensor = np.zeros((8, 8, 12), dtype=np.int8)
    for square, piece in board.piece_map().items():
        row = square // 8
        col = square % 8
        index = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        tensor[row][col][index] = 1
    return np.expand_dims(tensor, axis=0)


def index_to_move(index):
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)


def decode_prediction_to_move(prediction, board):
    pred_index = np.argmax(prediction)
    move = index_to_move(pred_index)

    if move in board.legal_moves:
        return move

    top_indices = np.argsort(prediction[0])[::-1]
    for idx in top_indices:
        move = index_to_move(idx)
        if move in board.legal_moves:
            return move

    return list(board.legal_moves)[0]


@app.route("/predict", methods=["POST"])
def predict_with_cnn():
    data = request.json
    fen = data["fen"]
    difficulty = data.get("difficulty", "easy")
    board = chess.Board(fen)

    model_input = convert_board_to_model_input(board)

    if difficulty == "easy":
        prediction = model.predict(model_input)
    elif difficulty == "medium":
        prediction = model_medium.predict(model_input)
    elif difficulty == "hard":
        prediction = model_hard.predict(model_input)
    else:
        return jsonify({"error": "Invalid difficulty level"}), 400

    best_move = decode_prediction_to_move(prediction, board)

    return jsonify({"move": best_move.uci()})


def evaluate_position_with_model(board, difficulty="easy"):
    model_input = convert_board_to_model_input(board)

    if difficulty == "easy":
        prediction = model.predict(model_input)
    elif difficulty == "medium":
        prediction = model_medium.predict(model_input)
    else:
        prediction = model_hard.predict(model_input)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return 0

    move_scores = []
    for move in legal_moves:
        move_index = move.from_square * 64 + move.to_square
        move_scores.append(prediction[0][move_index])

    return max(move_scores) if board.turn == chess.WHITE else -max(move_scores)


def minimax(board, depth, alpha, beta, maximizing, difficulty):
    if depth == 0 or board.is_game_over():
        return evaluate_position_with_model(board, difficulty)

    legal_moves = list(board.legal_moves)

    if maximizing:
        max_eval = -float("inf")
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, difficulty)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, difficulty)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def choose_best_move_minimax(board, depth=2, difficulty="easy"):
    best_move = None
    best_score = -float("inf") if board.turn == chess.WHITE else float("inf")

    for move in board.legal_moves:
        board.push(move)
        score = minimax(board, depth - 1, -float("inf"), float("inf"), not board.turn, difficulty)
        board.pop()

        if (board.turn == chess.WHITE and score > best_score) or (
            board.turn == chess.BLACK and score < best_score):
            best_score = score
            best_move = move

    return best_move


@app.route("/predict-minimax", methods=["POST"])
def predict_with_minimax():
    data = request.json
    fen = data["fen"]
    difficulty = data.get("difficulty", "easy")
    depth = int(data.get("depth", 2))
    board = chess.Board(fen)

    try:
        best_move = choose_best_move_minimax(board, depth=depth, difficulty=difficulty)
        return jsonify({"move": best_move.uci()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

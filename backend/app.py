from flask import Flask, request, jsonify
import chess
import tensorflow as tf
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains
model = tf.keras.models.load_model('../models/chess_move_cnn.h5')

def convert_board_to_model_input(board):
    """Converts a chess.Board into 8x8x12 numpy array for model input."""
    tensor = np.zeros((8, 8, 12), dtype=np.int8)
    for square, piece in board.piece_map().items():
        row = square // 8
        col = square % 8
        index = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        tensor[row][col][index] = 1
    return np.expand_dims(tensor, axis=0)  # Add batch dimension for model.predict

def index_to_move(index):
    """Convert model's predicted index back to a chess.Move."""
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)

def decode_prediction_to_move(prediction, board):
    """Converts model prediction to a legal move on the board."""
    pred_index = np.argmax(prediction)  # Choose move with highest probability
    move = index_to_move(pred_index)
    
    # If move is legal, return it
    if move in board.legal_moves:
        return move
    
    # If not, try sorted top moves
    top_indices = np.argsort(prediction[0])[::-1]
    for idx in top_indices:
        move = index_to_move(idx)
        if move in board.legal_moves:
            return move

    # If nothing is legal, return a random legal move
    return list(board.legal_moves)[0]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    fen = data["fen"]
    board = chess.Board(fen)

    model_input = convert_board_to_model_input(board)
    prediction = model.predict(model_input)
    best_move = decode_prediction_to_move(prediction, board)

    return jsonify({"move": best_move.uci()})

if __name__ == '__main__':
    app.run(debug=True)

import chess
import chess.engine
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict

model = tf.keras.models.load_model("models/chess_move_cnn_medium.h5")

def board_to_tensor(board):
    tensor = np.zeros((8, 8, 12), dtype=np.int8)
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        index = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        tensor[row][col][index] = 1
    return tensor

def move_to_index(move):
    return move.from_square * 64 + move.to_square

def cnn_score_move(board, move):
    tensor = np.expand_dims(board_to_tensor(board), axis=0)
    prediction = model.predict(tensor, verbose=0)[0]
    return prediction[move_to_index(move)]

def evaluate_material(board):
    values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.2,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    score = 0
    for pt in values:
        score += len(board.pieces(pt, chess.WHITE)) * values[pt]
        score -= len(board.pieces(pt, chess.BLACK)) * values[pt]
    return score

def alpha_beta(board, depth, alpha, beta, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate_material(board)

    legal_moves = list(board.legal_moves)
    legal_moves.sort(key=lambda m: cnn_score_move(board, m), reverse=maximizing)

    if maximizing:
        max_eval = -float("inf")
        for move in legal_moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta, False)
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
            eval = alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def choose_best_move(board, depth=2):
    best_score = -float("inf") if board.turn == chess.WHITE else float("inf")
    best_move = None
    for move in board.legal_moves:
        board.push(move)
        score = alpha_beta(board, depth - 1, -float("inf"), float("inf"), not board.turn)
        board.pop()

        if (board.turn == chess.WHITE and score > best_score) or \
           (board.turn == chess.BLACK and score < best_score):
            best_score = score
            best_move = move
    return best_move

def random_bot(board):
    return random.choice(list(board.legal_moves))

def play_match(cnn_bot_white=True, depth=2):
    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = choose_best_move(board, depth) if cnn_bot_white else random_bot(board)
        else:
            move = random_bot(board) if cnn_bot_white else choose_best_move(board, depth)
        board.push(move)
    return board.result()  # '1-0', '0-1', '1/2-1/2'

def run_tournament(games=20, depth=2):
    results = defaultdict(int)
    for i in range(games):
        result = play_match(cnn_bot_white=(i % 2 == 0), depth=depth)
        print(f"Game {i+1}/{games}: Result = {result}")
        results[result] += 1

    print("\n=== Summary ===")
    print(f"CNN Wins:   {results['1-0'] if games % 2 == 0 else results['0-1']}")
    print(f"Random Wins:{results['0-1'] if games % 2 == 0 else results['1-0']}")
    print(f"Draws:      {results['1/2-1/2']}")
    print(f"Win Rate:   {100 * (results['1-0'] + results['0-1']) / games:.2f}%")

if __name__ == "__main__":
    run_tournament(games=20, depth=2)

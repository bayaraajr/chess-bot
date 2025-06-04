import chess
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict

# Load CNN model trained to evaluate move quality (policy head)
model = tf.keras.models.load_model("models/chess_move_cnn_medium.h5")

# Convert board to input tensor
def board_to_tensor(board):
    tensor = np.zeros((8, 8, 12), dtype=np.int8)
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        index = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        tensor[row][col][index] = 1
    return tensor

# Convert a move to a 1D index in the model's output
def move_to_index(move):
    return move.from_square * 64 + move.to_square

# Score a move using the CNN (policy head)
def cnn_score_move(board, move):
    tensor = np.expand_dims(board_to_tensor(board), axis=0)
    prediction = model.predict(tensor, verbose=0)[0]
    return prediction[move_to_index(move)]

# Material-based board evaluation
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

# Alpha-beta pruning with CNN move ordering
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

# Choose best move using alpha-beta + CNN ordering
def choose_best_move(board, depth=2):
    best_score = -float("inf") if board.turn == chess.WHITE else float("inf")
    best_move = None

    legal_moves = list(board.legal_moves)
    legal_moves.sort(key=lambda m: cnn_score_move(board, m), reverse=board.turn == chess.WHITE)

    for move in legal_moves:
        board.push(move)
        score = alpha_beta(board, depth - 1, -float("inf"), float("inf"), not board.turn)
        board.pop()

        if (board.turn == chess.WHITE and score > best_score) or \
           (board.turn == chess.BLACK and score < best_score):
            best_score = score
            best_move = move
    return best_move

# Random-move bot
def random_bot(board):
    return random.choice(list(board.legal_moves))

# Simulate a single game
def play_match(cnn_bot_white=True, depth=2, verbose=False):
    board = chess.Board()
    while not board.is_game_over():
        if verbose:
            print(board, "\n")
        if board.turn == chess.WHITE:
            move = choose_best_move(board, depth) if cnn_bot_white else random_bot(board)
        else:
            move = random_bot(board) if cnn_bot_white else choose_best_move(board, depth)
        if verbose:
            print(f"{'White' if board.turn == chess.WHITE else 'Black'} plays: {move}")
        board.push(move)
    if verbose:
        print(board)
        print("Result:", board.result())
    return board.result()  # '1-0', '0-1', or '1/2-1/2'

# Run multiple games and summarize
def run_tournament(games=10, depth=1):
    results = defaultdict(int)
    for i in range(games):
        cnn_as_white = (i % 2 == 0)
        result = play_match(cnn_bot_white=cnn_as_white, depth=depth, verbose=True)
        print(f"Game {i+1}/{games}: Result = {result}")
        results[result] += 1

    cnn_wins = results['1-0'] if games % 2 == 0 else results['0-1']
    random_wins = results['0-1'] if games % 2 == 0 else results['1-0']
    draws = results['1/2-1/2']

    print("\n=== Summary ===")
    print(f"CNN Wins:    {cnn_wins}")
    print(f"Random Wins: {random_wins}")
    print(f"Draws:       {draws}")
    win_rate = 100 * (cnn_wins + random_wins) / games
    print(f"Win Rate:    {win_rate:.2f}%")

if __name__ == "__main__":
    run_tournament(games=10, depth=2)

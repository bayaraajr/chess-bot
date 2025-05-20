import chess
import chess.engine

def evaluate_board(board):
    """Simple evaluation: +1 for white win, -1 for black win, 0 otherwise."""
    if board.is_checkmate():
        return -1 if board.turn else 1
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0
    return sum(piece_value(piece) for piece in board.piece_map().values())

def piece_value(piece):
    """Assigns a simple material value to each piece."""
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King is invaluable
    }
    sign = 1 if piece.color == chess.WHITE else -1
    return sign * values[piece.piece_type]

def minimax(board, depth, alpha, beta, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None

    best_move = None
    if maximizing:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

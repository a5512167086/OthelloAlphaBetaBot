import numpy as np

from othello.OthelloGame import BLACK, WHITE
from othello.OthelloUtil import executeMove, getValidMoves, isEndGame


class AlphaBetaBot:
    def __init__(self, depth=7):
        self.depth = depth  # 初始深度

    def evaluateBoard(self, board, color):
        """改進的評估函數：綜合棋子數量差距、棋盤位置權重以及穩定棋子"""
        # 棋盤權重矩陣
        WEIGHTS = np.array(
            [
                [100, -20, 10, 10, -20, 100],
                [-20, -50, -2, -2, -50, -20],
                [10, -2, 1, 1, -2, 10],
                [10, -2, 1, 1, -2, 10],
                [-20, -50, -2, -2, -50, -20],
                [100, -20, 10, 10, -20, 100],
            ]
        )

        # 棋子數量差距
        disk_difference = np.sum(board == color) - np.sum(board == -color)

        # 棋盤權重加權得分
        positional_score = np.sum(board * WEIGHTS * color)

        # 穩定棋子數量計算
        stable_disks = self.countStableDisks(board, color)
        stable_score = stable_disks * 50  # 每顆穩定棋子給予高權重

        return disk_difference + positional_score + stable_score

    def countStableDisks(self, board, color):
        """計算穩定棋子的數量"""
        stable_count = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for y in range(board.shape[0]):
            for x in range(board.shape[1]):
                if board[y, x] == color and self.isStable(board, y, x, directions):
                    stable_count += 1
        return stable_count

    def isStable(self, board, y, x, directions):
        """判斷棋子是否穩定"""
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            while 0 <= ny < board.shape[0] and 0 <= nx < board.shape[1]:
                if board[ny, nx] == 0:  # 空位
                    return False
                if board[ny, nx] == -board[y, x]:  # 對手棋子
                    return False
                ny += dy
                nx += dx
        return True

    def getAction(self, board, color):
        """主入口：搜索最佳步驟"""
        valid_moves = getValidMoves(board, color)
        if valid_moves.size == 0:
            return None

        best_move = None
        best_score = float("-inf")

        for move in valid_moves:
            new_board = board.copy()
            executeMove(new_board, color, move)
            score = self.alphabeta(
                new_board, self.depth - 1, float("-inf"), float("inf"), -color
            )

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def alphabeta(self, board, depth, alpha, beta, color):
        """Alpha-Beta 搜索遞歸"""

        valid_moves = getValidMoves(board, color)
        if valid_moves.size == 0:
            return self.evaluateBoard(board, color)
        elif depth == 0 or isEndGame(board):
            return self.evaluateBoard(board, color)

        if color == BLACK:  # 最大化
            max_eval = float("-inf")
            for move in valid_moves:
                new_board = board.copy()
                executeMove(new_board, color, move)
                eval = self.alphabeta(new_board, depth - 1, alpha, beta, WHITE)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:  # 最小化
            min_eval = float("inf")
            for move in valid_moves:
                new_board = board.copy()
                executeMove(new_board, color, move)
                eval = self.alphabeta(new_board, depth - 1, alpha, beta, BLACK)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

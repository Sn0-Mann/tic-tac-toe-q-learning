import numpy as np

class TicTacToeEnv:
    """The Tic-Tac-Toe game itself. Contains the rules and board, but nothing else."""

    def __init__(self):
        # Board represented as a flat list of 9 cells (positions 0-8),
        # left-to-right, top-to-bottom. 0 = empty, 1 = player 1 (X), -1 = player 2 (O)
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1  # player 1 (X) always starts

    def reset(self, starting_player=1):
        self.board = np.zeros(9, dtype=int)
        self.current_player = starting_player
        return self.board.copy()

    def legal_actions(self):
        # Returns a list of all empty cell positions (valid moves).
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action):
        # Places the current player's mark at the given position (action).
        # Returns: (new_board, reward, done)
        if self.board[action] != 0:
            # Illegal move / cell already taken
            return self.board.copy(), -10, True

        self.board[action] = self.current_player

        winner = self.check_winner()
        if winner != 0:
            return self.board.copy(), 1, True  # current player just won

        if len(self.legal_actions()) == 0:
            return self.board.copy(), -0.1, True  # draw, board full

        self.current_player *= -1  # switch turns
        return self.board.copy(), 0, False  # game continues

    def check_winner(self):
        # Returns 1 or -1 if that player has three in a row, else 0.
        lines = [
            [0,1,2], [3,4,5], [6,7,8],  # rows
            [0,3,6], [1,4,7], [2,5,8],  # columns
            [0,4,8], [2,4,6],           # diagonals
        ]
        for line in lines:
            total = self.board[line[0]] + self.board[line[1]] + self.board[line[2]]
            if total == 3:
                return 1
            if total == -3:
                return -1
        return 0

    def render(self):
        # Prints the board as a readable 3x3 grid.
        symbols = {1: "X", -1: "O", 0: "·"}
        for row in range(3):
            cells = self.board[row*3:(row+1)*3]
            print(" ".join(symbols[c] for c in cells))
        print()

if __name__ == "__main__":
    env = TicTacToeEnv()
    env.reset()

    moves = [0, 3, 1, 4, 7, 2, 6, 8, 5]
    for move in moves:
        state, reward, done = env.step(move)
        env.render()
        print("reward:", reward, "done:", done)
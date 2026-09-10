import random
import pickle

# the 8 ways a tic-tac-toe board can be rotated/flipped and still
# represent the "same" strategic situation
SYMMETRIES = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],  # identity, no change
    [6, 3, 0, 7, 4, 1, 8, 5, 2],  # rotate 90
    [8, 7, 6, 5, 4, 3, 2, 1, 0],  # rotate 180
    [2, 5, 8, 1, 4, 7, 0, 3, 6],  # rotate 270
    [2, 1, 0, 5, 4, 3, 8, 7, 6],  # flip left-right
    [0, 3, 6, 1, 4, 7, 2, 5, 8],  # flip along main diagonal
    [6, 7, 8, 3, 4, 5, 0, 1, 2],  # flip top-bottom
    [8, 5, 2, 7, 4, 1, 6, 3, 0],  # flip along other diagonal
]


def canonical(state, action):
    best_state = None
    best_action = None
    for perm in SYMMETRIES:
        transformed = tuple(state[perm[i]] for i in range(9))
        transformed_action = perm.index(action)

        # prefer a smaller state, but if tied, prefer a smaller action --
        # keeps things consistent even when the board itself is symmetric
        if (best_state is None
                or transformed < best_state
                or (transformed == best_state and transformed_action < best_action)):
            best_state = transformed
            best_action = transformed_action

    return best_state, best_action

class QLearningAgent:
    # no neural network here. just a big dictionary.
    # value of moves

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}       # knows nothing yet
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # how much it cares about the future
        self.epsilon = epsilon  # % chance for exploration

    def get_q(self, state, action):
        state, action = canonical(state, action)
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, legal_actions):
        # explore or exploit?
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        # check every legal move, see what it currently rates best
        q_values = [self.get_q(state, a) for a in legal_actions]
        max_q = max(q_values)

        # ties happen a lot early on. don't always pick the first one.
        best_actions = [a for a, q in zip(legal_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_legal_actions, done):
        state, action = canonical(state, action)   # convert once, use everywhere below
        old_q = self.get_q(state, action)

        if done or len(next_legal_actions) == 0:
            future_estimate = 0
        else:
            future_estimate = max(self.get_q(next_state, a) for a in next_legal_actions)

        new_q = old_q + self.alpha * (reward + self.gamma * future_estimate - old_q)
        self.q_table[(state, action)] = new_q

    def save(self, filename):
        # just a plain dict, pickle handles it fine
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)

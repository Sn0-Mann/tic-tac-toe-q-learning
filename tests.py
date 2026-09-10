from agent import QLearningAgent
import numpy as np

agent = QLearningAgent()
agent.load("agent1_qtable.pkl")

empty_board = np.zeros(9, dtype=int)
for action in range(9):
    print(action, agent.get_q(empty_board, action))
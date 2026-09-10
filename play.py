from environment import TicTacToeEnv
from agent import QLearningAgent

env = TicTacToeEnv()
agent = QLearningAgent(epsilon=0)

choice = input("\ngo first? (y/n): ").lower()

if choice == "y":
    human_mark = 1
    agent_mark = -1
    agent.load("agent2_qtable.pkl")   # agent's playing second, needs second-mover experience
else:
    human_mark = -1
    agent_mark = 1
    agent.load("agent1_qtable.pkl")   # agent's playing first, needs first-mover experience

def print_legend():
    print("\npositions:")
    print("0 1 2")
    print("3 4 5")
    print("6 7 8")
    print()

print_legend()

state = env.reset(starting_player=1)   # mark 1 always moves first, matches training
env.render()
done = False

while not done:
    if env.current_player == agent_mark:
        action = agent.choose_action(state, env.legal_actions())
        print(f"agent plays: {action}")
    else:
        action = int(input("your move (0-8): "))

    state, reward, done = env.step(action)
    env.render()

    if done:
        if reward == 1:
            winner = "agent" if env.current_player == agent_mark else "you"
            print(f"{winner} wins!")
        elif reward == -10:
            print("illegal move!")
        else:
            print("draw!")
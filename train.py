from environment import TicTacToeEnv
from agent import QLearningAgent

env = TicTacToeEnv()
agent1 = QLearningAgent()   # plays X (1)
agent2 = QLearningAgent()   # plays O (-1)

NUM_EPISODES = 1000000   # one episode = one full game

for episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    last = {1: None, -1: None}   # each player's last (state, action)

    while not done:   # play until the game ends
        player = env.current_player
        agent = agent1 if player == 1 else agent2
        legal = env.legal_actions()
        action = agent.choose_action(state, legal)   # explore or exploit

        # close out this player's previous move, nothing ended it, reward 0
        if last[player] is not None:
            prev_state, prev_action = last[player]
            agent.update(prev_state, prev_action, 0, state, legal, False)

        next_state, reward, done = env.step(action)   # actually make the move
        last[player] = (state, action)   # remember it for next time

        if done:   # game just ended on this move
            agent.update(state, action, reward, next_state, [], True)

            # other player didn't get to react, but still needs closure
            other_player = -player
            if last[other_player] is not None:
                other_agent = agent1 if other_player == 1 else agent2
                prev_state, prev_action = last[other_player]
                other_reward = -1 if reward == 1 else reward   # my win = your loss
                other_agent.update(prev_state, prev_action, other_reward, next_state, [], True)

        state = next_state

    if (episode + 1) % 100000 == 0:   # just a progress check-in
        print(f"Episode {episode + 1}/{NUM_EPISODES} done")

agent1.save("agent1_qtable.pkl")   # save learned tables so we don't retrain
agent2.save("agent2_qtable.pkl")
print("done, both q-tables saved")
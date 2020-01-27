#!/usr/bin/env python
# coding: utf-8

"""
### Task 1 - Environment ###
Create the environment of your choice.
Example of such environments is the GridWorld environment.
You can here do whatever you want, but* I recommend that you create a **deterministic** environment.
Use the template provided (class Environment)

### Task 2 - Agents
Implement SARSA and Q-Learning, use the supplied template(s) for the implementation.
You can choose to use the same class for both agents if you desire to do so.
(Set the use flag as described in the code)

http://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf
"""


import numpy as np
import matplotlib.pyplot as plt


results = []  # Record results in this list


class Environment:
    def __init__(self):
        self.state = 5
        self.start = 5
        self.goal = 10
        self.reward = -1

    def step(self, _action):
        _terminal = False
        _reward = self.reward
        if _action == ACTION_LEFT:
            if self.state > 1:
                self.state -= 1
        if _action == ACTION_RIGHT:
            if self.state < 10:
                self.state += 1

        if self.state == self.goal:
            _terminal = True
            _reward = 100

        return self.state, _reward, _terminal

    def reset(self):
        self.state = self.start
        return self.state

    def render(self, mode='human', close=False):
        pass
        # return state # Return the state information here


class Agent:
    def __init__(self, lr, discount):
        self.lr = lr
        self.discount = discount
        self.use = False
        self.Q = np.zeros((STATES, len(ACTIONS)))

    def predict(self, s):
        if np.random.binomial(1, EPSILON) == 1:
            return np.random.choice(ACTIONS)
        else:
            values_ = self.Q[s, :]
            # return np.argmax(values_)
            return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])


class SarsaAgent(Agent):
    def __init__(self, lr, discount):
        super().__init__(lr, discount)
        self.use = True
        self.name = "SARSA"

    def learn(self, s, a, r, s1, a1):
        self.Q[s, a] += ALPHA * (r + self.Q[s1, a1] - self.Q[s, a])


class QLearningAgent(Agent):
    def __init__(self, lr, discount):
        super().__init__(lr, discount)
        self.use = True
        self.name = "Q_LEARNING"

    def learn(self, s, a, r, s1):
        self.Q[s, a] += ALPHA * (r + GAMMA * np.max(self.Q[s1, :]) - self.Q[s, a])


if __name__ == "__main__":

    EPISODES = 100      #
    MAX_STEPS = 100     #
    LR = 0.2            #
    GAMMA = 0.95        #
    EPSILON = 0.1       # Probability for exploration
    ALPHA = 0.5         # Step size

    STATES = 11
    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTIONS = [ACTION_LEFT, ACTION_RIGHT]

    agents = [
        SarsaAgent(LR, GAMMA),
        QLearningAgent(LR, GAMMA),
        # Agent(LR, GAMMA)
    ]

    for agent in [x for x in agents if x.use]:

        # Initialise environment
        env = Environment()

        for i in range(EPISODES):

            # Reset environment
            state, step, terminal = env.reset(), 0, False

            while not terminal and step <= MAX_STEPS:

                # Predict/choose action given state pi(a | s)
                action = agent.predict(state)

                # Let environment transition from s to s' with its Pss' function
                state1, reward, terminal = env.step(action)
                if terminal:
                    break

                # Update algorithm
                if agent.name == "SARSA":
                    action1 = agent.predict(state1)
                    agent.learn(state, action, reward, state1, action1)
                    action = action1

                if agent.name == "Q_LEARNING":
                    agent.learn(state, action, reward, state1)

                # state is now the next state
                state = state1

                # For plotting
                step += 1

            # Record number of steps used for this episode
            results.append(step)

        # Plot
        plt.plot(np.arange(len(results)), results)
        plt.show()
        plt.clf()

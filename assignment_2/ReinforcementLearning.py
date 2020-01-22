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

############################################################
# Hva med å bare bruke openAi sine "Gyms" som Environment? #
# env = gym.make('CartPole-v0')                            #
# https://pylessons.com/CartPole-reinforcement-learning/   #
############################################################

import numpy as np
import matplotlib.pyplot as plt


results = []  # Record results in this list


class Environment:
    def __init__(self):
        # Define objects here, such as map.. etc
        pass

    def step(self, action):
        # Code the game logic here
        # return self.render()
        pass

    def reset(self):
        # Code a reset function here
        # return self.render()
        pass

    def render(self, mode='human', close=False):
        pass
        # return state # Return the state information here


class Agent:
    def __init__(self, lr, discount):
        self.use = False  # TODO - Set this to true if you implement here

    def learn(self, s, a, r, s1, t):
        pass

    def predict(self, s):
        pass
        # return action # Return the action here


class SarsaAgent(Agent):
    def __init__(self, lr, discount):
        super().__init__(lr, discount)
        self.use = False  # TODO - Set this to true if you implement here


class QLearningAgent(Agent):
    def __init__(self, lr, discount):
        super().__init__(lr, discount)
        self.use = False  # TODO - Set this to true if you implement here


if __name__ == "__main__":

    EPISODES = 200      # TODO - Set number of episodes here
    MAX_STEPS = 200     # TODO - Max number of steps per episode
    LR = 0.2            # TODO - Set the learning rate
    GAMMA = 0.95        # TODO - Set the discount factor

    agents = [
        SarsaAgent(LR, GAMMA),
        QLearningAgent(LR, GAMMA),
        Agent(LR, GAMMA)
    ]

    for agent in [x for x in agents if x.use]:

        # Initialise environment
        # TODO - Do we need to create a new environment for each agent?
        env = Environment()

        for i in range(EPISODES):

            # Reset environment
            state, step, terminal = env.reset(), 0, False

            while not terminal and step <= MAX_STEPS:

                # Predict action given state pi(a | s)
                action = agent.predict(state)

                # Let environment transition from s to s' with its Pss' function
                state1, reward, terminal, _ = env.step(action)

                # Update algorithm
                agent.learn(state, action, reward, state1, terminal)

                # s is now s'
                state = state1

                # For plotting results
                step += 1

            # Record number of steps used for this episode
            results.append(step)

        # Plot
        plt.plot(np.arange(len(results)), results)
        plt.show()
        plt.clf()

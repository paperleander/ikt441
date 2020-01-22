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

To implement the algorithms, you can find implementation details here:
http://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf
or on google :)
"""



import matplotlib.pyplot as plt
results = []  # Record results in this list

class Environment:
    
    def __init__(self):
        # Define objects here, such as map.. etc
        pass
    
    def step(self, action):
        # Code the game logic here
        return self.render()
    
    def reset(self):
        # Code a reset function here
        return self.render()
    
    def render(self, mode='human', close=False):
        pass
        #return state # Return the state information here
        
class Agent:
    
    def __init__(self, lr, discount):
        self.use = False  # TODO - Set this to true if you implement here
    
    def learn(self, s, a, r, s1, t):
        pass
    
    def predict(s):
        pass
        #return action # Return the action here

class SarsaAgent(Agent):
    
    def __init__(self, lr, discount):
        super().__init__(lr, discount)
        self.use = False  # TODO - Set this to true if you implement here
    
class QLearningAgent(Agent):
    def __init__(self, lr, discount):
        super().__init__(lr, discount)
        self.use = False  # TODO - Set this to true if you implement here

        
EPISODES = 200 # TODO - Set number of episodes here
MAX_STEPS = 200 # TODO - Max number of steps per episode
LR = None  # TODO - Set the learning rate
GAMMA = None # TODO - Set the discount factor
env_args = () # TODO - Arguments to the environment

agents = [
    SarsaAgent(LR, GAMMA), 
    QLearningAgent(LR, GAMMA),
    Agent(LR, GAMMA),
    Agent(LR, GAMMA)
]


for agent in [x for x in agents if x.use]:
    
    # Initialise environment
    env = Environment(*env_args)
    
    for i in range(EPISODES):
        
        # Reset environment 
        s, step, t = env.reset(), 0, False
        
        # While not terminal and step is below limit
        while not t and step <= MAX_STEPS:
            
            # Predict action given state pi(a | s)
            a = agent.predict(s)
            
            # Let environment transition from s to s' with its Pss' function
            s1, r, t, _ = env.step(a)
            
            # Update algorithm
            agent.learn(s, a, r, s1, t)
            
            # s is now s' 
            s = s1 
            step += 1
        
        # Record number of steps used for this episode
        results.append(step)
    
    # Plot
    plt.plot(np.arange(len(results)), results)
    plt.show()
    plt.clf()


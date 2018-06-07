import gym
import numpy as np
from numpy import float32, uint32
env = gym.make('CartPole-v0')
### set parameters and set ###
discount_factor=0.99
eposide_length=[]
expected_value=[]
for i_episode in range(100):

    observation_init = env.reset()

    for t in range(300):
        ### select action by uniform distribution ###
        action= np.random.uniform(0,1,1)

        action=np.round(action)
        action=int(action)

        observation, reward, done, info = env.step(action)

        #print(reward)
        if done:
            ### when each eposide ended record the return and eposide's length
            print("Episode length is  {} ".format(t+1))
            eposide_length.append(t+1)
            reward=-1
            reward_return=reward*(discount_factor**(t))
            expected_value.append(reward_return)
            break


print("the episode's  length", eposide_length)
print('the mean of episode length',np.mean(eposide_length))

print('the standard deviation of episode length',np.std(eposide_length))

print('....expected return from the initial state.....')
print(expected_value)
print('the mean of initial return',np.mean(expected_value,axis=0))
print('the standard deviation of initial return', np.std(expected_value,axis=0))

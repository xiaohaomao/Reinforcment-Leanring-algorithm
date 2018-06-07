import gym
import numpy as np

env = gym.make('CartPole-v0')

### set parameters and set ###
discount_factor=0.99
eposide_length=[]
expect_value=[]

for i_episode in range(3):

    observation_init = env.reset()
    for t in range(300):
        env.render()
        ### select action by uniform distribution ###
        action= np.random.uniform(0,1,1)
        action=np.round(action)
        action=int(action)

        observation, reward, done, info = env.step(action)

        if done:
            ### when each eposide ended record the return and eposide's length
            reward=-1
            reward_return=reward*(discount_factor**(t))
            expect_value.append(reward_return)

            print("Episode length is  {} ".format(t+1))
            eposide_length.append(t+1)

            break

print("the trajectories'length :",eposide_length)
print("the return from the starting state:",expect_value)

#### collect data based on random policy with 2000 eposides ####
import gym
import numpy as np
from numpy import float32, uint32

env = gym.make('CartPole-v0')

discount_factor=0.99
eposide_length=[]
expected_value=[]
transition=[]
for i_episode in range(2000):
    print(i_episode)
    observation_init = env.reset()

    for t in range(300):

        env.render()

        action= np.random.uniform(0,1,1)
        action=np.round(action)
        action=int(action)
        observation, reward, done, info = env.step(action)

        action=np.array(action)


        if done is False:
            reward = 0
            reward = np.array(reward)
            print(observation, reward, done, info)


            if t==0:
                this_observation=observation_init
                next_observation=observation
                transition.append((this_observation,next_observation,reward,action))

            else:
                this_observation=next_observation
                next_observation=observation
                transition.append((this_observation,next_observation,reward,action))

        if done is True:
            print("Episode length is  {} ".format(t+1))
            eposide_length.append(t+1)
            reward=-1
            reward=np.array(reward)
            this_observation = next_observation
            next_observation = observation
            transition.append((this_observation, next_observation, reward, action))

            reward_return=reward*(discount_factor**(t))
            expected_value.append(reward_return)

            break

print("the episode's  length", eposide_length)
print('the mean of episode length',np.mean(eposide_length))


print('the standard deviation of episode length',np.std(eposide_length))


print('....expected return from the initial state.....')
print('the expected value of return',expected_value)

outfile1 =np.array(transition)
np.save('train_data_2',outfile1)





















import gym
import numpy as np
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')
train_reward=np.load('reward_data_part3_4_300.npy')
train_length=np.load('length_data_part3_4_300.npy')
train_loss=np.load('loss_data_part3_4_300.npy')


plt.plot(np.mean(train_loss,axis=0))
plt.xlabel('ith Num of training episode')
plt.ylabel('train_Mean loss')
plt.show()
plt.plot(np.mean(train_reward,axis=0))
plt.xlabel('ith Num of training episode')
plt.ylabel('train_Mean reward')
plt.show()

plt.plot(np.mean(train_length,axis=0))
plt.xlabel('ith Num of training episode')
plt.ylabel('train_Mean length')
plt.show()



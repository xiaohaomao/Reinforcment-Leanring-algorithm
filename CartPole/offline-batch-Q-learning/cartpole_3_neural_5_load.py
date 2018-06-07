import gym
import numpy as np
import tensorflow as tf
import random
import os

import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env._max_episode_steps = 300



print("......Loading train_data......")
### load the stored data ###
### load the stored data ###
train_data=np.load('train_data_2.npy')


#### set variable and parameters ####
x1=tf.placeholder(tf.float32, shape=[None,4])
x2=tf.placeholder(tf.float32, shape=[None,4])
x3=tf.placeholder(tf.float32, shape=[None])
x4=tf.placeholder(tf.float32, shape=[None])
x5=tf.placeholder(tf.float32, shape=[None])

##### set the variable#######
batch_size=1000
discount=0.99
learn_rate=0.0001
input_size=4
hidden_size=100
output_size=2
eplison=0.05


Weight_1=tf.Variable(tf.truncated_normal(shape=[input_size,hidden_size]))
Weight_2=tf.Variable(tf.truncated_normal(shape=[hidden_size,output_size]))
Bias_1=tf.Variable(tf.constant(0.1,shape=[hidden_size]))
Bias_2=tf.Variable(tf.constant(0.1,shape=[output_size]))



###   one hiddle layer neural network as function approximation ###
middle_now=tf.matmul(x1,Weight_1)+Bias_1
prediction_No=tf.nn.relu(middle_now)
prediction_now=tf.matmul(prediction_No,Weight_2)+Bias_2

middle_next=tf.matmul(x2,Weight_1)+Bias_1
prediction_Ne=tf.nn.relu(middle_next)
prediction_next=tf.matmul(prediction_Ne,Weight_2)+Bias_2


test_action=tf.cast(tf.argmax(prediction_now,1),tf.int32)

True_action=tf.cast(x4,tf.int32)
True_action=tf.reshape(True_action,shape=[-1,1])
action_repeat=tf.reshape(tf.cast(x5,tf.int32),shape=[-1,1])
action_double=tf.concat([action_repeat,True_action],1)

### calcaulate the loss and training ###
Q_value=tf.gather_nd(params=prediction_now,indices=action_double)

Max_Q_value_next=tf.reduce_max(prediction_next,axis=1)


delta=tf.add(x3+discount*tf.stop_gradient((1+x3)*Max_Q_value_next),(-1*Q_value))

q_loss=tf.reduce_sum(tf.square(delta))/2

train_optimizer=tf.train.AdamOptimizer(learn_rate).minimize(q_loss)




#### save the model ####
saver=tf.train.Saver()

### use gpu us training data ###
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        ### reload the model ###
        saver.restore(sess, './part3_neural_5_300/')

        eposide_length = []
        expected_value = []

        test_size=50
        all_eposide_length = np.zeros((1, test_size))
        all_reward = np.zeros((1, test_size))

        ### test the performance for final model ###
        ### reset 50 times to test the performance ###

        for i_episode in range(test_size):

            observation_init = env.reset()
            observation_init = [observation_init]

            for t in range(300):

                if t == 0:

                    Action = test_action.eval(feed_dict={x1: observation_init, x2: observation_init})
                    observation_curr, reward_curr, done, info = env.step(Action[0])

                    observation_next = [observation_curr]
                else:
                    Action = test_action.eval(feed_dict={x1: observation_next, x2: observation_next})

                    observation_curr, reward_curr, done, info = env.step(Action[0])
                    observation_next = [observation_curr]

                if done is True:

                    eposide_length.append(t + 1)
                    reward = -1
                    reward_return = reward * (discount ** (t))
                    expected_value.append(reward_return)

                    break
            all_eposide_length[0, i_episode] = t + 1
            all_reward[0, i_episode] = reward_return


        all_eposide_length = np.sum(all_eposide_length, axis=0)
        all_reward = np.sum(all_reward, axis=0)



        print('the mean of episode length', np.mean(eposide_length))
        print('the mean of  episode length', np.mean(all_reward))

        print('the standard deviation of episode length', np.std(eposide_length))
        ### print the eposide length and all reward during test ###
        plt.plot(all_eposide_length)
        plt.xlabel('Num of episode')
        plt.ylabel('length of eposide')
        plt.show()
        plt.plot(all_reward)
        plt.xlabel('Num of episode')
        plt.ylabel('reward')


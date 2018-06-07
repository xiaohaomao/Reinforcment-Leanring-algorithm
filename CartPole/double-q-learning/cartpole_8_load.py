
import gym
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env._max_episode_steps = 300

print("......Loading train_data......")


train_data=np.load('train_data_2.npy')

#### set variable and parameters ####

x1=tf.placeholder(tf.float32, shape=[None,4])
x2=tf.placeholder(tf.float32, shape=[None,4])
x3=tf.placeholder(tf.float32, shape=[None,2])
x4=tf.placeholder(tf.float32, shape=[None])
x5=tf.placeholder(tf.float32,shape=[None])



batch_size=128
discount=0.99
learn_rate=0.0001
input_size=4
hidden_size=100
output_size=2
max_eposide_length=300
eplison=0.05

Weight_1=tf.Variable(tf.truncated_normal(shape=[input_size,hidden_size]))
Weight_2=tf.Variable(tf.truncated_normal(shape=[hidden_size,output_size]))
Bias_1=tf.Variable(tf.constant(0.1,shape=[hidden_size]))
Bias_2=tf.Variable(tf.constant(0.1,shape=[output_size]))

Weight_double_1=tf.Variable(tf.truncated_normal(shape=[input_size,hidden_size]))
Weight_double_2=tf.Variable(tf.truncated_normal(shape=[hidden_size,output_size]))
Bias_double_1=tf.Variable(tf.constant(0.1,shape=[hidden_size]))
Bias_double_2=tf.Variable(tf.constant(0.1,shape=[output_size]))

middle_now=tf.matmul(x1,Weight_1)+Bias_1
prediction_No=tf.nn.relu(middle_now)
prediction_now=tf.matmul(prediction_No,Weight_2)+Bias_2


middle_next=tf.matmul(x2,Weight_1)+Bias_1
prediction_Ne=tf.nn.relu(middle_next)
prediction_next=tf.matmul(prediction_Ne,Weight_2)+Bias_2


middle_now_double=tf.matmul(x1,Weight_double_1)+Bias_double_1
prediction_No_double=tf.nn.relu(middle_now_double)
prediction_now_double=tf.matmul(prediction_No_double,Weight_double_2)+Bias_double_2

middle_next_double=tf.matmul(x2,Weight_double_1)+Bias_double_1
prediction_Ne_double=tf.nn.relu(middle_next_double)
prediction_next_double=tf.matmul(prediction_Ne_double,Weight_double_2)+Bias_double_2
#


test_action=tf.cast(tf.argmax(prediction_now,1),tf.int32)
test_action_double=tf.cast(tf.argmax(prediction_now_double,1),tf.int32)

True_action=tf.cast(x3,tf.int32)

Q_value=tf.gather_nd(params=prediction_now,indices=True_action)
Q_value_double=tf.gather_nd(params=prediction_now_double,indices=True_action)



next_action_b=tf.cast(tf.argmax(prediction_next_double,1),tf.int32)
next_action_b=tf.reshape(next_action_b,[-1,1])
action_repeat_b=tf.reshape(tf.cast(x5,tf.int32),shape=[-1,1])#
action_b_next=tf.concat([action_repeat_b,next_action_b],1)

next_action_a=tf.cast(tf.argmax(prediction_next,1),tf.int32)
next_action_a=tf.reshape(next_action_a,[-1,1])
action_repeat_a=tf.reshape(tf.cast(x5,tf.int32),shape=[-1,1])#
action_a_next=tf.concat([action_repeat_a,next_action_a],1)


Max_Q_value_next=tf.gather_nd(params=prediction_next_double,indices=action_a_next)
Max_Q_value_next_double=tf.gather_nd(params=prediction_next,indices=action_b_next)

delta_a=tf.add(x4+discount*tf.stop_gradient((1+x4)*Max_Q_value_next),(-1*Q_value))
delta_b=tf.add(x4+discount*tf.stop_gradient((1+x4)*Max_Q_value_next_double),(-1*Q_value_double))


q_loss_a=tf.reduce_mean((tf.square(delta_a))/2)

q_loss_b=tf.reduce_mean((tf.square(delta_b))/2)

train_optimizer_a=tf.train.AdamOptimizer(learn_rate).minimize(q_loss_a)

train_optimizer_b=tf.train.AdamOptimizer(learn_rate).minimize(q_loss_b)



saver = tf.train.Saver()


with tf.device('/cpu:0'):
    with tf.Session() as sess:
        ## reload the weights ###
        saver.restore(sess, './part8_double_dqn/')
        eposide_length = []
        expected_value = []
        all_eposide_length = np.zeros((1, 10))
        all_reward = np.zeros((1, 100))

        for i_episode in range(10):

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

        all_eposide_length = np.mean(all_eposide_length, axis=0)
        all_reward = np.mean(all_reward, axis=0)



        print('the mean of episode length', np.mean(eposide_length))
        print('the mean of reward ',np.mean(expected_value))

        print('the standard deviation of episode length', np.std(eposide_length))
        plt.plot(all_eposide_length)
        plt.xlabel('Num of episode')
        plt.ylabel('length of eposide')
        plt.show()







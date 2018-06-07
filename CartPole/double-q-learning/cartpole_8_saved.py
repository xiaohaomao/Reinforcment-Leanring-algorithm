import gym
import numpy as np
import tensorflow as tf
import random
import os

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

### the second neural network ###
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

test_action=tf.cast(tf.argmax(prediction_now,1),tf.int32)
test_action_double=tf.cast(tf.argmax(prediction_now_double,1),tf.int32)

True_action=tf.cast(x3,tf.int32)

Q_value=tf.gather_nd(params=prediction_now,indices=True_action)
Q_value_double=tf.gather_nd(params=prediction_now_double,indices=True_action)

### calculate the target by the actual action which calculate by current network ###
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


### calculate the loss by update network a or b ###
delta_a=tf.add(x4+discount*tf.stop_gradient((1+x4)*Max_Q_value_next),(-1*Q_value))
delta_b=tf.add(x4+discount*tf.stop_gradient((1+x4)*Max_Q_value_next_double),(-1*Q_value_double))


q_loss_a=tf.reduce_mean((tf.square(delta_a))/2)
q_loss_b=tf.reduce_mean((tf.square(delta_b))/2)

train_optimizer_a=tf.train.AdamOptimizer(learn_rate).minimize(q_loss_a)
train_optimizer_b=tf.train.AdamOptimizer(learn_rate).minimize(q_loss_b)


saver = tf.train.Saver()

with tf.device('/cpu:0'):

    eposide_size = 2000
    run_size = 1
    all_episode_length = np.zeros((run_size, int(eposide_size)))
    all_total_reward = np.zeros((run_size, int(eposide_size)))
    all_test_episode_length = np.zeros((run_size, int(eposide_size)))
    all_test_reward = np.zeros((run_size, int(eposide_size / 20)))
    all_train_loss = np.zeros((run_size, int(eposide_size / 20)))

    length_of_train = len(train_data)
    for i_run in range(1, run_size + 1):
        ### build the experience replay ###

        buffer_size = 1024
        mini_batch_size = 64

        length_of_train=len(train_data)

        buffer_sample=random.sample(range(0, length_of_train), buffer_size)
        buffer_replay=train_data[buffer_sample]

        buffer_observation_now = []
        buffer_observation_next=[]
        buffer_action=[]
        buffer_reward=[]

        for i_sele in range(buffer_size):
            buffer_observation_now.append( buffer_replay[i_sele][0])
            buffer_observation_next.append( buffer_replay[i_sele][1])
            buffer_reward.append( buffer_replay[i_sele][2])
            buffer_action.append( buffer_replay[i_sele][3])


        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for i_eposide in range(1,1+eposide_size):

                observation_0 = env.reset()

                total_QQ_loss = 0

                for i_step in range(max_eposide_length):

                    if np.random.random() <= eplison:
                        action_select_now = np.random.randint(2)
                    else:
                        Q = sess.run(test_action, feed_dict={x1: np.reshape(observation_0, [1, 4])})
                        action_select_now=int(Q)

                    observation_1, _, done_0, _ = env.step(action_select_now)

                    if done_0:
                        reward = -1
                    else:
                        reward = 0

                    # ##add new data to replay memory
                    buffer_observation_now = np.append(buffer_observation_now, np.reshape(observation_0, [1, 4]), axis=0)
                    buffer_observation_next = np.append(buffer_observation_next, np.reshape(observation_1, [1, 4]), axis=0)
                    buffer_action = np.append(buffer_action, [action_select_now], axis=0)
                    buffer_reward = np.append(buffer_reward, [reward], axis=0)

                    ### update the first neural network ###
                    if np.random.randint(2) == 0:
                        select_order = np.arange(mini_batch_size)

                        this_batch = random.sample(range(len(buffer_replay)), mini_batch_size)



                        _, loss_train = sess.run([train_optimizer_a, q_loss_a], feed_dict={x1: buffer_observation_now[this_batch, :],
                                                                             x2: buffer_observation_next[this_batch, :],
                                                                             x3: np.concatenate((np.reshape(
                                                                                 np.arange(mini_batch_size),
                                                                                 [mini_batch_size, 1]), np.reshape(
                                                                                 buffer_action[this_batch],
                                                                                 [mini_batch_size, 1])), axis=1)
                                                                                , x4: buffer_reward[this_batch],
                                                                                   x5:select_order})
                    else:
                        ### update the second neural network ###
                        this_batch = random.sample(range(len(buffer_replay)), mini_batch_size)
                        select_order = np.arange(mini_batch_size)

                        _, loss_train = sess.run([train_optimizer_b, q_loss_b],
                                                 feed_dict={x1: buffer_observation_now[this_batch, :],
                                                            x2: buffer_observation_next[this_batch, :],
                                                            x3: np.concatenate((np.reshape(
                                                                np.arange(mini_batch_size),
                                                                [mini_batch_size, 1]), np.reshape(
                                                                buffer_action[this_batch],
                                                                [mini_batch_size, 1])), axis=1)
                                                     , x4: buffer_reward[this_batch],
                                                            x5:select_order})

                    total_QQ_loss +=loss_train

                    observation_0 = observation_1

                    if (i_eposide - 1) % 20 == 0:

                        if done_0 is True:
                            if i_step + 1 == 300:
                                report_reward = 0
                            else:
                                report_reward = -1 * discount ** (i_step)

                            all_episode_length[i_run - 1, i_eposide - 1] = i_step + 1
                            all_total_reward[i_run - 1, i_eposide - 1] = report_reward

                            ### record average test performance ###
                            test_size = 10
                            Small_test_eposide_length = np.zeros((1, test_size))
                            Small_test_reward = np.zeros((1, test_size))

                            for i_test_run in range(1, test_size + 1):
                                observation_test_0 = env.reset()

                                for i_test_length in range(max_eposide_length):
                                    action_test_now = test_action.eval(
                                        feed_dict={x1: np.reshape(observation_test_0, [1, 4])})
                                    action_test_now = int(action_test_now)
                                    observation_test_1, _, test_done, test_info = env.step(action_test_now)

                                    observation_test_0 = observation_test_1

                                    if test_done is True:
                                        if i_test_length+1==300:
                                            reward_test=0
                                        else:
                                            reward_test=-1
                                        Small_test_eposide_length[0, i_test_run - 1] = i_test_length + 1
                                        Small_test_reward[0, i_test_run - 1] = reward_test * (
                                        discount ** (i_test_length))


                                        break

                            small_mean_test_length = np.mean(np.mean(Small_test_eposide_length, axis=0), axis=0)
                            small_mean_test_reward = np.mean(np.mean(Small_test_reward, axis=0), axis=0)
                            print('the ith running',i_run,'the ith eposide', i_eposide - 1, 'the test_average_length',
                                  small_mean_test_length,
                                  'the total_test_length ', Small_test_eposide_length, '..loss..',
                                  total_QQ_loss / (i_step + 1))
                            all_test_episode_length[i_run - 1, int((i_eposide - 1) / 20)] = small_mean_test_length

                            all_test_reward[i_run - 1, int((i_eposide - 1) / 20)] = small_mean_test_reward
                            all_train_loss[i_run - 1, int((i_eposide - 1) / 20)] = total_QQ_loss / (i_step + 1)


                            if all_test_episode_length[i_run-1, int((i_eposide - 1) / 20)] == np.amax(
                            all_test_episode_length):

                                print('.....', all_test_episode_length[i_run-1, int((i_eposide - 1) / 20)])
                                print(np.amax(all_test_episode_length))



                                if not os.path.exists('./part8_double_dqn/'):
                                    os.mkdir('./part8_double_dqn/')
                                saver.save(sess, "./part8_double_dqn/")
                                print('saved')

                            break
                    else:
                        if done_0 is True:
                            reward = -1

                            final_reward = reward * discount ** (i_step)

                            all_episode_length[i_run - 1, i_eposide - 1] = i_step + 1
                            all_total_reward[i_run - 1, i_eposide - 1] = final_reward

                            break

    outfile1=all_total_reward
    outfile2=all_episode_length
    outfile3=all_train_loss
    outfile4=all_test_reward
    outfile5=all_test_episode_length

    np.save('reward_data_train_part8', outfile1)
    np.save('length_data_train_part8',outfile2)

    np.save('loss_data_train_part8', outfile3)
    np.save('length_data_test_part8', outfile4)
    np.save('reward_data_test_part8', outfile5)


    mean_episode_len = np.mean(all_episode_length, axis=0)
    mean_total_reward = np.mean(all_total_reward, axis=0)
    mean_loss_train=np.mean(all_train_loss,axis=0)
    mean_test_eposide_length=np.mean(all_test_episode_length,axis=0)
    mean_test_reward=np.mean(all_test_reward,axis=0)

    std_episode_len = np.std(all_episode_length, axis=0)
    std_total_reward = np.std(all_total_reward, axis=0)


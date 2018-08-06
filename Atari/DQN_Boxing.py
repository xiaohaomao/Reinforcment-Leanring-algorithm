### the environment Boxing ##
import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import random
import tensorflow as tf
import time
import os
env = gym.make('Boxing-v0')
### saved trained model ###
def save_final_model(model):
    if not os.path.exists('./Boxing_a_model/'):
        os.mkdir('./Boxing_a_model/')
    saver = tf.train.Saver()
    saver.save(model, './Boxing_a_model/model.checkpoint1')

### the function to chnage image from RGB to greyscale ###
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

### transfer input to size(28*28) ###
def tran_size(x, x_size, y_size):
    output = rgb2gray(x)
    output = scipy.misc.imresize(output, size=[x_size, y_size])
    return output

### stack four frames to size (28*28*4) ###
def stack(x, index, x_size, y_size):
    output = np.reshape([x[index - 4], x[index - 3], x[index - 2], x[index - 1]], [x_size, y_size, 4])
    output = np.reshape(output, [-1, 4 * x_size * y_size])
    return output

### define the weights and bias in the CNN neural network ###
def weight_variable(shape):
    output = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(output)

def bias_variable(shape):
    output = tf.constant(0.1, shape=shape)
    return tf.Variable(output)

### the convolution function API ###
def conv2d(input, Weight, strides):
    return tf.nn.conv2d(input, Weight, strides, padding='SAME')

###  the cnn approximator  ###
def cnn_approximator(x, weight_convol_1, bias_convol_1, weight_convol_2, bias_convol_2,
                     weight_flat, bias_flat, weight_out, bias_out,flat_width,flat_length,keep_prob):

    output_convol_1 = tf.nn.relu(conv2d(input=x, Weight=weight_convol_1, strides=[1, 2, 2, 1]) + bias_convol_1)
    output_convol_2 = tf.nn.relu(
        conv2d(input=output_convol_1, Weight=weight_convol_2, strides=[1, 2, 2, 1]) + bias_convol_2)

    output_reshape = tf.reshape(output_convol_2, [-1, flat_width* flat_length * 32])
    output_flat = tf.matmul(output_reshape, weight_flat) + bias_flat

    out_drop=tf.nn.dropout(output_flat,keep_prob=keep_prob)
    weight_out=weight_variable([256,action_space])
    bias_out=bias_variable([action_space])
    y = tf.matmul(output_flat, weight_out) + bias_out
    print('====the cnn approximator is running====')
    return y

### set hyperparameter and variables  for the CNN approximator ###
discount = 0.99
learn_rate = 0.001
eplison = 0.1
action_space = 18
width_size = 64
length_size = 64
flat_layer_width=int(width_size/4)
flat_layer_length=int(length_size/4)

###  the Q_value weights ###
## first layer ##
weight_convol_1 = weight_variable([6, 6, 4, 16])
bias_convol_1 = bias_variable([16])
## second layer ##
weight_convol_2 = weight_variable([4, 4, 16, 32])
bias_convol_2 = bias_variable([32])
## flat layer ##
weight_flat = weight_variable([flat_layer_width * flat_layer_length * 32, 256])
bias_flat = bias_variable([256])
## linear layer ##
weight_out = weight_variable([256, action_space])
bias_out = bias_variable([action_space])

### the target Q value weights placeholder ###

Weight_convol_1_target= tf.placeholder(tf.float32,shape=[6, 6, 4, 16])
Bias_convol_1_target = tf.placeholder(tf.float32,shape=[16])
Weight_convol_2_target = tf.placeholder(tf.float32,shape=[4, 4, 16, 32])
Bias_convol_2_target = tf.placeholder(tf.float32,shape=[32])
Weight_flat_target = tf.placeholder(tf.float32,[flat_layer_width * flat_layer_length * 32, 256])
Bias_flat_target = tf.placeholder(tf.float32,shape=[256])
Weight_out_target = tf.placeholder(tf.float32,shape=[256, action_space])
Bias_out_target = tf.placeholder(tf.float32,shape=[action_space])


### the  Input Placeholder ###
x1 = tf.placeholder(tf.float32, shape=[None, width_size * length_size * 4])
x2 = tf.placeholder(tf.float32, shape=[None, width_size * length_size * 4])
x3 = tf.placeholder(tf.float32, shape=[None, 1])
x4 = tf.placeholder(tf.int32, shape=[None, 1])
x5 = tf.placeholder(tf.int32, shape=[None, 1])
## dropout ratio ##
keep_prob = tf.placeholder(tf.float32)

### reshape the stacked pictures before into cnn model ###
x_1_image = tf.reshape(x1, [-1, width_size, length_size,4])
x_2_image = tf.reshape(x2, [-1, width_size, length_size,4])

### caucalate the q avlue and max _next value
prediction_now = cnn_approximator(x_1_image, weight_convol_1, bias_convol_1, weight_convol_2, bias_convol_2,
                                  weight_flat, bias_flat, weight_out,
                                  bias_out,flat_layer_width,flat_layer_length,keep_prob)
prediction_next = cnn_approximator(x_2_image, Weight_convol_1_target, Bias_convol_1_target,
                                   Weight_convol_2_target, Bias_convol_2_target,
                                   Weight_flat_target, Bias_flat_target, Weight_out_target,
                                   Bias_out_target,flat_layer_width,flat_layer_length,keep_prob)

### test action when test agent performance ###
test_action = tf.cast(tf.argmax(prediction_now, 1), tf.int32)

### take Q value underlying the actual action ###
True_action = tf.cast(x4, tf.int32)
True_action = tf.reshape(True_action, shape=[-1, 1])
action_repeat = tf.reshape(tf.cast(x5, tf.int32), shape=[-1, 1])
action_double = tf.concat([action_repeat, True_action], 1)

### calcaulate the loss and training ###
Q_value = tf.gather_nd(params=prediction_now, indices=action_double)
Max_Q_value_next = tf.reduce_max(prediction_next, axis=1)
print('......the reward is clipped .....')
## when the game break, just use reward as the Q target approximation ##
delta=tf.add(x3 + discount * tf.stop_gradient((1+x3)*Max_Q_value_next), (-1 * Q_value))
q_loss = tf.reduce_mean(tf.square(delta) / 2)
train_optimizer_Boxing = tf.train.RMSPropOptimizer(learn_rate).minimize((q_loss))


with tf.device('/cpu:0'):
    with tf.Session() as sess:
        start_time = time.time()
        print('======== build the experience replay ==========')
        ### set the variable and empty set ###
        length = []
        total_score = []
        experience_size = 100000 * 3
        number_episode_buffer = 200
        ### the buffer experience replay ###
        tran_size_buffer = []
        start_time = time.time()
        experience_buffer = []

        for i_buffer in range(number_episode_buffer):
            observation_0 = env.reset()
            if (i_buffer+1) % 20 == 0:
                print('......the {} th episodes and  information '
                      'of observation_initial {}.....'.format(i_buffer+1,np.shape(observation_0)))

            Score = []
            Action = []
            for i_step in range(experience_size):
                ### collect data ###
                action = env.action_space.sample()
                # print('the action',action)
                observation_0, score, done, _ = env.step(action)
                observation_1 = tran_size(observation_0, width_size, length_size)
                # print('score,done._', score, done, _)
                Score.append(int(score))
                Action.append(action)
                tran_size_buffer.append(observation_1)

                if (i_step + 1) % 4 == 0 and i_step >= 7:
                    # print('......the step is {} the size of tran_size_buffer is {}......'.format(i_step + 1,np.shape(tran_size_buffer)))
                    sub_example = []
                    sub_example.append(stack(tran_size_buffer, i_step + 1 - 4, width_size, length_size))
                    sub_example.append(stack(tran_size_buffer, i_step + 1, width_size, length_size))
                    ## clip the reward betweewn [-1 0 1] ##
                    ## the one type of state is [state_i,state_{i+1},reward,action] ##
                    if sum(Score[i_step + 1 - 8:i_step + 1 - 4]) == 0:
                        sub_example.append(0)
                    elif sum(Score[i_step + 1 - 8:i_step + 1 - 4]) > 0:
                        sub_example.append(1)
                    else:
                        sub_example.append(-1)
                    sub_example.append(Action[i_step - 4])
                    experience_buffer.append(sub_example)

                if done is True:
                    final_example = []
                    final_example.append(stack(tran_size_buffer, i_step + 1 - 4, width_size, length_size))
                    final_example.append(stack(tran_size_buffer, i_step + 1, width_size, length_size))
                    final_example.append(-1)
                    final_example.append(Action[i_step - 4])
                    total_score.append(np.sum(Score, axis=0))
                    length.append(i_step + 1)
                    experience_buffer.append(final_example)
                    break
        tran_size_buffer = []
        print('the information of generated experience buffer ', np.shape(experience_buffer), type(experience_buffer))
        print('length of each episode', length)
        print('total score of each episode', total_score)
        print('==================the experience buffer process done ==============')
        print('==================the total generated time is {}=================== (分)'.format(
            (time.time() - start_time) / 60))

        print('######################################## starting training the DQN algorithm '
              '###################################')
        ### training the  DQN algorithm ###
        sess.run(tf.global_variables_initializer())
        episode_number_training = 1000
        training_step = 1000000
        batch_size = 32
        total_time = 0
        total_training_step = 0
        total_number_data = 100000
        total_test_score = []
        total_computer_test_score = []
        total_training_loss=[]
        print('....... the number of data points in experience buffer ........', np.shape(experience_buffer))
        print('....... the number of data points in experience buffer ........',
              np.shape(experience_buffer[-1 * total_number_data:]))
        for i_episode in range(1, episode_number_training):
            ### hold the old weights for target calculation in each 5 training episodes ###
            if ((i_episode - 1) % 5 == 0):
                weight_convol_1_target, bias_convol_1_target, weight_convol_2_target, bias_convol_2_target, \
                weight_flat_target, bias_flat_target, weight_out_target, bias_out_target = \
                    sess.run([weight_convol_1, bias_convol_1, weight_convol_2, bias_convol_2,
                              weight_flat, bias_flat, weight_out, bias_out])
                print('-----------the target parameters updated----------')
                print('========='
                      '================================================'
                      '{} th episode training is starting ================'
                      '=========================================='.format(i_episode))

            start_time = time.time()
            ## reset the environment at the beginning of each episode ##
            env.reset()
            ## the list to store the updating experience ##
            update_Score = []
            update_Action = []
            update_transition = []
            action = env.action_space.sample()
            each_episode_loss=0
            for i_training in range(training_step):
                # env.render()
                observation_1, reward, done, info = env.step(action)
                update_Action.append(action)
                update_Score.append(reward)
                update_transition.append(tran_size(observation_1, width_size, length_size))
                ## For consistence, we use the same action at 4 consecutive steps ##
                if i_training >= 7 and (i_training + 1) % 4 == 0:
                    ## update the experience ##
                    update_Action, update_Score, update_transition = update_Action[-8:], \
                                                                     update_Score[-8:],\
                                                                     update_transition[-8:]
                    experience_buffer = experience_buffer.tolist()
                    if sum(update_Score[4:]) == 0:
                        added_sum_score = 0
                    elif sum(update_Score[4:]) > 0:
                        added_sum_score = 1
                    else:
                        added_sum_score = -1
                    experience_buffer.append([stack(update_transition, 4, width_size, length_size),
                                              stack(update_transition, 8, width_size, length_size),
                                              added_sum_score, update_Action[4]])
                    ## keep the fixed size of experience replay ##
                    experience_buffer = experience_buffer[-1 * total_number_data:]
                    experience_buffer = np.array(experience_buffer)

                    ## take the randm action or greedy action ##
                    if np.random.random() <= eplison:
                        action = np.random.randint(0, 18)
                    else:
                        action = int(
                            sess.run(test_action, feed_dict={x1: stack(update_transition, 8, width_size, length_size),
                                                             keep_prob: 0.8,
                                                             Weight_convol_1_target: weight_convol_1_target,
                                                             Bias_convol_1_target: bias_convol_1_target,
                                                             Weight_convol_2_target: weight_convol_2_target,
                                                             Bias_convol_2_target: bias_convol_2_target,
                                                             Weight_flat_target: weight_flat_target,
                                                             Bias_flat_target: bias_flat_target,
                                                             Weight_out_target: weight_out_target,
                                                             Bias_out_target: bias_out_target})[0])


                batch_sample = np.reshape(random.sample(range(0, len(experience_buffer)), batch_size), [-1, 1])
                # print('=====the batch-sample====', batch_sample)
                experience_buffer = np.array(experience_buffer)
                mini_sample = np.reshape(experience_buffer[batch_sample], [batch_size, -1])

                Input_1 = np.concatenate(mini_sample[:, 0], axis=0)
                Input_2 = np.concatenate(mini_sample[:, 1], axis=0)
                Input_3 = np.reshape(mini_sample[:, 2], [-1, 1])
                Input_4 = np.reshape(mini_sample[:, 3], [-1, 1])
                Input_5 = np.reshape(np.arange(batch_size), [-1, 1])

                ## running the training step ##
                _, loss = sess.run([train_optimizer_Boxing, q_loss],
                                   feed_dict={x1: Input_1, x2: Input_2, x3: Input_3, x4: Input_4,
                                              x5: Input_5, keep_prob: 0.8,
                                              Weight_convol_1_target: weight_convol_1_target,
                                              Bias_convol_1_target: bias_convol_1_target,
                                              Weight_convol_2_target: weight_convol_2_target,
                                              Bias_convol_2_target: bias_convol_2_target,
                                              Weight_flat_target: weight_flat_target,
                                              Bias_flat_target: bias_flat_target,
                                              Weight_out_target: weight_out_target,
                                              Bias_out_target: bias_out_target})
                each_episode_loss+=loss

                if done is True:
                    ## record score for agent and computer each eposide ##
                    ## always set -1(reward) when episode is done ##
                    total_training_loss.append(each_episode_loss/i_training+1)
                    experience_buffer = experience_buffer.tolist()
                    experience_buffer.append([stack(update_transition, 4, width_size, length_size),
                                              stack(update_transition, 8, width_size, length_size),
                                              -1, update_Action[4]])
                    experience_buffer = np.array(experience_buffer)
                    total_time += (time.time() - start_time)
                    total_training_step += i_training
                    if (i_episode - 1) % 5 == 0:
                        print(
                            '*************** the {} th step trainning '
                            'loss is {} ***************'.format(i_training + 1, loss))
                        print(
                            '*************** the {} th episode trainning'
                            ' time is {} 分 ***************'.format(i_episode,(time.time() - start_time) / 60))

                        print('************* the total steps trainning '
                              'until now is {} **************'.format( total_training_step + 1))
                        print('************* the total trainning time '
                              'is {} 分 **************'.format(total_time / 60))
                    print('========='
                          '================================================'
                          '{} th episode is finished  ================'
                          '=========================================='.format(i_episode))

                    break
            ### test the agent performance  until now ###
            if (i_episode - 1) % 10== 0:
                print('========='
                      '================================================'
                      'After {} th training episode, the agent testing is starting ================'
                      '=========================================='.format(i_episode))
                test_episode_number = 5
                test_score = 0
                test_computer_score = 0
                test_step = 100000
                for i_test_number in range(test_episode_number):
                    test_observation_0 = env.reset()
                    test_observation_0 = tran_size(test_observation_0, width_size, length_size)
                    test_update_Action = []
                    test_update_transition = []
                    test_episode_action = env.action_space.sample()
                    for i_test_step in range(test_step):
                        # env.render()
                        test_observation_1, test_reward, test_done, test_info = env.step(test_episode_action)
                        test_update_transition.append(tran_size(test_observation_1, width_size, length_size))
                        if test_reward > 0:
                            test_score += test_reward
                        else:
                            test_computer_score += -1 * test_reward

                        if (i_test_step + 1) % 4 == 0:
                            test_update_transition = test_update_transition[-4:]
                            test_episode_action = int(sess.run(test_action, feed_dict={
                                x1: stack(test_update_transition, 4, width_size, length_size), keep_prob: 0.8}))
                        if test_done:
                            print('+++++++++++test {} th episode is done score of agent and computer  is'
                                  ' {},{} respectively until now +++++++++'.format(i_test_number, test_score,
                                                                                   test_computer_score))
                            break
                print('++++++++++++the test average agent score is {} , computer score '
                      'is {}+++++++++++++'.format(test_score / test_episode_number,
                                                  test_computer_score / test_episode_number))
                total_test_score.append(test_score / test_episode_number)
                total_computer_test_score.append(test_computer_score / test_episode_number)
                ### save the model each 10 turn training ###
                save_final_model(sess)
                print('//////saved the model///////', i_episode)

        print('the total_test_score as training',total_test_score)
        print('the total_computer_test_score as training', total_computer_test_score)
        print('the training loss',total_training_loss)
        np.save('agent_score',total_test_score)
        np.save('computer_score',total_computer_test_score)
        np.save('training_loss',total_training_loss)
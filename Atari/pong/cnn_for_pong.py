import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

import tensorflow as tf

env = gym.make('Pong-v0')


### the function to chnage image from RGB to greyscale ###
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

### transfer input to size(28*28*1) ###
def tran_size(x):
    output = scipy.misc.imresize(x, size=[28, 28])
    output = rgb2gray(output)
    return output

### stack four frames to size (28*28*4) ###
def stack(x, index):
    output = np.reshape([x[index], x[index - 1], x[index - 2], x[index - 3]], [28,28,4])
    return output

### define the weights in the CNN neural network###
def weight_variable(shape):
    output = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(output)


def bias_variable(shape):
    output = tf.constant(0.1, shape=shape)
    return tf.Variable(output)

### the convolution function ####
def conv2d(input, Weight,strides):

    return tf.nn.conv2d(input, Weight, strides, padding='SAME')

###  the cnn function ###
def cnn_pong(x):
    ### first layer ###
    weight_convol_1=weight_variable([6,6,4,16])
    bias_convol_1=bias_variable([16])

    output_convol_1=tf.nn.relu(conv2d(input=x,Weight=weight_convol_1,strides=[1,2,2,1])+bias_convol_1)
    ### second layer ###
    weight_convol_2=weight_variable([4,4,16,32])
    bias_convol_2=bias_variable([32])
    output_convol_2=tf.nn.relu(conv2d(input=output_convol_1,Weight=weight_convol_2,strides=[1,2,2,1])+bias_convol_2)
    ### flat layer ###
    weight_flat=weight_variable([7*7*32,256])
    bias_flat=bias_variable([256])
    output_reshape=tf.reshape(output_convol_2,[-1,7*7*32])
    output_flat=tf.matmul(output_reshape,weight_flat)+bias_flat

    ### linear layer ###
    out_drop=tf.nn.dropout(output_flat,0.8)
    weight_out=weight_variable([256,action_space])

    bias_out=bias_variable([action_space])

    y=tf.matmul(out_drop,weight_out)+bias_out

    return y

### set hyperparameter and variables ###
discount=0.99
learn_rate=0.001
eplison=0.1
action_space=6


keep_drop=tf.placeholder(tf.float32)
x1=tf.placeholder(tf.float32,shape=[None,28,28,4])
x2=tf.placeholder(tf.float32,shape=[None,28,28,4])
x3=tf.placeholder(tf.float32,shape=[None,1])
x4=tf.placeholder(tf.int32,shape=[None,2])

### caucalate the q avlue and max _next value
prediction_now=cnn_pong(x1)
prediction_next=cnn_pong(x2)

### test action when test agent performance ###
test_action=tf.cast(tf.argmax(prediction_now,1),tf.int32)


### calcaulate the loss and training ###
Q_value=tf.gather_nd(params=prediction_now,indices=x4)
Max_Q_value_next=tf.reduce_max(prediction_next,axis=1)

delta=tf.add(x3+discount*tf.stop_gradient(Max_Q_value_next),(-1*Q_value))
q_loss=tf.reduce_sum(tf.square(delta)/2)

train_optimizer=tf.train.RMSPropOptimizer(learn_rate).minimize((q_loss))

#### save the model ####
saver=tf.train.Saver()

with tf.device('/cpu:0'):
    with tf.Session() as sess:

        for i_run in range(1, 1 + 1):
            sess.run(tf.global_variables_initializer())

            print('......start training data......')

            ### set the variable and empty set ###
            length = []
            total_score_a = []
            total_score_b = []
            total_absolute=[]
            ### the 100 eposides ###
            eposide_number = 100

            ### the x axis value ###
            x = np.arange(eposide_number)
            x = x + 1
            ### the buffer experience replay ###
            initial_buffer = []
            buffer_replay = []
            for i_eposide in range(eposide_number):
                env.reset()
                ### record score for computer and agent ###
                Score_a = []
                Score_b = []

                for i_step in range(100000):

                    if len(initial_buffer) < 4:
                        ### collect data ###
                        action = env.action_space.sample()
                        obser_1, score, done, _ = env.step(action)
                        obser_initial = tran_size(obser_1)
                        if score < 0:
                            Score_b.append(score)
                        if score > 0:
                            Score_a.append(score)
                        #print(score, done)

                        initial_buffer.append(obser_initial)

                    else:

                        state_i = stack(initial_buffer, i_step - 1)

                        buffer_replay.append(state_i)
                        ### select action by eplison policy ###
                        if np.random.random() <= eplison:
                            action_select=np.random.randint(6)
                            #print('ewqrwqr......')
                        else:

                            action_select = sess.run(test_action, feed_dict={x1: [state_i]})

                        action_select = int(action_select)
                        obser_1, score, done, _ = env.step(action_select)
                        if score < 0:
                            Score_b.append(score)
                        if score > 0:
                            Score_a.append(score)
                        #print(score, done)

                        obser_initial = tran_size(obser_1)

                        initial_buffer.append(obser_initial)

                    if done is True:
                        ### record score for agent and computer each eposide ###
                        total_score_a.append(np.sum(Score_a, axis=0))
                        total_score_b.append(-1 * np.sum(Score_b, axis=0))
                        length.append(i_step + 1)
                        total_absolute.append((np.sum(Score_a, axis=0)+np.sum(Score_b, axis=0)))

                        break

### calculate the standard of score and frame counts ###
std_length = np.std(length, axis=0)
std_score = np.std(total_score_a, axis=0)
std_score_abso=np.std(total_absolute,axis=0)

print('the length...',length)
print('the agent score...',total_score_a)
print('the absolute value...',total_absolute)
print('the std of agent score..',std_score)
print('the std_score_abso..',std_score_abso)
print('the std_length..',std_length)
# print(std_length)
print('the mean of total_score_a...',np.mean(total_score_a, axis=0))
print('the mean of length...',np.mean(length, axis=0))
print('the mean of total_absolute...',np.mean(total_absolute, axis=0))

### plot the mean the score and length ###
plt.plot(x, total_score_a)
plt.xlabel('ith Num of episode')
plt.ylabel('agent scores')
plt.show()

plt.plot(x, total_score_b)
plt.xlabel('ith Num of episode')
plt.ylabel('computer scores')
plt.show()

plt.plot(x, total_absolute)
plt.xlabel('ith Num of episode')
plt.ylabel('difference between agent and computer')
plt.show()


plt.plot(x, length)
plt.xlabel('ith Num of episode')
plt.ylabel('agent frames count')
plt.show()


np.save('part2_pong_a_score', total_score_a)
np.save('part2_pong_b_score', total_score_b)
np.save('part2_pong_length', length)
np.save('part2_pong_difference_score',total_absolute)


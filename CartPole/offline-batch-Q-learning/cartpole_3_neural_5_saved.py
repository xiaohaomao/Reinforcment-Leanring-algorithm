import gym
import numpy as np
import tensorflow as tf
import random
import os

#import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env._max_episode_steps = 300



print("......Loading train_data......")

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
        sess.run(tf.global_variables_initializer())

        print('......start training data......')
        length_total_data=len(train_data)
        print(train_data[1])

        #print(length_total_data)

        select_order=np.arange(batch_size)
        #print(select_order)
        batch_number=5000
        test_size=20

        #epoch=5000

        #eposide_number=100
        all_eposide_length=np.zeros((1,batch_number))
        all_reward=np.zeros((1,batch_number))
        all_loss=np.zeros((1,batch_number))

        for i_batch in range(batch_number):
        #for i_eposide in range(1000000):
            batch_sample = random.sample(range(0, length_total_data), batch_size)
            insert_data=train_data[batch_sample]

            obser_now = []
            obser_next = []
            reward_now = []
            action_now = []

            for i_select in range(batch_size):
                obser_now.append(insert_data[i_select][0])
                obser_next.append(insert_data[i_select][1])

                reward_now.append(insert_data[i_select][2])
                action_now.append(insert_data[i_select][3])

            _, train_loss = sess.run([train_optimizer, q_loss],feed_dict={x1: obser_now, x2: obser_next, x3: reward_now, x4: action_now,x5:select_order
                                    })
            if i_batch % 1 == 0:

                print('...ith training....:', i_batch, 'average training loss:', train_loss/batch_size)

                eposide_length = np.zeros((1,test_size))
                expected_value = np.zeros((1,test_size))

                for i_episode in range(test_size):

                    observation_init = env.reset()
                    observation_next = [observation_init]

                    for t in range(300):
                        #env.render()

                        ### greedy policy to select action ###
                        if np.random.random() <= eplison:
                            Action = np.random.randint(2)

                        else:

                            Action = test_action.eval(feed_dict={x1: observation_next})

                        observation_curr, reward_curr, done, info = env.step(int(Action))

                        observation_next = [observation_curr]


                        if done is True:

                            eposide_length[0,i_episode]=t + 1
                            reward = -1
                            reward_return = reward * (discount ** (t))
                            expected_value[0,i_episode]=reward_return

                            break


                all_eposide_length[0,i_batch]=np.mean(np.mean(eposide_length,axis=0),axis=0)
                all_reward[0,i_batch]=np.mean(np.mean(expected_value,axis=0),axis=0)
                all_loss[0, i_batch] = train_loss/batch_size
            if i_batch>=2:
                if i_batch==np.argmax(all_eposide_length):
                    print(i_batch)
                    print(np.argmax(all_eposide_length))

                    if not os.path.exists('./part3_neural_5_300/'):
                        os.mkdir('./part3_neural_5_300/')
                    saver.save(sess, "./part3_neural_5_300/")
                    print('saved')


            print('....the averagelength of test eposide....',np.mean(np.mean(eposide_length,axis=0),axis=0))



        outfile1 = all_reward
        outfile2 = all_eposide_length
        outfile3=all_loss
        #print(outfile2)
        #print(outfile1)
        #print(outfile3)
        #print('....the all eposide_length....',all_eposide_length)

        np.save('reward_data_part3_5_neural_300', outfile1)
        np.save('length_data_part3_5_neural_300', outfile2)
        np.save('loss_data_part3_5_neural_300', outfile3)


        mean_episode_len = np.mean(all_eposide_length,axis=0)
        mean_total_reward = np.mean(all_reward,axis=0)
        mean_total_loss =np.mean(all_loss,0)











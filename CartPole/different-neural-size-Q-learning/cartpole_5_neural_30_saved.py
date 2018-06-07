import gym
import numpy as np
import tensorflow as tf

import os

env = gym.make('CartPole-v0')
env._max_episode_steps = 300



#### set variable and parameters ####
x1=tf.placeholder(tf.float32, shape=[None,4])
x2=tf.placeholder(tf.float32, shape=[None,4])
x3=tf.placeholder(tf.float32, shape=[None,1])
x4=tf.placeholder(tf.int32, shape=[None,2])


discount=0.99
learn_rate=0.0001
input_size=4
hidden_size=30
output_size=2
eplison=0.05
max_eposide_length=300

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

### the best action based on observation_now ###
test_action=tf.cast(tf.argmax(prediction_now,1),tf.int32)



### calcaulate the loss and training ###
Q_value=tf.gather_nd(params=prediction_now,indices=x4)

Max_Q_value_next=tf.reduce_max(prediction_next,axis=1)


delta=tf.add(x3+discount*tf.stop_gradient((1+x3)*Max_Q_value_next),(-1*Q_value))

q_loss=tf.reduce_sum(tf.square(delta)/2)

train_optimizer=tf.train.AdamOptimizer(learn_rate).minimize(q_loss)


#### save the model ####
saver=tf.train.Saver()



with tf.device('/cpu:0'):

#### set the set to save data ####
    run_size = 1
    all_episode_length = np.zeros((run_size, 2000))
    all_total_reward = np.zeros((run_size, 2000))
    all_test_episode_length = np.zeros((run_size, 100))
    all_test_reward = np.zeros((run_size, 100))
    all_train_loss = np.zeros((run_size, 100))


    with tf.Session() as sess:
        for i_run in range(1,1+run_size):
            sess.run(tf.global_variables_initializer())

            print('......start training data......')

            for i_eposide in range(1,2000+1):

                ### begin a new eposide ###
                observation_00 = env.reset()
                total_reward=0
                total_QQ_loss=0

                for i_step in range(max_eposide_length):


                    ### greedy policy to select action ###
                    if np.random.random() <= eplison:
                        action_select_now=np.random.randint(2)

                    else:
                    ### use Q function to select action ###
                        action_select_now=sess.run(test_action,feed_dict={x1:np.reshape(observation_00, [1, 4])})
                        action_select_now=int(action_select_now)

                    observation_11,_,done_0,info=env.step(action_select_now)

                    if done_0 is False:
                        reward=0
                    else:
                        reward=-1
                    ### training step ###
                    _,train_loss=sess.run([train_optimizer,q_loss], feed_dict={x1:np.reshape( observation_00,[1,4]), x2: np.reshape( observation_11,[1,4]), x3:np.reshape(reward,[1,1]),x4:np.reshape([0,action_select_now],[1,2])})

                    total_QQ_loss +=train_loss

                    observation_00 = observation_11
                    if (i_eposide-1)%20==0:

                        if done_0 is True:
                            reward=-1

                            final_reward =reward* discount**(i_step)

                            all_episode_length[i_run-1, i_eposide-1] = i_step + 1
                            all_total_reward[i_run-1, i_eposide-1] = final_reward



                        ### record average test performance ###
                            test_size=10
                            Small_test_eposide_length = np.zeros((1, test_size))
                            Small_test_reward = np.zeros((1, test_size))

                            for i_test_run in range(1,test_size+1):
                                observation_test_0 = env.reset()


                                for i_test_length in range(max_eposide_length):
                                    #env.render()
                                    action_test_now = test_action.eval(feed_dict={x1: np.reshape(observation_test_0, [1, 4])})
                                    action_test_now=int(action_test_now)
                                    observation_test_1, _, test_done, test_info = env.step(int(action_test_now))

                                    observation_test_0=observation_test_1

                                    if test_done is False:
                                        reward_test = 0,
                                    else:
                                        reward_test = -1

                                    if test_done is True:
                                        Small_test_eposide_length[0,i_test_run-1]=i_test_length+1
                                        Small_test_reward[0,i_test_run-1]=reward_test*(discount**(i_test_length))
                                        #print(i_test_length+1)

                                        break


                            small_mean_test_length=np.mean(np.mean(Small_test_eposide_length,axis=0),axis=0)
                            small_mean_test_reward=np.mean(np.mean(Small_test_reward,axis=0),axis=0)
                            print('ith_run', i_run-1, 'the ith eposide', i_eposide-1,
                                  'the test average length', small_mean_test_length , '..loss..',
                                  train_loss)
                            all_test_episode_length[i_run-1, int((i_eposide-1)/20)]=small_mean_test_length
                            #print((i_eposide-1)/20)
                            #print(int((i_eposide-1)/20))
                            all_test_reward[i_run-1, int((i_eposide-1)/20)]=small_mean_test_reward
                            all_train_loss[i_run-1, int((i_eposide-1)/20)] = total_QQ_loss/(i_step+1)

                            if all_test_episode_length[i_run-1, int((i_eposide - 1) / 20)] == np.amax(
                                   all_test_episode_length):

                                print('.....', all_test_episode_length[i_run-1, int((i_eposide - 1) / 20)])
                                print(np.amax(all_test_episode_length))



                                if not os.path.exists('./part5_neural_30_300/'):
                                    os.mkdir('./part5_neural_30_300/')
                                saver.save(sess, "./part5_neural_30_300/")
                                print('saved')


                            break
                    else:
                        if done_0 is True:
                            reward = -1

                            final_reward = reward * discount ** (i_step)

                            all_episode_length[i_run - 1, i_eposide-1] = i_step + 1
                            all_total_reward[i_run - 1, i_eposide-1] = final_reward
                            break




### save and plot performance during training and tes ####
outfile1=all_total_reward
outfile2=all_episode_length
outfile3=all_train_loss
outfile4=all_test_reward
outfile5=all_test_episode_length


np.save('part_5_train_reward_30_300', outfile1)
np.save('part5_train_eposide_length_30_300',outfile2)

np.save('part5_train_loss_30_300', outfile3)
np.save('part5_test_reward_30_300', outfile4)
np.save('part5_test_length_30_300', outfile5)




import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Pong-v0')

### set the variable and empty set ###
length = []
total_score_a=[]
total_score_b=[]
### the 100 eposides ###
eposide_number=100

### the x axis value ###
x=np.arange(eposide_number)
x=x+1

for i_eposide in range(eposide_number):
    env.reset()
    ### record computer and agent scors ###
    Score_a=[]
    Score_b=[]
    for i_step in range(100000):
        #env.render()
        ### random select action ###
        action=np.random.randint(6)
        _, score, done,_ = env.step(action)

        if score <0:
            Score_b.append(score)
        if score>0:
            Score_a.append(score)

        if done is True:
            total_score_a.append(np.sum(Score_a,axis=0))
            total_score_b.append(-1*np.sum(Score_b,axis=0))
            length.append(i_step + 1)
            break


### calculate the standard of score and frame counts ###
std_length=np.std(length,axis=0)
std_score=np.std(total_score_a,axis=0)

print('the length...',length)
print('the score of agent...',total_score_a)
print('the score of computer..',total_score_b)
print(std_score)
print(std_length)
print(np.mean(total_score_a,axis=0))
print(np.mean(length,axis=0))


### plot the mean the score and length ###
plt.plot(x,total_score_a)
plt.xlabel('ith Num of episode')
plt.ylabel('agent scores')
plt.show()

plt.plot(x,total_score_b)
plt.xlabel('ith Num of episode')
plt.ylabel('computer scores')
plt.show()

plt.plot(x,length)
plt.xlabel('ith Num of episode')
plt.ylabel('agent frames count')
plt.show()


np.save('part1_pong_a_score',total_score_a)
np.save('part1_pong_b_score',total_score_b)
np.save('part1_pong_length',length)











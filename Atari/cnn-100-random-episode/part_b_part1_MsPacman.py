import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MsPacman-v0')
### set the variable and empty set ###
length = []
total_score=[]
### the 100 eposides ###
eposide_number=100
### the x axis value ###
x=np.arange(eposide_number)
x=x+1

for i_eposide in range(eposide_number):
    env.reset()
    Score=[]
    for i_step in range(100000):
        ### record computer and agent scors ###

        action=np.random.randint(9)
        _, score, done,_ = env.step(action)

        Score.append(score)

        if done is True:
            total_score.append(np.sum(Score,axis=0))
            length.append(i_step + 1)

            break
std_length=np.std(length,axis=0)
std_score=np.std(total_score,axis=0)




print('the length...',length)
print('the score of agent...',total_score)
print(std_score)
print(std_length)
print(np.mean(total_score,axis=0))
print(np.mean(length,axis=0))



plt.plot(x,total_score)
plt.xlabel('ith Num of episode')
plt.ylabel('agent scores')
plt.show()


plt.plot(x,length)
plt.xlabel('ith Num of episode')
plt.ylabel('agent frames count')
plt.show()

np.save('part1_boxing_a_score',total_score)

np.save('part1_boxing_length',length)











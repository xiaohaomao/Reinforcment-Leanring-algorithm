
import numpy as np
import matplotlib.pyplot as plt


a_score=np.load('part2_pong_a_score.npy')
b_score=np.load( 'part2_pong_b_score.npy')
length=np.load('part2_pong_length.npy')
differenct=np.load('part2_pong_difference_score.npy')

eposide_number = 100

### the x axis value ###
x = np.arange(eposide_number)
x = x + 1

plt.plot(x,a_score )
plt.xlabel('ith Num of episode')
plt.ylabel('agent scores')
plt.show()

plt.plot(x, b_score)
plt.xlabel('ith Num of episode')
plt.ylabel('computer scores')
plt.show()

plt.plot(x,differenct)
plt.xlabel('ith Num of episode')
plt.ylabel('difference between agent and computer')
plt.show()


plt.plot(x, length)
plt.xlabel('ith Num of episode')
plt.ylabel('agent frames count')

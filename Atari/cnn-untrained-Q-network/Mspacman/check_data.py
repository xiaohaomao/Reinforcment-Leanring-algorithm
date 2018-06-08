
import numpy as np
import matplotlib.pyplot as plt


a_score=np.load('part2_MsPacman_a_score.npy')

length=np.load('part2_MsPacman_length.npy')


eposide_number = 100

### the x axis value ###
x = np.arange(eposide_number)
x = x + 1

plt.plot(x,a_score )
plt.xlabel('ith Num of episode')
plt.ylabel('agent scores')
plt.show()


plt.plot(x, length)
plt.xlabel('ith Num of episode')
plt.ylabel('agent frames count')
plt.show()

# 经典强化学习算法在CartPole和几个Atari游戏的实现



### OpenAI Gym  环境的安装

------

##### **假如已经安装python3.5+  可以通过以下两种方式简单安装gym环境**



```
pip install gym
```

或者：

```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

通过以上简易安装, 已经可以执行一些基本的游戏 如 Cartpole

运行以下实验, 来验证gym 安装成功

```
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    
```



当然你也可以增加要安装的环境包,  在上面第二种安装方法的最后一行代码中加入 ['环境名称‘]

```
pip install -e .['names']
```

特别的, 当 'names’=‘all'  将执行安装全部的环境, 这需要更多的依赖包如 cmake 和较新版本的pip, 由于这里我们要安装 Atari环境, 但往往个人主机里会缺少 atari_py、cmake

**因此 我们可以按照以下的步骤配置 Atari 环境 (windows)：**

```
# 更新pip
python -m pip install --upgrade pip  
# 安装atari_py
pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py

git clone https://github.com/openai/gym
cd gym

#安装 cmake
pip install cmake

pip install -e .[atari] 
```



> "更多的关于gym环境的documents：" http://gym.openai.com/docs/

### 项目实现步骤

#### Cart-Pole 游戏:

首先我们在搭建、调通一些经典的强化学习算法, 包括batch(offline) Q-learng、online Q-learning、Deep Q-Network, Double-Q-learning, 通过经典的平衡杆游戏 CartPole 测试每个算法的表现.

由于平衡杆游戏每次给的的反馈是一个 四维数组$(S_t,S_{t+1},A,R)$, 因此我们只需要用一个前馈神经网络去作为值函数近似器, 但在Atari 游戏中, 输入一个action, 游戏系统给的反馈是游戏进行中场景的图片, 因此, 我们用卷积神经网络代替前馈神经网络作为值函数近似器. 然后分别在上述的几个强化学习算法中使用.



#### Random Policy

为了方便计算 我们人为设置 discount factor 为 0.99, reward 为 -1 当步骤是一个episode 的最后一步时, 否则 reward 为 0, 设置一个episode步长上限为300 `env._max_episode_steps = 300`. PS(以上设置仅针对Cart-Pole游戏成立)

CartPole 游戏的 action只有 0或1, 首先使用随机选择action的策略熟悉游戏环境和观察平均步长 和回报。

py文件分别在文件夹three-random-episode和 hundred-random-episode 

可以通过`env.render()`打开flash 观察游戏的进程. 平均步长和回报大概分别是22、-0.81.

#### batch (offline) Q-learning

先收集2000个随机策略下的episodes 数据, 然后仅仅基于收集好的数据, 通过直接训练动作值函数 $$Q(s; a)$$ 来学习控制平衡杆, 在这里我们分别用一个 线性转换 和一个单层隐藏层(神经元数为100)的前馈神经网络来表达动作值函数, 尝试的学习率分别是$$[10^{-5},10^{-4},10^{-3},10^{-2},10^{-1},0.5,].$$总的训练模型、更新参数次数为5000，每次训练的数据量为1000; 学习率、优化器分别是 0.001 和Adam.

实验发现, 相对前馈神经网络, 训练过程中线性转换的动作值函数能更快、高校的控制平衡杆达到300步, 但极易overfitting, 相反前馈神经网络表现的学习过程表现的更稳定, 最终的学习效果也更好.

***learning rate =0.001，linear transformation***

![](learning_curve/batch_Q_learning_linear_0.001_length.png)

![](learning_curve/batch_Q_learning_linear_0.001_reward.png)



***learning_rate=0.0001,hidden layer(100) { linear transformation + ReLU }***

![](learning_curve/batch_Q_learning_neural_0.0001_length.png)







![](learning_curve/batch_Q_learning_neural_0.0001_reward.png)



#### online Q-learning

从这里开始我们将仅仅使用神经网络来近似动作值函数,

在offline-Q-learning中训练前馈神经网络用的是离线的数据, 且每次输入模型的数据量可以自由控制, 一旦训练次数偏多或每次的训练量偏大易造成overfitting, 这里我们每次输入的训练量即某个进行中的episode的最新一步的反馈, 即每次只根据一个数据的信息更新动作值函数的参数.

为了让模型更好、更快的学习, 在训练过程中, 我们采用epsilon-greedy Q-learning算法, epsilon rate=0.05, 即有0.05 的概率使用随机策略, 而在测试过程中则全部采用值函数给出的action.

根据2中的 经验 学习率、优化器分别是 0.001，Adam；其它的设置与2中的单隐藏层前馈神经网络一致. 更多的,为了防止初始化参数带来的偏差, 我们训练一百个模型, 观察平均的步长和回报

![](learning_curve/online_Q_learning_neural_0.001_length.png)



![](learning_curve/online_Q_learning_neural_0.001_reward.png)

虽然，每个episode 的平均步长为120还远远未到达目标步长300, 但从学习曲线得知, 我们的模型一直在学习如何更好的控制平衡杆, 随着训练时间的增加, 会逐渐接近设定的目标, 相比于offline Q-learning,  online Q-learning 会更稳定的收敛,  虽然消耗的训练时间更多. 

Note that with the automatic gradient computation in tensor
ow,you must apply a stop gradient operation to avoid adapting the learning target.


$$
δ=R_{t+1}+γ*tf.stop_gradident(max_{A_{t+1}}Q(S_{t+1,A_{t+1}}))-
Q(S_{t},A_t)
$$

$$
loss=0.5*δ^{2}
$$

#### Different Neural Size 

这里我们使用不同的neural size, 测试online Q-learning的性能

neural size=30

或者

neural size=1000

#### Experience Replay and Target Parameter

Deep Q-NetWork 是近些年提出的一种增强学习模型, 相比于传统的Q-learning 算法, 其增加了两个重要的机制：经验回放、目标函数参数固定.

NIPS DQN在基本的Deep Q-Learning算法的基础上使用了Experience Replay经验池. 通过将训练得到的数据储存起来然后随机采样的方法降低了数据样本的相关性, 提升了性能, 接下来, Nature DQN做了一个改进, 就是增加Target Q网络. 也就是我们在计算目标Q值时使用专门的一个目标Q网络来计算, 而不是直接使用预更新的Q网络. 

这样做的目的是为了减少目标计算与当前值的相关性.
$$
Loss=(r+γ max_{a^{'}}Q(s^{'},a^{'},w^{-})-Q(s,a,w))^2
$$
如上面的损失函数公式所示, 计算目标Q值的函数使用的参数是$w^{-}$, $相比$之下, Nips 版本DQN 的 目标Q网络是随着Q网络实时更新的, 这样会导致 目标Q值与当前的Q值相关性较大, 容易造成过度估计（over estimation）问题

 因此提出单独使用一个目标Q网络. 那么目标Q网络的参数如何来呢？还是从Q网络中来, 只不过是延迟更新. 也就是每次等训练了一段时间再将当前Q网络的参数值复制给目标Q网络.

**其中在Q-learning 中仅加入 Experience Replay效果如下:**

![](learning_curve/experience_replay_length.png)

![](learning_curve/experience_replay_reward.png)

***在 Q-learning中仅加入Target Parameter机制***

![](learning_curve/target_parameter_length.png)

![](learning_curve/target_parameter_reward.png)

#### Double Q-learning

上面的 target-parameter 中, 对于target Q值 与目前Q 值, 我们使用同一个Q网络, 只不过参数更新的频率不一样.

而在double Q-learning 里,为了减少因为目标Q值里 max Q值计算带来的计算偏差, 或者称为过度估计（over estimation）问题, 用当前的Q网络来选择动作, 用目标Q网络来计算目标Q.

![](learning_curve/DQN_PICTURE.JPG)

![](learning_curve/double_Q_learning_length.png)

![](learning_curve/double_Q_learning_reward.png)

#### Atari Game(pong、Boxing、Mspacman)：



#### Random Policy

#### 







#### Cnn+DQN


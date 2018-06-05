

## OpenAI Gym  环境的安装

###假如已经安装python3.5+  可以通过以下两种方式简单安装gym环境



```
pip install gym
```

或者：

```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

通过以上简易安装，已经可以执行一些基本的游戏 如 Cartpole，

通过cartpole 实验，来验证gym 安装成功

```
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
```





当然你也可以增加要安装的环境包 ，在上面第二种简易安装方法的最后一行代码加入 *‘[环境名称]’

```
pip install -e .['names']
```

特别的 'names=all'  将执行安装全部的环境，这需要安装更多的 依赖包如 cmake 和较新版本的pip，由于这里我们要安装 Atari环境，但往往个人主机里会缺少 atari_py、cmake

*因此 我们可以按照以下的步骤配置 Atari 环境* (windows)：

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



[http://gym.openai.com/docs/]: 	"更多关于Gym环境的documents"


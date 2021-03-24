# football

---

本文为 Ubuntu 使用 google reasearch football 单机ppo示例教程。

关于google research football 环境详见 [GitHub代码](https://github.com/google-research/football)。

## 环境配置

---

经过测试，建议使用 python3.7 搭建google research football 环境，并使用1.15gpu版本tensorflow,此外还需要安装一些依赖

```
conda create -n gf-ppo python=3.7 tensorflow-gpu=1.15.*
conda activate gf-ppo
python3 -m pip install dm-sonnet==1.*
```



除此之外还需要一些系统包，对于 openai baslines 和 google research football 分别需要进行如下配置:

### 配置google-research football

#### 1.  安装依赖

```
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip

python3 -m pip install --upgrade pip setuptools psutil
```

#### 2. 从GitHub仓库拷贝

```
git clone https://github.com/google-research/football.git
```

立即安装环境需要执行：

```
cd football
python3 -m pip install .
```

可能需要一些时间，因为这会在后台编译C++环境

同时由于需要编译C++环境，安装完成后直接修改本地代码无法直接加载到环境上，需要再次编译，因此这里先不进行编译，待修改完代码后再编译。

### 配置 openai Baslines

#### 1. 安装依赖

```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

#### 2. 从GitHub仓库拷贝

```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

#### 3a. 使用baslines自带logger设置log

在~/football/gfootball/examples/run_ppo2.py中添加语句：

```python
flags.DEFINE_string('dir', "~/log", 'Path to openai baselines log.')

```

并在

```python
  tf.Session(config=config).__enter__()
```

后添加：

```python
  logger.configure(dir=FLAGS.dir, format_strs=['stdout','log','csv','tensorboard'])
```

之后可以通过下列操作安装google research football 环境（若之前已经安装这里需要重新安装）

```
cd football
python3 -m pip install .
```


#### 3b. 自己修改baslines文件添加tensorboard

也可以将~/baselines/baselines/ppo2/model.py文件修改:

```python
# my code
from baselines import logger
```

```python
self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
# my code
    with tf.name_scope('env'):
        tf.summary.histogram('return',self.R)
```

```python
loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

# my code
self.count = 0
with tf.name_scope('loss'):
    tf.summary.scalar('policy_loss',pg_loss)
    tf.summary.scalar('value_loss',vf_loss)
    tf.summary.scalar('policy_entropy',entropy)
    tf.summary.scalar('approxkl',approxkl)
    tf.summary.scalar('clipfrac',clipfrac)
self.merged = tf.summary.merge_all()
self.writer=tf.summary.FileWriter(logger.get_dir(),sess.graph)
```

```python
if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

# my code
self.count+=1
if(self.count % 5 == 0):
    summary = self.sess.run(self.merged,td_map)
    self.writer.add_summary(summary,self.count)
```

然后在~/football/gfootball/examples/run_ppo2.py中设置路径：

```python
# my code
flags.DEFINE_string('dir', "~/log", 'Path to openai baselines log.')
```

```python
tf.Session(config=config).__enter__()
# my code
logger.configure(dir=FLAGS.dir)
```

之后重新安装环境

### 参数设置

---

除了可以直接设置的一些参数：

```python
flags.DEFINE_string('level', 'academy_empty_goal_close',
                    'Defines type of problem being solved')
flags.DEFINE_enum('state', 'extracted_stacked', ['extracted',
                                                 'extracted_stacked'],
                  'Observation to be used for training.')
flags.DEFINE_enum('reward_experiment', 'scoring',
                  ['scoring', 'scoring,checkpoints'],
                  'Reward to be used for training.')
flags.DEFINE_enum('policy', 'cnn', ['cnn', 'lstm', 'mlp', 'impala_cnn',
                                    'gfootball_impala_cnn'],
                  'Policy architecture')
flags.DEFINE_integer('num_timesteps', int(2e6),
                     'Number of timesteps to run for.')
flags.DEFINE_integer('num_envs', 8,
                     'Number of environments to run in parallel.')
flags.DEFINE_integer('nsteps', 128, 'Number of environment steps per epoch; '
                     'batch size is nsteps * nenv')
flags.DEFINE_integer('noptepochs', 4, 'Number of updates per epoch.')
flags.DEFINE_integer('nminibatches', 8,
                     'Number of minibatches to split one epoch to.')
flags.DEFINE_integer('save_interval', 100,
                     'How frequently checkpoints are saved.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('lr', 0.00008, 'Learning rate')
flags.DEFINE_float('ent_coef', 0.01, 'Entropy coeficient')
flags.DEFINE_float('gamma', 0.993, 'Discount factor')
flags.DEFINE_float('cliprange', 0.27, 'Clip range')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradient norm (clipping)')
flags.DEFINE_bool('render', False, 'If True, environment rendering is enabled.')
flags.DEFINE_bool('dump_full_episodes', True,
                  'If True, trace is dumped after every episode.')
flags.DEFINE_bool('dump_scores', False,
                  'If True, sampled traces after scoring are dumped.')
flags.DEFINE_string('load_path', None, 'Path to load initial checkpoint from.')
```

还有一些参数的设置也十分重要：

#### 1. replay保存路径

位于代码中，为  env = football_env.create_environment() 函数的参数 logdir ， 默认取为logger.get_dir()  ，但是由于代码运行到这里时，tensorflow会话还没有启动，logger路径也没有修改，建议改为某一固定路径，或者添加参数：

```python
flags.DEFINE_string('logdir', "dumps/", 'Path to load replay.')
```

并修改：

```python
def create_single_football_env(iprocess):
  """Creates gfootball environment."""
  env = football_env.create_environment(
      env_name=FLAGS.level, stacked=('stacked' in FLAGS.state),
      rewards=FLAGS.reward_experiment,
      # my code
      logdir=FLAGS.logdir,
      # my code end
      write_goal_dumps=FLAGS.dump_scores and (iprocess == 0),
      write_full_episode_dumps=FLAGS.dump_full_episodes and (iprocess == 0),
      render=FLAGS.render and (iprocess == 0),
      dump_frequency=50 if FLAGS.render and iprocess == 0 else 0)
  env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(),
                                                               str(iprocess)))
  return env
```

#### 2.replay 储存频率

即为create_single_football_env()函数中的 dump_frequency 参数，这一参数决定每多少个episode对dumps进行储存，同时在render时影响render的频率，即没经过dump_frequency个episode进行一个episode的渲染（其他时间画面冻结）

### 正式运行

---

可以直接使用

```shell
python3 -m gfootball.examples.run_ppo2 --level=academy_empty_goal_close 
```

来运行实例，或者在后面添加其他参数

也可以通过使用shell脚本

```shell
#!/bin/bash

python3 -u -m gfootball.examples.run_ppo2 \
  --level 5_vs_5 \
  --reward_experiment scoring,checkpoints \
  --policy impala_cnn \
  --cliprange 0.08 \
  --gamma 0.993 \
  --ent_coef 0.003 \
  --num_timesteps 50000000 \
  --max_grad_norm 0.64 \
  --lr 0.000343 \
  --num_envs 16 \
  --noptepochs 2 \
  --nminibatches 8 \
  --nsteps 512 \
  --dir '~/ppo_log/5_vs_5' \
  --logdir 'dumps/5_vs_5' \
  --dump_full_episodes True \
  "$@"
```

运行实例。

（通过level设置不同场景）



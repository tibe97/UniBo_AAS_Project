from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from audioop import add


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
import ipdb
import wandb
from operator import itemgetter
import tensorflow_addons as tfa



from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.networks import network

from tf_agents.policies import py_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import scripted_py_policy

from tf_agents.policies import tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import actor_policy
from tf_agents.policies import q_policy
from tf_agents.policies import greedy_policy

from tf_agents.trajectories import time_step as ts

from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Input, Conv2D, MaxPooling2D, Lambda, Flatten, BatchNormalization
from tensorflow.keras.models import Model

import tensorflow as tf
from tf_agents.networks.q_network import QNetwork


from tensorflow.python.framework import config

from agent import DQN_agent, make_Q_net, ReplayBuffer


wandb.login()

max_episode_steps = 27000
environment_name = "BreakoutNoFrameskip-v4"


tf_env_train = TFPyEnvironment(suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4]))

tf_env_eval = TFPyEnvironment(suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4]))


# Hyperparameters
max_T = 100000
step_counter = 0
eval_interval = 500
evaluation_episodes = 10
train_step = 0
batch_size = 32
optimizer_lr =2.5e-4
replay_buffer_size = 5000
tau = 0.99
add_BYOL=True
pretraining_steps = None


q_network_online = make_Q_net(tf_env_train.observation_spec(), tf_env_train.action_spec())
q_network_target = tf.keras.models.clone_model(q_network_online)
optimizer = tf.keras.optimizers.RMSprop(lr=optimizer_lr, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True)

replay_buffer = ReplayBuffer(replay_buffer_size, batch_size)
agent = DQN_agent(tf_env_train.time_step_spec(),
                    tf_env_train.action_spec(),
                    optimizer=optimizer,
                    q_network_online=q_network_online,
                    q_network_target=q_network_target,
                    replay_memory = replay_buffer,
                    target_update_period=2000,
                    gamma=0.99,
                    tau=tau,
                    train_step_counter=train_step,
                    add_BYOL=add_BYOL,
                    decay_steps=max_T,
                    pretraining_steps=pretraining_steps
                  )

# log the hyperparams on wandb
config_dict = {
    "batch_size": batch_size,
    "optim_lr": optimizer_lr,
    "replay_buffer_size": replay_buffer_size,
    "max_T": max_T,
    "eval_interval": eval_interval,
    "eval_episodes": evaluation_episodes,
    "tau": tau,
    "add_BYOL_Loss": add_BYOL,
    "pretraining_steps": pretraining_steps
}


wandb.init(project="AAS_MiniProject"+"_"+environment_name, config=config_dict)


while step_counter < max_T:
  time_step = tf_env_train.reset() # TimeStep('discount', 'obs', 'reward', 'step')
  #print("Episode: {}".format(ep))
  while not time_step.is_last() and step_counter < max_T:
    state = time_step.observation
    action = agent.action(state)
    time_step = tf_env_train.step(action) # perform action and get S_t+1
    reward = time_step.reward
    state_next = time_step.observation
    # store transition in replay buffer
    agent.replay_memory.store_transition((state, action, reward, state_next, time_step.is_last()))
    if agent.replay_memory.buffer_max_length == len(agent.replay_memory.buffer_states):
      loss, Q_loss, BYOL_loss, mean_std_p, mean_std_z = agent.learn()
      wandb.log({'train_loss': loss}, step=step_counter)
      wandb.log({'train_Q_loss': Q_loss}, step=step_counter)
      wandb.log({'train_BYOL_loss': BYOL_loss}, step=step_counter)
      wandb.log({'mean_std_p': mean_std_p}, step=step_counter)
      wandb.log({'mean_std_z': mean_std_z}, step=step_counter)

    agent.update_step()
    step_counter += 1


    if step_counter % eval_interval == 0: # eval agent on 100 episodes
      rewards = []
      for j in range(evaluation_episodes):
        ep_reward = 0.0
        time_step = tf_env_eval.reset()
        while not time_step.is_last():
          state = time_step.observation
          action = agent.action(state, train=False) # use eval epsilon of 0.05
          time_step = tf_env_eval.step(action) # perform action and get S_t+1
          ep_reward += time_step.reward
        rewards.append(ep_reward)
      print("Evaluation at Step: {},         Mean Reward: {},         Eval Episodes: {}".format(step_counter, np.mean(rewards), evaluation_episodes))
      wandb.log({'mean_reward': np.mean(rewards)}, step=step_counter)
      
    

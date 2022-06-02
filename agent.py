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





class DQN_agent:
  
  def __init__(self, time_step_spec, action_spec, optimizer, 
               q_network_online = None,
               q_network_target = None,
               replay_memory = None,
               target_update_period=2000, 
               gamma=0.99,
               train_step_counter=0,
               eps_start=1.0,
               eps_end=0.01,
               eps_eval=0.05,
               tau=0.99,
               decay_steps=100000,
               add_BYOL=True,
               pretraining_steps=None,
               use_next_state=False):
    self._time_step_spec = time_step_spec
    self._action_spec = action_spec
    self.optimizer = optimizer
    self.q_network_online = q_network_online
    self.q_network_target = q_network_target
    self.target_update_period = target_update_period
    self.gamma = gamma
    self.replay_memory = replay_memory
    self.train_step_counter = train_step_counter
    self.eps_start = eps_start
    self.eps_end = eps_end
    self.eps_eval = eps_eval
    self.decay_steps = decay_steps
    self.tau = tau
    self.add_BYOL = add_BYOL
    self.pretraining_steps = pretraining_steps
    self.use_next_state = use_next_state
 

  def reset(self):
    # return initial_policy_state.
    self.replay_memory.reset()
    self.eps = self.eps_start
    self.train_step_counter = 0

 
  def action(self, observation, train=True):
    # return an action to perform
    action = None
    eps = self.decayed_epsilon()
    if not train:
      eps = self.eps_eval

    if random.random() > eps:
      action_values, _, _ = self.q_network_online(observation)
      action = tf.argmax(action_values[0])
    else:
      action = tf.convert_to_tensor(random.randint(0, self._action_spec.maximum), dtype=tf.int64)
    
    return action
  
  def decayed_epsilon(self):
    step = min(self.train_step_counter, self.decay_steps)
    return ((self.eps_start - self.eps_end) *
            (1 - step / self.decay_steps)**2.0
          ) + self.eps_end

  
  def update_step(self):
    self.train_step_counter += 1

  def loss_BYOL_temporal(self, x, x_next):
    '''
    Cosine similarity between projection and prediction from the network.
    Symmetric loss since we compute it for both augmentations
    '''

    # Compute the projections and the predictions from the augmented views
    _, z1_online, p1_online = self.q_network_online(x) # projection of view 1
    _, z2_online, p2_online = self.q_network_online(x_next) # projection of view 2

    _, z1_target, p1_target = self.q_network_target(x) # projection of view 1
    _, z2_target, p2_target = self.q_network_target(x_next) # projection of view 2

    loss1 = 2 - 2 * tf.einsum('ik, ik -> i', tf.math.l2_normalize(p1_online, axis=1), tf.math.l2_normalize(z2_target, axis=1)) 
    loss2 = 2 - 2 * tf.einsum('ik, ik -> i', tf.math.l2_normalize(p2_online, axis=1), tf.math.l2_normalize(z1_target, axis=1)) 

    mean_std_p = tf.math.reduce_mean(tf.math.reduce_std(tf.math.l2_normalize(p1_online, axis=1), axis=1))
    mean_std_z = tf.math.reduce_mean(tf.math.reduce_std(tf.math.l2_normalize(z1_target, axis=1), axis=1))

    loss = tf.math.reduce_mean(loss1 + loss2)
    return loss, mean_std_p, mean_std_z
  
  def loss_BYOL(self, x):
    '''
    Cosine similarity between projection and prediction from the network.
    Symmetric loss since we compute it for both augmentations
    '''
    # Image augmentations
    img_size = self._time_step_spec.observation.shape[0]
    random_data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomCrop(70, 70),
      tf.keras.layers.Resizing(img_size, img_size),
      tf.keras.layers.RandomContrast(0.3),
      tf.keras.layers.RandomFlip(mode='horizontal')   
    ])
    x1 = random_data_augmentation(x)
    x1 = tf.image.random_brightness(x1, max_delta=0.3)
    x1 = tfa.image.gaussian_filter2d(x1, filter_shape=(5, 5), sigma=(1.0, 2.0)) # augmented view 1
    x2 = random_data_augmentation(x)
    x2 = tf.image.random_brightness(x2, max_delta=0.3)
    x2 = tfa.image.gaussian_filter2d(x2, filter_shape=(5, 5), sigma=(1.0, 2.0)) # augmented view 2

    # Compute the projections and the predictions from the augmented views
    _, z1_online, p1_online = self.q_network_online(x1) # projection of view 1
    _, z2_online, p2_online = self.q_network_online(x2) # projection of view 2

    _, z1_target, p1_target = self.q_network_target(x1) # projection of view 1
    _, z2_target, p2_target = self.q_network_target(x2) # projection of view 2

    
    loss1 = 2 - 2 * tf.einsum('ik, ik -> i', tf.math.l2_normalize(p1_online, axis=1), tf.math.l2_normalize(z2_target, axis=1)) 
    loss2 = 2 - 2 * tf.einsum('ik, ik -> i', tf.math.l2_normalize(p2_online, axis=1), tf.math.l2_normalize(z1_target, axis=1))

    mean_std_p = tf.math.reduce_mean(tf.math.reduce_std(tf.math.l2_normalize(p1_online, axis=1), axis=1))
    mean_std_z = tf.math.reduce_mean(tf.math.reduce_std(tf.math.l2_normalize(z1_target, axis=1), axis=1))
   
    loss = tf.math.reduce_mean(loss1 + loss2)
    return loss, mean_std_p, mean_std_z

  
  
  def learn(self):
    states, actions, rewards, states_next, is_lasts = self.replay_memory.sample()
    states = tf.squeeze(tf.convert_to_tensor(states))
    actions =  tf.convert_to_tensor(actions)
    rewards =  tf.convert_to_tensor(rewards)
    states_next =  tf.squeeze(tf.convert_to_tensor(states_next))
    is_lasts =  tf.convert_to_tensor(is_lasts)
    
    Q_values_indices = tf.transpose([tf.range(len(actions), dtype=tf.int64), tf.squeeze(actions)]) # we use it to select the Q values of the states at timestep s
    
    # compute Q_states with a single tensor of batch size
    with tf.GradientTape() as t:
        action_next_indices = tf.transpose([tf.range(len(actions), dtype=tf.int64), tf.argmax(self.q_network_online(states_next)[0], axis=1)]) # select max actions according to online net and estimate with target net
        Q_s_next = tf.gather_nd(tf.squeeze(self.q_network_target(states_next)[0]), action_next_indices)
        y = tf.squeeze((rewards + tf.where(is_lasts, 0.0, 1.0) * Q_s_next)) # double DQN-> use q_net_target instead of online
        Q_s = tf.gather_nd(tf.squeeze(self.q_network_online(states)[0]), Q_values_indices)
        Q_loss = tf.math.reduce_mean((y - Q_s)**2)
        loss = Q_loss 
        #loss = 0.0
        if self.add_BYOL:
          if self.use_next_state: # BYOL_Temporal
            loss_BYOL, mean_std_p, mean_std_z = self.loss_BYOL_temporal(states, states_next) # we use STATES but also STATES_NEXT could be used
          else:
            loss_BYOL, mean_std_p, mean_std_z = self.loss_BYOL(states)
          loss += loss_BYOL

    # perform optimization step
    vars = self.q_network_online.trainable_variables
    grads = t.gradient(loss, vars)
    self.optimizer.apply_gradients(zip(grads, vars))

    
    # print training stats  
    if self.train_step_counter % 1000 == 0:
        if self.add_BYOL:
            print("Step: {},         Total Loss: {},         Q_Loss: {},         BYOL_Loss: {},         Eps: {}".format(self.optimizer.iterations.numpy(), 
                                                                        loss.numpy(), 
                                                                        Q_loss.numpy(),
                                                                        loss_BYOL.numpy(),
                                                                        self.decayed_epsilon()))   
        else: 
            print("Step: {},         Total Loss: {},         Q_Loss: {},         Eps: {}".format(self.optimizer.iterations.numpy(), 
                                                                        loss.numpy(), 
                                                                        Q_loss.numpy(),                                                    
                                                                        self.decayed_epsilon())) 

    del t
    self.momentum_update()

    if self.add_BYOL:
        return loss.numpy(), Q_loss.numpy(), loss_BYOL.numpy(), mean_std_p, mean_std_z
    return loss.numpy(), Q_loss.numpy(), Q_loss.numpy()
  

  def momentum_update(self):
    # Update target network weights as exponential moving average of online network
    for i, _ in enumerate(self.q_network_online.trainable_weights):
      self.q_network_target.trainable_weights[i].assign( self.q_network_target.trainable_weights[i]* self.tau + (1.0-self.tau) * self.q_network_online.trainable_weights[i] )
  

# MY REPLAY BUFFER
# Implement as list of tuples (easiest way)
class ReplayBuffer:
  def __init__(self, buffer_max_length, batch_size):
    if buffer_max_length < batch_size:
      raise Exception("The buffer size can't be smaller than the batch size")

    self.buffer_max_length = buffer_max_length
    self.buffer_states = []
    self.buffer_states_next = []
    self.buffer_actions = []
    self.buffer_rewards = []
    self.buffer_is_last = []
    self.batch_size = batch_size

  def reset(self):
    self.buffer_states = []
    self.buffer_states_next = []
    self.buffer_actions = []
    self.buffer_rewards = []
    self.buffer_is_last = []
  
  
  def store_transition(self, transition):
    # transition is a named tuple of (S_t, a_t, r_t, S_t+1)
    state, action, reward, state_next, is_last = transition
    if len(self.buffer_states) >= self.buffer_max_length:
      self.buffer_states.pop(0)
      self.buffer_states_next.pop(0)
      self.buffer_actions.pop(0)
      self.buffer_rewards.pop(0)
      self.buffer_is_last.pop(0)
  
    self.buffer_states.append(state)
    self.buffer_states_next.append(state_next)
    self.buffer_actions.append(action)
    self.buffer_rewards.append(reward)
    self.buffer_is_last.append(is_last)

  def sample(self):
    # generate random indices of experiences to sample
    indices = np.random.randint(0, len(self.buffer_states), size=self.batch_size)
    states = list(itemgetter(*indices)(self.buffer_states))
    states_next = list(itemgetter(*indices)(self.buffer_states_next))
    actions = list(itemgetter(*indices)(self.buffer_actions))
    rewards = list(itemgetter(*indices)(self.buffer_rewards))
    is_lasts = list(itemgetter(*indices)(self.buffer_is_last))

    return states, actions, rewards, states_next, is_lasts





def make_Q_net(observation_spec,
              action_spec):
  input = Input(shape=observation_spec.shape)

  preproc_layer = Lambda(lambda obs: tf.cast(obs, np.float32)/255.)(input)
  conv1 = Conv2D(filters=32, kernel_size=(8,8), strides=4, activation='relu')(preproc_layer)
  conv2 = Conv2D(filters=64, kernel_size=(4,4), strides=2, activation='relu')(conv1)
  conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu')(conv2)
 
  embedding = Flatten()(conv3)
  embedding = Dense(512, activation='relu')(embedding)

  projection = Dense(256)(embedding)
  prediction = Dense(256)(embedding)

  output = Dense(action_spec.maximum+1)(embedding)
  model = Model(inputs=[input], outputs=[output, projection, prediction])
  return model
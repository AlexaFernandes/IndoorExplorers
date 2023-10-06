import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #supress tensorflow messages

import gym #OpenAI Gym
#import retro #Gym Retro
import argparse
import math
import numpy as np #NumPy
from time import time #calculate runtime
from collections import deque #needed for replay memory
from random import sample #used to get random minibacth
from colorama import Fore, Back, Style

#TensorFlow 2.0
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Lambda, Add
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution() #Disable Eager, IMPORTANT!

from learning.dddqn_wrappers import * #My wrappers for the Gym Retro Environment
from learning.dddqn_utils import convert_frames, Now, get_latest_file #My utilities, useful re-usable functions 
#from indoor_explorers.envs.settings import DEFAULT_CONFIG as conf
#from indoor_explorers.envs.explorerConf import ExplorerConf
from multi_agent.settings import DEFAULT_CONFIG as conf
from multi_agent.indoor_explorers import IndoorExplorers
from multi_agent.utils.multi_printMaps import *

class DDDQNAgent(object):
    def __init__(self, game, combos, time_limit=None, batch_size=32, learn_every=10, update_every=10000,
                 alpha=1e-4, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9999, memory_size=100000):
        #retro environment
        self.game = game #game rom name
        self.combos = combos #valid discrete button combinations
        self.env = self.build_env(time_limit=time_limit) #retro environment
        self.num_actions = len(combos) #number of possible actions for env
        self.state_shape = self.env.observation_space[0].shape #env state dims
        # self.state_shape = tf.reshape(self.env.observation_space, [self.state_shape[0],self.state_shape[1],1])
        # self.state_shape = self.state_shape.expand_dims()
        self.state = self.reset() #initialize state

        #training
        self.batch_size=batch_size #batch size
        self.steps = 0 #number of steps ran
        self.learn_every = learn_every #interval of steps to fit model
        self.update_every = update_every #interval of steps to update target model
        self.alpha = alpha #learning rate
        self.gamma = gamma #discount factor
        self.epsilon = epsilon #exploration factor
        self.epsilon_min = epsilon_min #minimum exploration probability
        self.epsilon_decay = epsilon_decay #exponential decay rate of epsilon
        #memory
        self.memory_size = memory_size #replay memory size
        self.memory = deque(maxlen=memory_size) #replay memory
        self.log = [] #stores information from training
        #models
        self.q_eval = self.build_network() #Q eval model
        self.q_target = self.build_network() #Q target model
        
    def build_env(self, time_limit=None, downsampleRatio=2, numStack=4):
    #Build the gym environment.
        #env = retro.make(game=self.game, state=retro.State.DEFAULT, scenario='scenario',
        #                 record=False, obs_type=retro.Observations.IMAGE)
        env_name = "multi_agent:{}-v0".format(self.game)
        env = gym.make(env_name, conf=conf) #'multi_agent:IndoorExplorers21x21-v0'
        #env = ExplorerConfs()  
        #env = Discretizer(env, combos=self.combos)
        if time_limit is not None: env = TimeLimit(env, time_limit)
        #env = SkipFrames(env)
        # env = Rgb2Gray(env)
        # env = Downsample(env, downsampleRatio)
        # env = FrameStack(env, numStack)
        # env = ScaledFloatFrame(env)
        return env

    

    def build_network(self):
    #Build the Dueling DQN Network
        X_input = Input(self.state_shape, name='input')
        X = Conv2D(32, 8, 1, activation='relu')(X_input)
        X = Conv2D(64, 4, 2, activation='relu')(X)
        X = Conv2D(64, 3, 4, activation='relu')(X)
        X = Flatten()(X)
        X = Dense(1024, activation='relu', kernel_initializer='he_uniform')(X)
        X = Dense(512, activation='relu', kernel_initializer='he_uniform')(X)
        #value layer
        V = Dense(1, activation='linear', name='V')(X) #V(S)
        #advantage layer
        A = Dense(self.num_actions, activation='linear', name='Ai')(X) #A(s,a)
        A = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), name='Ao')(A) #A(s,a)
        #Q layer (V + A)
        Q = Add(name='Q')([V, A]) #Q(s,a)
        Q_model = Model(inputs=[X_input], outputs=[Q], name='qvalue')
        Q_model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return Q_model
    
    def update_target(self):
    #Update the q_target model to weights of q_eval model.
        #don't learn until we have at least one batch and update interval steps reached
        if len(self.memory) < self.batch_size or self.steps % self.update_every != 0: return
        self.q_target.set_weights(self.q_eval.get_weights())
    
    def remember(self, action, next_state, reward, done):
    #Store data in memory and update current state.
        #print("remember self.state.shape {}, NDIM {}".format(self.state.shape, self.state.ndim))
        if self.state.ndim > 4: #this happens when we do reset, and we only need the info from agent 0 to remember
            self.memory.append([self.state[0], action, next_state, reward, done])
            #print(" WRONG DIMENSION self.state[0].shape used {}".format(self.state[0].shape))
        else: #em principio nunca entrará aqui
            print("Não era suposto!")
            self.memory.append([self.state, action, next_state, reward, done])
        #self.state = next_state

    def choose_action(self, training, agent_i):
    #Predict next action based on current state and decay epsilon.
        if training: #when training allow random exploration
            if np.random.random() < self.epsilon: #get random action
                action = np.random.randint(self.num_actions)
                #print("random action {}".format(action))
            else: #predict best actions
                action = np.argmax(self.q_eval.predict(self.state[agent_i])[0])
            #decay epsilon, if epsilon falls below min then set to min
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            elif self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
                print('Epsilon mininum of {} reached'.format(self.epsilon_min))
        else: #if not training then always get best action
            action = np.argmax(self.q_eval.predict(self.state[agent_i])[0])
            print("argmax {}".format(action))
        return action
                
    def learn(self):
        #don't learn until we have at least one batch and learn interval steps reached
        if len(self.memory) < self.batch_size or self.steps % self.learn_every != 0: return
        #sample memory for a minibatch
        mini_batch = sample(self.memory, self.batch_size)
        # for el in mini_batch:
        #     print(el[0].shape)
        #     print(el[1])
        #     print(el[2].shape)
        #     print(el[3])
        #     print(el[4])
        #separate minibatch into elements
        # for i in zip(*mini_batch):
        #     state, action, next_state, reward, done = np.squeeze(i)
        state, action, next_state, reward, done = [np.squeeze(i) for i in zip(*mini_batch)]
        state = np.expand_dims(state, axis=3)
        next_state = np.expand_dims(next_state, axis=3)
        Q = self.q_eval.predict(state, batch_size = self.batch_size) #get Q values for starting states
        Q_next = self.q_eval.predict(next_state) #get Q values for ending states
        Q_target = self.q_target.predict(next_state) #get Q values from target model
        #update q values
        for i in range(self.batch_size):
            if done[i]:
                Q[i][action[i]] = 0.0 #terminal state
            else:
                a = np.argmax(Q_next[i]) ## a'_max = argmax(Q(s',a'))
                Q[i][action[i]] = reward[i] + self.gamma * Q_target[i][a] #Q_max = Q_target(s',a'_max)
        #fit network on batch_size = minibatch_size
        self.q_eval.fit(state, Q, batch_size=self.batch_size, verbose=0, shuffle=False)
    
    def load(self, directory, qeval_name=None, qtarget_name=None):
    #Load the actor and critic weights.
        print('Loading models ...', end=' ')
        #if no names supplied try to load most recent
        if qeval_name is not None and qtarget_name is not None:
            qeval_path = os.path.join(directory, qeval_name)
            qtarget_path = os.path.join(directory, qtarget_name)
        elif qeval_name is None and qtarget_name is None:
            qeval_path = get_latest_file(directory + '/*QEval.h5')
            qtarget_path = get_latest_file(directory + '/*QTarget.h5')
        self.q_eval.load_weights(qeval_path)
        self.q_target.load_weights(qtarget_path)
        print('Done. Models loaded from {}'.format(directory))
        print('Loaded Q_Eval model {}'.format(qeval_path))
        print('Loaded Q_Target model {}'.format(qtarget_path))

    def save(self, directory, fileName):
    #Save the actor and critic weights.
        print('Saving models ...', end=' ')
        if not os.path.exists(directory): os.makedirs(directory)
        qeval_name = fileName + '_QEval.h5'
        qtarget_name = fileName + '_QTarget.h5'
        self.q_eval.save_weights(os.path.join(directory, qeval_name))
        self.q_target.save_weights(os.path.join(directory, qtarget_name))
        print('Done. Saved to {}'.format(os.path.abspath(directory)))
        
    def save_log(self, directory, fileName, clear=False):
    #Save the information currently stored in log list.
        print('Saving log ...', end=' ')
        if not os.path.exists(directory): os.makedirs(directory)
        f = open(os.path.join(directory, fileName + '.csv'), 'w')
        for line in self.log:
            f.write(str(line)[1:-1].replace('None', '') + '\n')
        f.close()
        if clear: self.log = []
        print('Done. Saved to {}'.format(os.path.abspath(directory)))

    def reset(self):
    #Reset environment and return expanded state.
        self.state = np.expand_dims(self.env.reset(), axis=1) #old axis=0 
        #print("reset self.state.shape {}".format(self.state.shape))

    def close(self):
    #Close the environment.
        self.env.close()

    #TODO
    def step(self, action_n):
    #Run one step for given action and return data.
        observation_n, reward_n, done_n, info = self.env.step(action_n)
        observation_n = np.expand_dims(observation_n, axis=1) #old axis=0 
        return observation_n, reward_n, done_n, info
    
    def run(self, num_episodes=100, render=False, checkpoint=False, cp_render=False, cp_interval=None, otype='AVI', n_intelligent_agents = 1):
    #Run num_episodes number of episodes and train the q model after each learn_every number of steps. The target
    #model is updated every update_every number of steps. If render is true then render each episode to monitor.
    #If checkpoint is true then save model weights and log and evaluate the model and convert and save the frames
    #every cp_interval number of of episodes. The evaluation is rendered if cp_render is true.
        printSTR = 'Episode: {}/{} | Score: {:.4f} | AVG 50: {:.4f} | Elapsed Time: {} mins'
        start_time = time()
        scores = []
        action_n = [None, None, None, None]
        self.reset()
        if render: self.env.render()
        #self.state = self.state#[0] #since we only want to train agent 0, we only need agent's 0 observations
        for e in range(1, num_episodes + 1):
            score = 0
            done_n = [False for _ in range(self.env.n_agents)]
            ep_reward = 0
            LIVES = None #will store starting lives

            while True:
                #for now it will only be 1 intelligent agent (TODO adapt for variable number of intelligent agents)
                for i in range(n_intelligent_agents):
                    action_n[i] = self.choose_action(training=True, agent_i=i) #predict action
                for i in range(n_intelligent_agents, self.env.n_agents):
                    action_n[i] = np.random.randint(self.num_actions)
                
                if self.env.conf["viewer"]["print_prompts"]:
                    print("\u001b[34m {}\u001b[34m,\033[91m {}\033[00m,\u001b[32m {}\u001b[32m,\u001b[33m {}\u001b[33m" .format(ACTION_MEANING[action_n[0]], ACTION_MEANING[action_n[1]],ACTION_MEANING[action_n[2]],ACTION_MEANING[action_n[3]]))
                    print(Style.RESET_ALL)

                obs_n, reward_n, done_n, info = self.step(action_n) #perform action
                
                #for i in range(self.env.n_agents): #TODO should the score be the cumulative score of all agents or just agent 0(the intelligent one)?
                score += reward_n[0] #cumulative score for episode
                reward = np.clip(reward_n[0], -1.0, 1.0).item() #clip reward to range [-1.0, 1.0]
                #if LIVES is None: LIVES = info['lives'] #get starting lives
                #if info['lives'] < LIVES: done = True #flag for reset when dead
                self.remember(action_n[0], obs_n[0], reward, done_n[0]) #store results
                self.state = obs_n #update state 
                self.steps += 1 #increment steps
                self.update_target() #update target network (update_every)
                self.learn() #fit q model (learn_every)

                if render: self.env.render()
                if all(done_n):
                    scores.append(score) #store scores for all epsisodes
                    self.reset()
                    self.state = self.state#[0] #since we only want to train agent 0, we only need agent's 0 observations
                    break
            elapsed_time = round((time() - start_time)/60, 2)        
            print(printSTR.format(e, num_episodes, round(score, 4), np.average(scores[-50:]), elapsed_time))
            if checkpoint and (e % cp_interval) == 0:
                eval_score, frames = self.evaluate(render=cp_render)
                print('EVALUATION: {}'.format(round(eval_score, 4)))
                self.log.append([e, score, np.average(scores[-50:]), elapsed_time, eval_score])
                fileName = 'DDDQN_{}_{}_{}'.format(e, self.game, Now(separate=False))
                # print(Fore.RED)
                # print("in save")
                # print(Style.RESET_ALL)
                self.save('learning/models', fileName)
                self.save_log('learning/logs', fileName)
                convert_frames(frames, 'learning/renders', fileName, otype=otype)
            elif checkpoint:
                self.log.append([e, score, np.average(scores[-50:]), elapsed_time, None]) 
        elapsed_time = round((time() - start_time)/60, 2)
        print('Finished training {} episodes in {} minutes.'.format(num_episodes, (time() - start_time)/60))
    
    #TODO
    #in the evaluation, each agent will use what the agent 0 learnt in training
    def evaluate(self, render=False):
    #Run an episode and return the score and frames.
        frames = []
        score = 0
        action_n = [None, None, None, None]
        self.reset()
        if render: self.env.render()
        while True:
            for i in range(self.env.n_agents):
                action_n[i] = self.choose_action(training=False, agent_i=i) #get best action for each agent
            observation_n, reward_n, done_n, info = self.step(action_n) #perform action
            self.state = observation_n #update current state
            score += np.sum(reward_n) #cumulative score for episode for all agents
            if render: self.env.render()
            frames.append(self.env.render(mode='rgb_array'))
            if all(done_n):
                self.reset()
                break
        return score, frames
            
    def play_episode(self, render=False, render_and_save=False, otype='AVI'):
    #Run one episode. If render is true then render each episode to monitor.
    #If render_and_save is true then save frames and convert to GIF image or AVI movie.
    #The reward for the epsiode is returned.
        frames = []
        score = 0
        action_n = [None, None, None, None]
        self.reset()
        if render: self.env.render()
        while True:
            for i in range(self.env.n_agents):
                action_n[i] = self.choose_action(training=False, agent_i=i) #get best action
            print(action_n)
            observation_n, reward_n, done_n, info = self.step(action_n) #perform actions
            self.state = observation_n #update current state
            score += np.sum(reward_n) #cumulative score for episode
            if render: self.env.render()
            if render_and_save: frames.append(self.env.render(mode='rgb_array'))
            if all(done_n):
                print('Finished! Score: {}'.format(score))
                self.reset()
                break
        if render_and_save:
            fileName = 'DDDQN_PLAY_{}_{}'.format(self.game, Now(separate=False))
            convert_frames(frames, 'renders', fileName, otype=otype)


ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}
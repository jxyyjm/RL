import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, Multiply, Reshape, LeakyReLU, Embedding
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K

from agent2.ddpg import DDPG 
from agent2.td3 import TD3 
from agent2.dqn import DQN 
from data3 import DataInput
from memory import Transition

import numpy as np
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.disable_eager_execution()

'''
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
'''
#print(gpus)
class RlModel(object):
    def __init__(self, state_shape=1, action_size=1, batch_size=16, agent='ddpg', gamma=0.5, save_file='model/model'):
        self.gamma = gamma
        self.state_shape = state_shape 
        self.action_size = action_size
        self.batch_size = batch_size
        self.actor = self._generate_actor() 
        self.critic = self._generate_critic() 
        self.qfunc = self._generate_qfunc()
        self.agent_name = agent
        self.agent = self._generate_agent(agent)
        self.save_file = save_file
        self.t = 0 

    def _generate_actor(self):
        actor = Sequential([
            Dense(16, activation='relu', input_shape=(self.state_shape,)),
            Dense(16, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_size, activation='sigmoid'),
            #Lambda(lambda x: x*0.4 + 0.6)
            Lambda(lambda x: x*0.4 + 0.6)
        ])  
        #actor.summary()
        return actor

    def _generate_critic(self):
        action_input = Input(shape=(self.action_size,), name='action_input')
        state_input = Input(shape=(self.state_shape,), name='state_input')
        sq_a = K.square(action_input)
        ''' 
        cr_a = Concatenate()([sq_a, action_input])
        cr_a = Reshape((1, 2))(cr_a)
        '''
        cr_a_s = Multiply()([action_input, state_input])
        x = Concatenate()([action_input, sq_a, state_input, cr_a_s])
        x = Dense(32, activation='linear')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dense(32, activation='linear')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dense(32, activation='linear')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dense(1, activation='linear')(x)
        critic = Model(inputs=[action_input, state_input], outputs=x)
        #critic.summary()
        return critic

    def _generate_qfunc(self):
        state_input = Input(shape=(self.state_shape,), name='state_input')
        # 使用embedding
        x = Dense(128, activation='linear')(state_input)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(32, activation='linear')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(32, activation='linear')(x)
        x = LeakyReLU(alpha=0.1)(x)
        qfunc = Model(inputs=state_input, outputs=x)
        return qfunc

    def _generate_agent(self, agent):
        print('agent algo is %s' % agent)
        if agent == 'dqn':
            AGENT = DQN 
            agent = AGENT(model=self.qfunc, nsteps=1, actions=5, gamma=self.gamma, batch_size=self.batch_size, memsize=200)
        else: 
            if agent == 'ddpg':
                AGENT = DDPG
            else:
                AGENT = TD3 
            agent = AGENT(actor=self.actor, critic=self.critic, nsteps=1, gamma=self.gamma, batch_size=self.batch_size, memsize=200)
        return agent

    def save(self, overwrite=True):
        self.agent.save(self.save_file, overwrite=overwrite)

    def load(self):
        self.agent.load(self.save_file)

    def train(self, ds):
        dataset = ds.get_dataset()
        loss = 0
        q_score = 0
        act = 0
        N = 10000
        l, s, n = 0, 0, 0
        for i in range(30_000_000):
            state, action, reward, next_state, raw = next(dataset)
            self.agent.push(Transition(state, action, reward, next_state), i)

            if i % 2 == 0 and i:
                l, s = self.agent.train(i)
                if l:
                    loss += l; q_score += s; n+=1

            if i % N==0:
                act = self.agent.act(state, i)
                #print(time.time() - self.t)
                if n:
                    print('step %d, train: %d, act:%.2f, score:%.2f, loss:%.4f' % (i, n, act, q_score/n, loss/n))
                self.t = time.time()
                loss, q_score, n = 0, 0, 0
                act = 0.0
                #print('update model file')
                self.save()

    def test(self, ds):
        dataset = ds.get_dataset()
        self.load()
        for i in range(20):
            state, action, reward, next_state, raw= next(dataset)
            #print(state)
            #print(self.critic.predict_on_batch([np.array(action), np.array(state)]))
            inp = [np.array([action]), np.array([state])]
            #t = time.time()

            y = self.agent.critic.predict(inp)[0][0]
            y = '%.2f    ' % y
            print('critic1', y, raw, action)
            if self.agent_name == 'td3':
                y = self.agent.critic2.predict(inp)
                y = '%.2f    ' % y
                print('critic2', y, raw, action)


    def plot(self):
        plot_model(self.actor, to_file='actor.png')
        plot_model(self.critic, to_file='critic.png')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--state_shape', type=int, default=60)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--agent', type=str, default='td3')
    parser.add_argument('--t', type=str, default=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    parser.add_argument('--mode', type=str, default="train")
    args = parser.parse_args()

    model_file = 'model/model_%s_%s/' % (args.agent, args.t)
    #print('t is %s' % args.t)
    #print('model %s' % model_file)
    m = RlModel(agent=args.agent, state_shape=args.state_shape, gamma=args.gamma, save_file=model_file, batch_size=args.batch_size)
    ds = DataInput('../data/corpus2_shuf', batch_size=10, one_hot_size=args.state_shape)
    dt = DataInput('../data/corpus_test2', conffile='../hd_src/conf', batch_size=10, one_hot_size=args.state_shape)
    if args.mode in ["train"]:
        print('python run_model.py --mode test --agent %s --t %s' % (args.agent, args.t))
        m.train(ds)
        m.save()
    elif args.mode in ["test"]:
        m.test(dt)

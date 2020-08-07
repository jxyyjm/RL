import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, Multiply, Reshape, LeakyReLU, Embedding, BatchNormalization, Dropout
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.initializers import glorot_normal, glorot_uniform, RandomUniform, RandomNormal

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
    def __init__(self, state_shape=500, action_size=1, action_space=10, batch_size=16, agent='ddpg', gamma=0.5, save_file='model/model', memsize=1024):
        self.gamma = gamma
        self.state_shape = state_shape
        self.action_size = action_size
        self.action_space= action_space
        self.batch_size = batch_size
        self.memsize = memsize
        self.actor = self._generate_actor()
        self.critic = self._generate_critic() 
        self.qfunc = self._generate_qfunc()
        self.agent_name = agent
        self.agent = self._generate_agent(agent)
        self.save_file = save_file
        self.log_file = 'logs/log_'+agent+ save_file.split(agent)[1].strip('/')
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
        ''' 
        sq_a = K.square(action_input)
        sp_a = K.pow(action_input)

        cr_a = Concatenate()([sq_a, sp_a, state_input, action_input])
        cr_a = Reshape((1, 2))(cr_a)
        cr_a = Dense(32, activation='relu')(cr_a)
        cr_a = Dense(8, activation='relu')(cr_a)
        cr_a = Dense(1, activation='sigmoid')(cr_a)
        critic = Model(inputs=[action_input,state_input], outputs=cr_a)

        # states
        '''
        x = Dense(32, activation='linear')(state_input)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(rate=0.3)(x)
        x = Dense(32, activation='linear')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(rate=0.3)(x)

        # actions
        a = K.pow(action_input, 3)

        # merge
        x = Concatenate(axis=1)([x, action_input])
        x = Dense(32, activation='sigmoid')(x)
        x = Dense(8, activation='sigmoid')(x)
        x = Dense(1, activation='linear')(x)
        #x = Dense(1, activation='sigmoid')(x)
        critic = Model(inputs=[action_input, state_input], outputs=x)

        #critic.summary()
        return critic

    def _generate_qfunc(self):
        action_input = Input(shape=(self.action_size,), name='action_input')
        state_input = Input(shape=(self.state_shape,), name='state_input')

        # states
        x = Dense(32, activation='linear', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l1_l2(0.01,0.02))(state_input)
        #x = Dense(32, activation='linear', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l1_l2(0.01,0.02))(state_input)
        #x = Dense(32, activation='linear', kernel_initializer=RandomNormal(-0.06,0.004), kernel_regularizer=regularizers.l1_l2(0.01,0.02))(state_input)
        #x = Dense(32, activation='linear')(state_input)
        x = LeakyReLU(alpha=0.1)(x)

        # actions
        a = Lambda(lambda x: x*10-0)(action_input)
        a = K.cast(a, 'uint8')
        a = K.one_hot(a, self.action_space)
        a = K.squeeze(a, axis=1)

        # merge
        x = Concatenate(axis=1)([x, a])
        x = Dense(32, activation='linear', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l1_l2(0.01,0.02))(x)
        #x = Dense(32, activation='linear', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l1_l2(0.01,0.02))(x)
        #x = Dense(32, activation='linear', kernel_initializer=RandomNormal(-0.06,0.004), kernel_regularizer=regularizers.l1_l2(0.01,0.02))(x)
        #x = Dense(32, activation='linear')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dense(1, activation='linear', kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l1_l2(0.01,0.02))(x)
        #x = Dense(1, activation='linear', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l1_l2(0.01,0.02))(x)
        #x = Dense(1, activation='linear', kernel_initializer=RandomUniform(-0.06,0.004), kernel_regularizer=regularizers.l1_l2(0.01,0.02))(x)
        #x = Dense(1, activation='linear')(x)
        #x = Dense(1, activation='sigmoid')(x)
        qfunc = Model(inputs=[action_input, state_input], outputs=x)
        return qfunc

    def _generate_agent(self, agent):
        print('agent algo is %s' % agent)
        if agent == 'dqn':
            AGENT = DQN
            agent = AGENT(model=self.qfunc, nsteps=1, action_space=self.action_space, gamma=self.gamma, batch_size=self.batch_size, memsize=self.memsize)
        else:
            if agent == 'ddpg':
                AGENT = DDPG
            else:
                AGENT = TD3
            agent = AGENT(actor=self.actor, critic=self.critic, nsteps=1, gamma=self.gamma, batch_size=self.batch_size, memsize=2048)
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
        N = 1000
        l, s, n = 0, 0, 0
        logf = open(self.log_file, 'a')
        logf.write('here begin train '+str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(time.time()))))+'\n')
        for i in range(900_000_000):
            state, action, reward, next_state, raw = next(dataset)
            self.agent.push(Transition(state, action, reward, next_state), i)

            if i % self.batch_size == 0 and i:
                l, s = self.agent.train(i)
                if l:
                    loss += l; q_score += s; n+=1

            if (i % N==0 and i<10000) or (i%(10*N)==0):
                if self.agent_name != 'dqn':
                    act = self.agent.act(state, i)
                else:
                    act = 0
                #print(time.time() - self.t)
                if n:
                    #print('step %d, train: %d, act:%.2f, score:%.2f, loss:%.4f' % (i, n, act, q_score/n, loss/n))
                    logf.write('step %d, train: %d, act:%.2f, score:%.4f, loss:%.4f, %s' % (i, n, act, q_score/n, loss/n, '\n'))
                self.t = time.time()
                loss, q_score, n = 0, 0, 0
                act = 0.0
                #print('update model file')
                self.save()
            if (i % 10000 == 0):
                #self.test(logf, reLoad=False, step=i)
                self.test_avg_qval(logf, reLoad=False, step=i)
        logf.close()

    def test_avg_qval(self, logf, reLoad=False, step=0):
        if reLoad==True: self.load()
        test_data_file = '../data/corpus_yjm.test'
        kv_data_file = '../data/corpus_yjm.kv'
        dt = DataInput(test_data_file, conffile=kv_data_file, one_hot_size=self.state_shape)
        dataset = dt.get_dataset()
        res = []
        for i in range(1000):
            try:
                state, action, reward, next_state, raw= next(dataset)
                inp = [np.array([action]), np.array([state])]
                qval = self.agent.critic1.predict(inp)[0][0]
                res.append(qval)
                tmp = ''
                for act in range(1,11,1):
                    act = round(0.1*act, 1)
                    inp = [np.array([act]), np.array([state])]
                    y = self.agent.critic1.predict(inp)[0][0]
                    tmp += ' '+str(act)+'->'+ str(round(y,4))
                logf.write('step= ' +str(step) +' c1 predict sample' + str(i) +' '+ str(tmp) + '\n')
            except: break
        avg_qval = np.mean(res)
        logf.write('step= ' +str(step) +' c1 predict avg_qval->' + str(avg_qval) + '\n')
    
    def test(self, logf=None, reLoad=True, step=0):
        test_data_file = '../data/corpus_yjm.test'
        kv_data_file = '../data/corpus_yjm.kv'
        dt = DataInput(test_data_file, conffile=kv_data_file, one_hot_size=self.state_shape)

        dataset = dt.get_dataset()
        if reLoad==True: self.load()
        for i in range(5):
            res = ''
            state, action, reward, next_state, raw= next(dataset)
            #state, action, reward, next_state, raw= 
            for act in range(1,11,1):
                act = round(0.1*act, 1)
                inp = [np.array([act]), np.array([state])]
                y = self.agent.critic2.predict(inp)[0][0]
                res += ' '+str(act)+'->'+ str(round(y,4))
            #print('step=',step,'c2',raw, res)
            print ('step= ' +str(step) +' c2 ' +str(raw) +' '+ str(res))
            #logf.write('step= ' +str(step) +' c2 ' +str(raw) +' '+ str(res) + '\n')
            res = ''
            for act in range(1,11,1):
                act = round(0.1*act, 1)
                inp = [np.array([act]), np.array([state])]
                y = self.agent.critic.predict(inp)[0][0]
                res += ' '+str(act)+'->'+ str(round(y,3))
            #print('step=',step,'c1',raw, res)
            print ('step= ' +str(step) +' c1 ' +str(raw) +' '+ str(res))
            #logf.write('step= ' +str(step) +' c1 ' +str(raw) +' '+ str(res) + '\n')

            '''
            state, action, reward, next_state, raw= next(dataset)
            #print(state)
            #print(self.critic.predict_on_batch([np.array(action), np.array(state)]))
            inp = [np.array([action]), np.array([state])]
            #t = time.time()

            y = self.agent.critic.predict(inp)[0][0]
            y = '%.3f    ' % y
            print('critic1', int(i/5)%2, y, raw, action)
            if self.agent_name in ['td3', 'dqn']:
                y = self.agent.critic2.predict(inp)
                y = '%.3f    ' % y
                print('critic2', int(i/5)%2, y, raw, action)
            '''


    def plot(self):
        plot_model(self.actor, to_file='actor.png')
        plot_model(self.critic, to_file='critic.png')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--state_shape', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--memsize', type=int, default=128)
    parser.add_argument('--agent', type=str, default='dqn')
    parser.add_argument('--action_space', type=int, default=10)
    parser.add_argument('--t', type=str, default=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    parser.add_argument('--mode', type=str, default="train")
    args = parser.parse_args()

    model_file = 'model/model_%s_%s/' % (args.agent, args.t)
    #print('t is %s' % args.t)
    #print('model %s' % model_file)
    m = RlModel(agent=args.agent, action_space=args.action_space, state_shape=args.state_shape, gamma=args.gamma, save_file=model_file, batch_size=args.batch_size)

    train_data_file = '../data/corpus_yjm.train.shuf'
    test_data_file = '../data/corpus_yjm.test'
    kv_data_file = '../data/corpus_yjm.kv'
    ds = DataInput(train_data_file,conffile=kv_data_file, one_hot_size=args.state_shape)
    dt = DataInput(test_data_file, conffile=kv_data_file, one_hot_size=args.state_shape)

    if args.mode in ["train"]:
        print('python run_model.py --mode train --agent %s --t %s' % (args.agent, args.t))
        print('train-data:',train_data_file, '\ntest-data:',test_data_file, '\nkv-data:', kv_data_file)
        m.train(ds)
        m.save()
    elif args.mode in ["test"]:
        m.test(dt)

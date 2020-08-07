from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE 
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf
#import tensorflow_addons as tfa
import numpy as np
import sys, os
sys.path.append('../')
import memory
from policy import EpsGreedy, Greedy

class DQN():
    def __init__(self, model, action_space, optimizer=None, policy=None, test_policy=None,
                 memsize=10_000, target_update=10, gamma=0.99, batch_size=64, nsteps=1,
                 enable_double_dqn=True, enable_dueling_network=False, dueling_type='avg'):
        self.action_space = action_space
        #self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1, decay_steps=100000, decay_rate=0.7, staircase=True)
        #self.optimizer = Adam(learning_rate=self.lr_schedule) if optimizer is None else optimizer
        self.optimizer = Adam(learning_rate=1e-5) if optimizer is None else optimizer
        #self.optimizer = tfa.optimizers.LazyAdam(learning_rate=2e-5) if optimizer is None else optimizer
        #self.optimizer = tfa.optimizers.LazyAdam(learning_rate=self.lr_schedule) if optimizer is None else optimizer
    
        self.policy = EpsGreedy(0.1) if policy is None else policy
        self.test_policy = Greedy() if test_policy is None else test_policy

        self.memsize = memsize
        #self.memory = memory.PrioritizedExperienceReplay(memsize, nsteps)
        self.memory = memory.ExperienceReplay(memsize, nsteps)

        self.target_update = target_update
        self.gamma = gamma
        self.batch_size = batch_size
        self.nsteps = nsteps
        self.training = True

        # Extension options
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        self.model = model
        self.critic1 = self.critic = self.model
        self.critic2 = self.model2 = tf.keras.models.clone_model(self.model)

    def masked_q_loss(y_target, y_pred):
            return tf.keras.losses.mse(y_target, y_pred)
            #return tf.keras.losses.Huber(y_target, y_pred)

        self.model.compile(optimizer=self.optimizer, loss=masked_q_loss)
        self.critic2.compile(optimizer=self.optimizer, loss=masked_q_loss)

        # Clone model to use for delayed Q targets
        self.target_model1 = tf.keras.models.clone_model(self.model)
        self.target_model1.set_weights(self.model.get_weights())

        self.target_model2 = tf.keras.models.clone_model(self.model2)
        self.target_model2.set_weights(self.model2.get_weights())

    def save(self, path, overwrite=False):
        """Saves the model parameters to the specified file(s)."""
        if not os.path.exists(path):
            os.mkdir(path)
        self.model.save_weights(path+"model.h5", overwrite=overwrite)
        self.critic2.save_weights(path+"critic2.h5", overwrite=overwrite)

    def load(self, path):
        """Loads the model parameters to the specified file(s)."""
        self.model.load_weights(path+"model.h5")
        self.critic2.load_weights(path+"critic2.h5")
        print ('load from ', path)

    def predict_act_on_batch(self, model, state):
        """Returns the action to be taken given a state."""
        batch_size = np.shape(state)[0]
        # actions spaces
        actions = np.array([np.arange(self.action_space)])
        actions = np.repeat(actions, batch_size, axis=0)
        actions = np.reshape(actions, [-1, 1])

        # states
        states = np.repeat(state, self.action_space, axis=0)

        # qvals
        qvals = model.predict([actions, states])
        qvals = qvals.reshape([batch_size, self.action_space])
        return qvals

    def predict_point_act_on_batch(self, states, actions):
        """Returns the action to be taken given a state."""
        # qvals
        qvals = self.model.predict_on_batch([np.array(actions), np.array(states)])
        return qvals

    def push(self, transition, instance=0):
        """Stores the transition in memory."""
        self.memory.put(transition)

    def train(self, step):
        """Trains the agent for one step."""
        if len(self.memory) == 0:
            return None, None

        # Update target network
        if self.target_update >= 1 and step % self.target_update == 0:
            # Perform a hard update
            self.target_model1.set_weights(self.model.get_weights())
            self.target_model2.set_weights(self.model2.get_weights())
        elif self.target_update < 1:
            # Perform a soft update
            mw = np.array(self.model.get_weights())
            tmw = np.array(self.target_model1.get_weights())
            self.target_model1.set_weights(self.target_update * mw + (1 - self.target_update) * tmw)

            mw = np.array(self.model2.get_weights())
            tmw = np.array(self.target_model2.get_weights())
            self.target_model2.set_weights(self.target_update * mw + (1 - self.target_update) * tmw)

        # Train even when memory has fewer than the specified batch_size
        batch_size = min(len(self.memory), self.batch_size)

        # Sample batch_size traces from memory
        state_batch, action_batch, reward_batches, end_state_batch, not_done_mask = self.memory.get(batch_size)

        # Compute the value of the last next states
        target_qvals = np.zeros(batch_size)
        non_final_last_next_states = [es for es in end_state_batch if es is not None]

        if len(non_final_last_next_states) > 0:
            #selected_target_q_vals = self.target_model.predict_on_batch(np.array(non_final_last_next_states)).max(1)
            target_q_vals1 = self.predict_act_on_batch(self.target_model1, np.array(non_final_last_next_states))
            target_q_vals2 = self.predict_act_on_batch(self.target_model2, np.array(non_final_last_next_states))
            target_q_vals = np.minimum(target_q_vals1, target_q_vals2)
            #print(target_q_vals1, target_q_vals2, target_q_val)
            selected_target_q_vals = target_q_vals.max(1)
            non_final_mask = list(map(lambda s: s is not None, end_state_batch))
            target_qvals[non_final_mask] = selected_target_q_vals

        rewards = np.array(reward_batches)
        target_qvals = rewards + (self.gamma * target_qvals)

        # Compile information needed by the custom loss function
        # loss_data = [[i[0] for i in action_batch], target_qvals]

        # Train model
        q_score = self.model.predict_on_batch([np.array(action_batch), np.array(state_batch)])
        q_score = np.mean(q_score)
        #print(np.array(state_batch))
        #print(target_qvals)
        loss = self.model.train_on_batch([np.array(action_batch), np.array(state_batch)], target_qvals)
        self.critic2.train_on_batch([np.array(action_batch), np.array(state_batch)], target_qvals)
        return loss, q_score

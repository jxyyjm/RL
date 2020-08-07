from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.losses import MSE 
import tensorflow as tf
import numpy as np
import sys, os
sys.path.append('../')
import memory

class DDPG():
    def __init__(self, actor=None, critic=None, optimizer_critic=None, optimizer_actor=None,
                 memsize=10_000, target_update=1e-3, gamma=0.00099, batch_size=4, nsteps=1):
        self.actor = actor
        self.critic = critic

        self.optimizer_actor = Adam(lr=5e-3) if optimizer_actor is None else optimizer_actor
        self.optimizer_critic = Adam(lr=5e-3) if optimizer_critic is None else optimizer_critic

        self.memsize = memsize
        #self.memory = memory.PrioritizedExperienceReplay(memsize, nsteps, prob_alpha=0.2)
        self.memory = memory.ExperienceReplay(memsize, nsteps)

        self.target_update = target_update
        self.gamma = gamma
        self.batch_size = batch_size
        self.nsteps = nsteps
        self.training = True

        self.target_actor = tf.keras.models.clone_model(self.actor)
        self.target_critic = tf.keras.models.clone_model(self.critic)

        # Define loss function that computes the MSE between target Q-values and cumulative discounted rewards
        # If using PrioritizedExperienceReplay, the loss function also computes the TD error and updates the trace priorities
        def q_loss(data, qvals):
            """Computes the MSE between the Q-values of the actions that were taken and    the cumulative discounted
            rewards obtained after taking those actions. Updates trace priorities if using PrioritizedExperienceReplay.
            """
            target_qvals = data[:, 0, np.newaxis]
            if isinstance(self.memory, memory.PrioritizedExperienceReplay):
                def update_priorities(_qvals, _target_qvals, _traces_idxs):
                    """Computes the TD error and updates memory priorities."""
                    td_error = np.abs((_target_qvals - _qvals).numpy())[:, 0]
                    _traces_idxs = (tf.cast(_traces_idxs, tf.int32)).numpy()
                    self.memory.update_priorities(_traces_idxs, td_error)
                    return _qvals
                qvals = tf.py_function(func=update_priorities, inp=[qvals, target_qvals, data[:, 1]], Tout=tf.float32)
            return MSE(target_qvals, qvals)

        self.critic.compile(optimizer=self.optimizer_critic, loss=q_loss)

        # To train the actor we want to maximize the critic's output (action value) given the predicted action as input
        # Namely we want to change the actor's weights such that it picks the action that has the highest possible value
        state_input = self.critic.input[1]
        critic_output = self.critic([self.actor(state_input), state_input])
        actor_loss = -tf.keras.backend.mean(critic_output)
        actor_updates = self.optimizer_actor.get_updates(params=self.actor.trainable_weights, loss=actor_loss)
        self.actor_train_on_batch = tf.keras.backend.function(inputs=[state_input], outputs=[self.actor(state_input)], updates=actor_updates)

    def save(self, path, overwrite=False):
        """Saves the model parameters to the specified file(s)."""
        if not os.path.exists(path):
            os.mkdir(path)
        self.actor.save_weights(path+"actor.h5", overwrite=overwrite)
        self.critic.save_weights(path+"critic.h5", overwrite=overwrite)

    def load(self, path):
        """Loads the model parameters to the specified file(s)."""
        self.actor.load_weights(path+"actor.h5")
        self.critic.load_weights(path+"critic.h5")

    def act(self, state, instance=0):
        """Returns the action to be taken given a state."""
        action = self.actor.predict(np.array([state]))[0]
        return action

    def act_on_batch(self, state, instance=0):
        """Returns the action to be taken given a state."""
        action = self.target_actor.predict_on_batch(np.array(state))
        return action

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
            self.target_actor.set_weights(self.actor.get_weights())
            self.target_critic.set_weights(self.critic.get_weights())
        elif self.target_update < 1:
            # Perform a soft update
            a_w = np.array(self.actor.get_weights())
            ta_w = np.array(self.target_actor.get_weights())
            self.target_actor.set_weights(self.target_update*a_w + (1-self.target_update)*ta_w)
            c_w = np.array(self.critic.get_weights())
            tc_w = np.array(self.target_critic.get_weights())
            self.target_critic.set_weights(self.target_update*c_w + (1-self.target_update)*tc_w)

        # Train even when memory has fewer than the specified batch_size
        batch_size = min(len(self.memory), self.batch_size)

        # Sample batch_size traces from memory
        state_batch, action_batch, reward_batches, end_state_batch, not_done_mask = self.memory.get(batch_size)

        # Compute the value of the last next states
        target_qvals = np.zeros(batch_size)
        non_final_last_next_states = [es for es in end_state_batch if es is not None]
        if len(non_final_last_next_states) > 0:
            non_final_mask = list(map(lambda s: s is not None, end_state_batch))
            #target_actions = self.target_actor.predict_on_batch(np.array(non_final_last_next_states))
            target_actions = self.act_on_batch(non_final_last_next_states)
            target_qvals[non_final_mask] = self.target_critic.predict_on_batch([target_actions, np.array(non_final_last_next_states)]).squeeze()

        # Compute n-step discounted return
        # If episode ended within any sampled nstep trace - zero out remaining rewards
        for n in reversed(range(self.nsteps)):
            rewards = np.array([b[n] for b in reward_batches])
            target_qvals *= np.array([t[n] for t in not_done_mask])
            target_qvals = rewards + (self.gamma * target_qvals)

        # Train actor
        if step > 100_000 and step % 10 == 0:
            self.actor_train_on_batch([np.array(state_batch)])
        #print(target_qvals)
        #print(self.memory.last_traces_idxs())

        # Train critic
        PER = isinstance(self.memory, memory.PrioritizedExperienceReplay)
        critic_loss_data = np.stack([target_qvals, self.memory.last_traces_idxs()], axis=1) if PER else target_qvals
        q_score = self.critic.predict_on_batch([np.array(action_batch), np.array(state_batch)])
        q_score = np.mean(q_score)
        loss = self.critic.train_on_batch(x=[np.array(action_batch), np.array(state_batch)], y=critic_loss_data)
        return loss, q_score

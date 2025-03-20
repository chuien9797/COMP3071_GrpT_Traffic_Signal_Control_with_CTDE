import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np


class PPOModel:
    def __init__(self, input_dim, output_dim, hidden_size=64, learning_rate=0.001, clip_ratio=0.2, update_epochs=10,
                 batch_size=32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self._batch_size = batch_size  # Added batch_size attribute

        # Build the actor and critic networks
        self.actor, self.critic = self.build_actor_critic()
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)

    def build_actor_critic(self):
        # Shared input layer
        inputs = layers.Input(shape=(self.input_dim,))
        common = layers.Dense(self.hidden_size, activation='relu')(inputs)

        # Actor branch (outputs action probabilities)
        actor_hidden = layers.Dense(self.hidden_size, activation='relu')(common)
        action_probs = layers.Dense(self.output_dim, activation='softmax')(actor_hidden)
        actor = models.Model(inputs=inputs, outputs=action_probs)

        # Critic branch (outputs state value)
        critic_hidden = layers.Dense(self.hidden_size, activation='relu')(common)
        state_value = layers.Dense(1, activation='linear')(critic_hidden)
        critic = models.Model(inputs=inputs, outputs=state_value)

        return actor, critic

    def select_action(self, state):
        """
        Given a state, sample an action according to the actor network's probabilities.
        Returns the chosen action and its probability.
        """
        state = state.reshape(1, -1)
        probs = self.actor(state).numpy().flatten()
        action = np.random.choice(self.output_dim, p=probs)
        return action, probs[action]

    def predict_one(self, state):
        """
        Returns the action probability distribution for a given state.
        This allows compatibility with simulation code that uses np.argmax on the result.
        """
        state = state.reshape(1, -1)
        return self.actor(state).numpy().flatten()

    def train(self, states, actions, advantages, returns, old_probs):
        """
        Update the actor and critic networks using the PPO clipped objective.
        This function assumes that you've precomputed the advantages and returns.
        """
        # Convert inputs to tensors if they aren't already
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)

        for _ in range(self.update_epochs):
            with tf.GradientTape() as tape:
                new_probs = self.actor(states)
                action_masks = tf.one_hot(actions, self.output_dim)
                selected_new_probs = tf.reduce_sum(new_probs * action_masks, axis=1)
                ratio = selected_new_probs / (old_probs + 1e-10)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
                critic_loss = tf.reduce_mean(tf.square(returns - self.critic(states)))
                total_loss = actor_loss + 0.5 * critic_loss

            grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

        return total_loss.numpy()

    @property
    def batch_size(self):
        return self._batch_size

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Lane Embedding Network
# -----------------------------
class LaneEmbeddingNetwork(tf.keras.Model):
    def __init__(self, lane_feature_dim, embedding_dim=32, hidden_units=64):
        super().__init__()
        self.dense1 = layers.Dense(hidden_units, activation='relu')
        self.dense2 = layers.Dense(embedding_dim, activation='relu')

    def call(self, lane_input):
        # lane_input shape: (batch_size, lane_feature_dim)
        x = self.dense1(lane_input)
        x = self.dense2(x)
        return x  # shape: (batch_size, embedding_dim)

# -----------------------------
# PPO Model Aggregator
# -----------------------------
class PPOModelAggregator(tf.keras.Model):
    def __init__(self, lane_feature_dim, embedding_dim, final_hidden, num_actions):
        super().__init__()
        self.lane_embed = LaneEmbeddingNetwork(lane_feature_dim, embedding_dim)
        self.final_hidden_layer = layers.Dense(final_hidden, activation='relu')
        self.policy_logits = layers.Dense(num_actions, activation=None)  # logits (no activation)
        self.value = layers.Dense(1, activation=None)  # scalar value

    def call(self, list_of_lanes):
        """
        Expects list_of_lanes with shape (batch_size, num_lanes, lane_feature_dim).
        We dynamically determine the dimensions and reshape accordingly.
        """
        # Dynamically determine dimensions:
        batch_size = tf.shape(list_of_lanes)[0]
        num_lanes = tf.shape(list_of_lanes)[1]
        lane_feature_dim = tf.shape(list_of_lanes)[2]
        # Reshape from (batch_size, num_lanes, lane_feature_dim) -> (batch_size*num_lanes, lane_feature_dim)
        x = tf.reshape(list_of_lanes, [batch_size * num_lanes, lane_feature_dim])
        # Compute lane embeddings:
        embeddings = self.lane_embed(x)
        # Reshape embeddings back to (batch_size, num_lanes, embedding_dim)
        embeddings = tf.reshape(embeddings, [batch_size, num_lanes, -1])
        # Aggregate lane embeddings by mean pooling:
        x = tf.reduce_mean(embeddings, axis=1)
        hidden = self.final_hidden_layer(x)
        logits = self.policy_logits(hidden)
        value = self.value(hidden)
        # Squeeze value to shape (batch_size,) if needed.
        return logits, tf.squeeze(value, axis=-1)

# -----------------------------
# TrainModelPPO Wrapper
# -----------------------------
class TrainModelPPO:
    def __init__(self,
                 lane_feature_dim,
                 embedding_dim,
                 final_hidden,
                 num_actions,
                 batch_size,
                 learning_rate):
        self._lane_feature_dim = lane_feature_dim
        self._embedding_dim = embedding_dim
        self._final_hidden = final_hidden
        self._num_actions = num_actions
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model()
        self.optimizer = Adam(learning_rate=self._learning_rate)

    def _build_model(self):
        model = PPOModelAggregator(
            lane_feature_dim=self._lane_feature_dim,
            embedding_dim=self._embedding_dim,
            final_hidden=self._final_hidden,
            num_actions=self._num_actions
        )
        return model

    def predict(self, state):
        """
        Accepts a state of shape (num_lanes, lane_feature_dim) and expands it to a batch of one.
        Returns a tuple (logits, value) as NumPy arrays.
        """
        state_exp = np.expand_dims(state, axis=0)  # shape becomes (1, num_lanes, lane_feature_dim)
        logits, value = self._model(state_exp)
        return logits.numpy(), value.numpy()

    def get_action(self, state):
        """
        Samples an action stochastically from the policy given state.
        Returns:
          - action (integer index),
          - log_prob (float),
          - value estimate for the state.
        """
        logits, value = self.predict(state)
        # Compute action probabilities using softmax
        action_probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
        action = np.random.choice(self._num_actions, p=action_probs)
        log_prob = np.log(action_probs[action] + 1e-10)
        return action, log_prob, value[0]

    def train(self, states, actions, old_log_probs, returns, advantages, clip_ratio, update_epochs):
        """
        Executes PPO updates.
          - states: tensor or numpy array of shape (N, num_lanes, lane_feature_dim)
          - actions: numpy array of shape (N,) with action indices
          - old_log_probs: numpy array of shape (N,) of log probabilities of chosen actions
          - returns: numpy array of shape (N,) of target returns
          - advantages: numpy array of shape (N,) of advantage estimates
          - clip_ratio: clipping ratio hyperparameter (float)
          - update_epochs: number of passes over the data (integer)
        """
        # Convert arrays to tensors:
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        for _ in range(update_epochs):
            with tf.GradientTape() as tape:
                logits, values = self._model(states)
                action_probs = tf.nn.softmax(logits, axis=-1)
                # Gather new log probabilities for the actions taken:
                indices = tf.stack([tf.range(tf.shape(logits)[0]), actions], axis=1)
                new_log_probs = tf.math.log(tf.gather_nd(action_probs, indices) + 1e-10)
                ratio = tf.exp(new_log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
                policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
                value_loss = tf.reduce_mean((returns - tf.squeeze(values)) ** 2)
                # Optional entropy bonus:
                entropy = -tf.reduce_mean(action_probs * tf.math.log(action_probs + 1e-10))
                total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            grads = tape.gradient(total_loss, self._model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

    def save_model(self, path):
        self._model.save(os.path.join(path, 'trained_model.h5'))


import os
import tensorflow as tf
import numpy as np


class TrainModelPPO(tf.keras.Model):
    def __init__(self, lane_feature_dim, hidden_size, learning_rate, clip_ratio,
                 update_epochs, training_epochs, num_actions, use_priority=False, reward_scale=1.0):
        """
        Initialize the adaptive PPO model with optional reward scaling and prioritized update.

        Parameters:
            lane_feature_dim (int): Dimensionality of each lane feature vector (e.g. 9).
            hidden_size (int): Number of neurons for the per-lane processing.
            learning_rate (float): Learning rate for the optimizer.
            clip_ratio (float): The PPO clip ratio.
            update_epochs (int): Number of update epochs for each batch of rollouts.
            training_epochs (int): (Optional) Training epochs parameter for bookkeeping.
            num_actions (int): Number of actions (output units) for the policy head.
            use_priority (bool): If True, weight sample losses by TD error (prioritized update).
            reward_scale (float): Scaling factor to amplify the reward signal (applied to advantages).
        """
        super(TrainModelPPO, self).__init__()
        self.lane_feature_dim = lane_feature_dim
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.training_epochs = training_epochs
        self.use_priority = use_priority
        self.reward_scale = reward_scale

        # Process each lane using TimeDistributed layers.
        self.shared_dense1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.hidden_size, activation='relu')
        )
        self.shared_dense2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.hidden_size, activation='relu')
        )
        # Global average pooling makes the network adaptive to a variable number of lanes.
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()

        # Policy head: outputs logits for each action.
        self.policy_logits = tf.keras.layers.Dense(self.num_actions, activation=None)
        # Value head: outputs a single state value.
        self.value = tf.keras.layers.Dense(1, activation=None)
        # Optimizer.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    @tf.function
    def call(self, inputs):
        """
        Forward pass.

        Parameters:
            inputs: Tensor of shape (batch_size, num_lanes, lane_feature_dim).

        Returns:
            logits: Shape (batch_size, num_actions)
            value: Shape (batch_size, 1)
        """
        # Process each lane independently.
        x = self.shared_dense1(inputs)
        x = self.shared_dense2(x)
        # Global pooling over lanes.
        x = self.global_pool(x)  # (batch_size, hidden_size)
        logits = self.policy_logits(x)
        value = self.value(x)
        return logits, value

    def act(self, state):
        """
        Given a single state (shape: (num_lanes, lane_feature_dim)), sample an action.

        Returns:
            action (int), log_prob (tf.Tensor), value_est (tf.Tensor)
        """
        state = tf.expand_dims(state, axis=0)  # (1, num_lanes, lane_feature_dim)
        logits, value = self.call(state)
        action_dist = tf.random.categorical(logits, num_samples=1)
        action = tf.squeeze(action_dist, axis=1)[0]
        probs = tf.nn.softmax(logits)
        selected_prob = probs[0, action]
        log_prob = tf.math.log(selected_prob + 1e-8)
        value_est = tf.squeeze(value, axis=1)[0]
        return int(action.numpy()), log_prob, value_est

    def ppo_update(self, states, actions, old_log_probs, advantages, returns):
        """
        Perform PPO update using the provided rollout batch.

        Parameters:
            states: np.array of shape (batch, num_lanes, lane_feature_dim)
            actions: np.array of shape (batch,)
            old_log_probs: np.array of shape (batch,)
            advantages: np.array of shape (batch,) -- will be scaled by self.reward_scale
            returns: np.array of shape (batch,)

        Returns:
            total_loss: final loss on batch.
        """
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        # Scale the advantages by the reward_scale factor.
        advantages = tf.convert_to_tensor(advantages * self.reward_scale, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        for epoch in range(self.update_epochs):
            with tf.GradientTape() as tape:
                logits, value = self.call(states)
                value = tf.squeeze(value, axis=1)
                # Compute log probabilities for the selected actions.
                action_masks = tf.one_hot(actions, self.num_actions)
                log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
                # Calculate probability ratio.
                ratio = tf.exp(log_probs - old_log_probs)
                surrogate1 = ratio * advantages
                surrogate2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                value_loss = tf.reduce_mean(tf.square(returns - value))
                loss = policy_loss + 0.5 * value_loss

                # If prioritized update is enabled, weight each sample by its TD error.
                if self.use_priority:
                    td_error = tf.abs(returns - value)
                    # Define weights: for example, use (1 + td_error).
                    weights = 1.0 + td_error
                    # Compute weighted losses:
                    policy_loss = -tf.reduce_mean(weights * tf.minimum(surrogate1, surrogate2))
                    value_loss = tf.reduce_mean(weights * tf.square(returns - value))
                    loss = policy_loss + 0.5 * value_loss

            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.save_weights(os.path.join(path, "ppo_weights.h5"))

    def load_model(self, path):
        self.load_weights(os.path.join(path, "ppo_weights.h5"))

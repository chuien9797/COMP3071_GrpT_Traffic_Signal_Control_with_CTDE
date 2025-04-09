import tensorflow as tf
import numpy as np
import os


class PPOActorCritic(tf.keras.Model):
    def __init__(self, lane_feature_dim, embedding_dim, final_hidden, num_actions):
        """
        Actor-Critic network for PPO.
        Instead of flattening the input, we use GlobalAveragePooling1D to aggregate per-lane features.

        Parameters:
            lane_feature_dim: Number of features per lane (e.g., 9 to match your simulation _get_state())
            embedding_dim: Unused in this basic implementation (reserved for further extension)
            final_hidden: Number of hidden units in the common dense layer
            num_actions: Number of available actions for the actor head
        """
        super(PPOActorCritic, self).__init__()
        # Aggregate per-lane features into one fixed-size vector.
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        # Common dense layer.
        self.dense1 = tf.keras.layers.Dense(final_hidden, activation='relu')
        # Actor head: outputs logits.
        self.actor_logits = tf.keras.layers.Dense(num_actions, activation=None)
        # Critic head: outputs a scalar state value.
        self.critic = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        """
        Forward pass.
            inputs: Tensor of shape (batch_size, num_lanes, lane_feature_dim)
        Returns:
            logits: Tensor of shape (batch_size, num_actions)
            value: Tensor of shape (batch_size, 1)
        """
        x = self.global_pool(inputs)
        x = self.dense1(x)
        logits = self.actor_logits(x)
        value = self.critic(x)
        return logits, value


class TrainModelPPO:
    def __init__(self,
                 lane_feature_dim,
                 embedding_dim,
                 final_hidden,
                 num_actions,
                 batch_size,
                 learning_rate,
                 clip_ratio,
                 update_epochs,
                 gamma):
        """
        Wrapper for training PPO.

        Parameters:
            lane_feature_dim: Number of features per lane (e.g., 9)
            embedding_dim: (Reserved for extended architectures)
            final_hidden: Number of hidden units in the shared dense layer
            num_actions: Number of available actions
            batch_size: Mini-batch size for the PPO update
            learning_rate: Learning rate for the optimizer
            clip_ratio: Clipping ratio in PPO surrogate objective
            update_epochs: Number of epochs over rollout data for updates
            gamma: Discount factor
        """
        self.lane_feature_dim = lane_feature_dim
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.gamma = gamma

        self.model = PPOActorCritic(lane_feature_dim, embedding_dim, final_hidden, num_actions)
        # Optionally, call self.model.build(...) to force weight initialization.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def predict(self, states):
        """
        Predict action probabilities and state values.

        Parameters:
            states: Tensor with shape (batch_size, num_lanes, lane_feature_dim)
        Returns:
            action_probs: Softmax probabilities; shape (batch_size, num_actions)
            values: State-value estimates; shape (batch_size, 1)
        """
        logits, values = self.model(states)
        action_probs = tf.nn.softmax(logits)
        return action_probs, values

    def act(self, state):
        """
        Sample an action from the current policy.

        Parameters:
            state: Numpy array with shape (num_lanes, lane_feature_dim)
        Returns:
            action: An integer representing the selected action.
            log_prob: Log probability of the action.
            value: State value estimate.
        """
        state = np.expand_dims(state, axis=0)  # Create a batch of size 1.
        logits, value = self.model(state)
        action_probs = tf.nn.softmax(logits)
        # Sample action.
        action_dist = tf.random.categorical(logits, num_samples=1)
        action = int(action_dist.numpy()[0, 0])
        log_probs = tf.nn.log_softmax(logits)
        log_prob = log_probs[0, action].numpy()
        return action, log_prob, value.numpy()[0, 0]

    def ppo_update(self, states, actions, old_log_probs, advantages, returns):
        """
        Perform a PPO update using mini-batch gradient descent.

        Parameters:
            states: Array with shape (N, num_lanes, lane_feature_dim)
            actions: Array with shape (N,)
            old_log_probs: Array with shape (N,)
            advantages: Array with shape (N,) (should be normalized beforehand)
            returns: Array with shape (N,)
        """
        # Create a dataset and shuffle it.
        dataset = tf.data.Dataset.from_tensor_slices(
            (states, actions, old_log_probs, advantages, returns)
        )
        dataset = dataset.shuffle(buffer_size=1024).batch(self.batch_size)

        for epoch in range(self.update_epochs):
            for batch in dataset:
                s_batch, a_batch, old_logp_batch, adv_batch, ret_batch = batch
                with tf.GradientTape() as tape:
                    logits, values = self.model(s_batch)
                    values = tf.squeeze(values, axis=1)
                    log_probs_all = tf.nn.log_softmax(logits)
                    # Gather the log probabilities corresponding to chosen actions.
                    indices = tf.stack([tf.range(tf.shape(a_batch)[0]), a_batch], axis=1)
                    new_logp = tf.gather_nd(log_probs_all, indices)
                    # Compute probability ratio.
                    ratio = tf.exp(new_logp - old_logp_batch)
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    surrogate_loss = -tf.reduce_mean(tf.minimum(ratio * adv_batch, clipped_ratio * adv_batch))
                    value_loss = tf.reduce_mean(tf.square(ret_batch - values))
                    probs = tf.nn.softmax(logits)
                    entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs_all, axis=1))
                    loss = surrogate_loss + 0.5 * value_loss - 0.01 * entropy
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def save_model(self, path):
        """
        Save the trained PPO model.
        """
        self.model.save(os.path.join(path, 'trained_model_ppo.h5'))


def compute_discounted_returns(rewards, gamma):
    """
    Compute cumulative discounted returns.

    Parameters:
        rewards: List or NumPy array of rewards.
        gamma: Discount factor.
    Returns:
        discounted_returns: NumPy array of the same shape as rewards.
    """
    discounted_returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        discounted_returns[t] = running_return
    return discounted_returns

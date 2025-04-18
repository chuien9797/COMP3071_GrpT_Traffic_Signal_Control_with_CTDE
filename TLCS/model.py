##############################################################################
# Filename: model.py
# Purpose:  Demonstration of Per-Lane Embedding + Aggregation for DQN
##############################################################################

import os
import sys
import numpy as np
import tensorflow as tf

from keras.src.saving.saving_lib import load_model
from keras.src.utils import plot_model
from tensorflow import keras
from keras import layers
from keras import losses
from tensorflow.keras.optimizers import Adam


class LaneEmbeddingNetwork(tf.keras.Model):
    """
    A small MLP that processes a single lane's feature vector into an embedding.
    """
    def __init__(self, lane_feature_dim, embedding_dim=32, hidden_units=64):
        super().__init__()
        self.dense1 = layers.Dense(hidden_units, activation='relu')
        self.dense2 = layers.Dense(embedding_dim, activation='relu')

    def call(self, lane_input):
        # lane_input shape: (batch_size, lane_feature_dim)
        x = self.dense1(lane_input)
        x = self.dense2(x)
        return x  # shape: (batch_size, embedding_dim)


class AggregatorDQN(tf.keras.Model):
    """
    Aggregator that takes variable number of lane embeddings, does mean pooling,
    then outputs Q-values using a final MLP.
    """
    def __init__(self, embedding_dim=32, final_hidden=64, num_actions=4):
        super().__init__()
        self.final_fc1 = layers.Dense(final_hidden, activation='relu')
        self.final_fc2 = layers.Dense(num_actions, activation='linear')

    def call(self, lane_embeddings):
        # lane_embeddings shape: (batch_size, num_lanes, embedding_dim)
        x = tf.reduce_mean(lane_embeddings, axis=1)  # shape: (batch_size, embedding_dim)
        x = self.final_fc1(x)
        q_values = self.final_fc2(x)
        return q_values  # shape: (batch_size, num_actions)


class DQNSetModel(tf.keras.Model):
    """
    The full pipeline that:
     1) Embeds each lane with LaneEmbeddingNetwork
     2) Aggregates the embeddings via mean pooling
     3) Outputs Q-values with a final MLP (AggregatorDQN)
    """
    def __init__(self, lane_feature_dim, embedding_dim, final_hidden, num_actions):
        super().__init__()
        self.lane_embed = LaneEmbeddingNetwork(lane_feature_dim, embedding_dim)
        self.aggregator = AggregatorDQN(embedding_dim, final_hidden, num_actions)

    def call(self, list_of_lanes):
        """
        list_of_lanes shape: (batch_size, num_lanes, lane_feature_dim)
        """
        batch_size = tf.shape(list_of_lanes)[0]
        num_lanes  = tf.shape(list_of_lanes)[1]
        x = tf.reshape(list_of_lanes, [-1, list_of_lanes.shape[-1]])
        # x shape: (batch_size * num_lanes, lane_feature_dim)
        embeddings = self.lane_embed(x)
        # embeddings shape: (batch_size * num_lanes, embedding_dim)
        embeddings = tf.reshape(embeddings, [batch_size, num_lanes, -1])
        q_values = self.aggregator(embeddings)
        return q_values


##############################################################################
# TrainModelAggregator: a wrapper class similar to your old TrainModel,
# but using the set-based aggregator approach.
##############################################################################
class TrainModelAggregator:
    def __init__(self,
                 lane_feature_dim= 6+3,
                 embedding_dim=64,
                 final_hidden=128,
                 num_actions=4,
                 batch_size=64,
                 learning_rate=1e-3,
                 model=None):
        self._lane_feature_dim = lane_feature_dim
        self._embedding_dim = embedding_dim
        self._final_hidden = final_hidden
        self._num_actions = num_actions
        self._batch_size = batch_size
        self._learning_rate = learning_rate

        if model is not None:
            self._model = model
        else:
            self._model = self._build_model()

    def _build_model(self):
        dqn_model = DQNSetModel(
            lane_feature_dim=self._lane_feature_dim,
            embedding_dim=self._embedding_dim,
            final_hidden=self._final_hidden,
            num_actions=self._num_actions
        )
        dqn_model.compile(
            loss=losses.MeanSquaredError(),
            optimizer=Adam(learning_rate=self._learning_rate)
        )
        return dqn_model

    def get_weights(self):
        return self._model.get_weights()

    def set_weights(self, weights):
        self._model.set_weights(weights)

    def predict_one(self, state):
        """
        state shape: (num_lanes, lane_feature_dim).
        Expand dims to (1, num_lanes, lane_feature_dim) and predict.
        """
        state = np.expand_dims(state, axis=0)
        q_values = self._model(state)
        return q_values.numpy()

    def predict_batch(self, states):
        """
        states shape: (batch_size, num_lanes, lane_feature_dim)
        """
        return self._model(states).numpy()

    def train_batch(self, states, q_sa):
        """
        states: (batch_size, num_lanes, lane_feature_dim)
        q_sa: (batch_size, num_actions)
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)

    def save_model(self, path):
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'),
                   show_shapes=True, show_layer_names=True)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def model(self):
        return self._model

    def load_from_disk(self, model_path):
        from keras.src.saving.saving_lib import load_model
        loaded = load_model(model_path)
        self._model = loaded

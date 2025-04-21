##############################################################################
# Filename: model.py
# Purpose:  Per‑Lane Embedding + Aggregation DQN (single shared policy ready)
##############################################################################

import os
import numpy as np
import tensorflow as tf
from keras.src.utils import plot_model
from keras import layers, losses
from tensorflow.keras.optimizers import Adam


# --------------------------------------------------------------------------- #
#  Basic building blocks
# --------------------------------------------------------------------------- #
class LaneEmbeddingNetwork(tf.keras.Model):
    """
    A small MLP that embeds one lane’s feature vector.
    """
    def __init__(self, lane_feature_dim, embedding_dim=32, hidden_units=64):
        super().__init__()
        self.dense1 = layers.Dense(hidden_units, activation='relu')
        self.dense2 = layers.Dense(embedding_dim, activation='relu')

    def call(self, lane_input):
        # lane_input: (batch, lane_feature_dim)
        x = self.dense1(lane_input)
        return self.dense2(x)           # (batch, embedding_dim)


class AggregatorDQN(tf.keras.Model):
    """
    Mean‑pool lane embeddings ▶ two‑layer MLP ▶ Q‑values.
    """
    def __init__(self, embedding_dim=32, final_hidden=64, num_actions=4):
        super().__init__()
        self.fc1 = layers.Dense(final_hidden, activation='relu')
        self.fc2 = layers.Dense(num_actions, activation='linear')

    def call(self, lane_embeddings):
        # lane_embeddings: (batch, num_lanes, embedding_dim)
        pooled = tf.reduce_mean(lane_embeddings, axis=1)  # (batch, embedding_dim)
        x      = self.fc1(pooled)
        return self.fc2(x)                                # (batch, num_actions)


class DQNSetModel(tf.keras.Model):
    """
    Full pipeline: lane‑wise embed ➜ mean pool ➜ Q‑network.
    """
    def __init__(self, lane_feature_dim, embedding_dim, final_hidden, num_actions):
        super().__init__()
        self.embedder  = LaneEmbeddingNetwork(lane_feature_dim, embedding_dim)
        self.aggregator = AggregatorDQN(embedding_dim, final_hidden, num_actions)

    def call(self, lanes):
        # lanes: (batch, num_lanes, lane_feature_dim)
        b, n = tf.shape(lanes)[0], tf.shape(lanes)[1]
        flat = tf.reshape(lanes, [-1, lanes.shape[-1]])          # (batch*num_lanes, feat)
        emb  = self.embedder(flat)                               # (batch*num_lanes, emb)
        emb  = tf.reshape(emb, [b, n, -1])                       # (batch, num_lanes, emb)
        return self.aggregator(emb)                              # (batch, num_actions)


# --------------------------------------------------------------------------- #
#  Convenience wrapper: TrainModelAggregator
# --------------------------------------------------------------------------- #
class TrainModelAggregator:
    """
    Thin wrapper that matches the interface used in your training loop.
    """
    def __init__(self,
                 lane_feature_dim=9,      # 6 raw + 3 one‑hot intersection type
                 embedding_dim=64,
                 final_hidden=128,
                 num_actions=4,
                 batch_size=64,
                 learning_rate=1e-3,
                 model=None):
        self._lane_feature_dim = lane_feature_dim
        self._embedding_dim    = embedding_dim
        self._final_hidden     = final_hidden
        self._num_actions      = num_actions
        self._batch_size       = batch_size
        self._learning_rate    = learning_rate

        self._model = model if model is not None else self._build_model()

    # ---------------- internal ------------------------------------------------
    def _build_model(self):
        net = DQNSetModel(
            lane_feature_dim=self._lane_feature_dim,
            embedding_dim=self._embedding_dim,
            final_hidden=self._final_hidden,
            num_actions=self._num_actions
        )
        net.compile(
            loss=losses.MeanSquaredError(),
            optimizer=Adam(self._learning_rate)
        )
        return net

    # ---------------- public API ---------------------------------------------
    def predict_one(self, state):
        """
        state: (num_lanes, lane_feature_dim) → returns (1, num_actions)
        """
        return self._model(np.expand_dims(state, 0)).numpy()

    def predict_batch(self, states):
        """
        states: (batch, num_lanes, lane_feature_dim)
        """
        return self._model(states).numpy()

    def train_batch(self, states, q_targets):
        """
        One SGD step (silent).
        """
        self._model.fit(states, q_targets, epochs=1, verbose=0)

    # ---------------- weight utils -------------------------------------------
    def get_weights(self):            return self._model.get_weights()
    def set_weights(self, weights):   self._model.set_weights(weights)

    # >>> NEW – helper to create a target net quickly <<<
    @classmethod
    def clone_from(cls, src, tau=1.0):
        """
        Build a *copy* (full or soft) from an existing TrainModelAggregator
        instance `src`.  `tau=1` ⇒ hard copy; 0<tau<1 ⇒ soft update.
        """
        cloned_keras = tf.keras.models.clone_model(src._model)
        cloned_keras.set_weights(src.get_weights())
        tgt = cls(lane_feature_dim=src._lane_feature_dim,
                  embedding_dim=src._embedding_dim,
                  final_hidden=src._final_hidden,
                  num_actions=src._num_actions,
                  batch_size=src._batch_size,
                  learning_rate=src._learning_rate,
                  model=cloned_keras)
        if tau < 1.0:
            tgt.soft_update_from(src, tau)
        return tgt

    def soft_update_from(self, src, tau=0.005):
        """
        Polyak averaging: θ_target ← τ θ_src + (1‑τ) θ_target.
        """
        new_w = []
        for w_tgt, w_src in zip(self.get_weights(), src.get_weights()):
            new_w.append((1. - tau) * w_tgt + tau * w_src)
        self.set_weights(new_w)

    # ---------------- misc ----------------------------------------------------
    # model.py  ── inside class TrainModelAggregator
    # model.py  – inside TrainModelAggregator.save_model()
    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        h5_path = os.path.join(path, "trained_model.h5")
        self._model.save(h5_path)
        print(f"[Model] ➜ saved HDF5 weights to: {h5_path}")

        # --- OPTIONAL diagram -------------------------------------------------
        try:
            from keras.utils import plot_model  # < will raise ImportError if pydot absent
            png_path = os.path.join(path, "model_structure.png")
            plot_model(self._model, to_file=png_path,
                       show_shapes=True, show_layer_names=True)
            print(f"[Model] ➜ architecture PNG saved to: {png_path}")
        except ImportError:
            print("[Model] (skipped architecture PNG – install `pydot` & GraphViz to enable)")

    def load_from_disk(self, filepath):
        self._model = tf.keras.models.load_model(filepath)

    # ---------------- properties ---------------------------------------------
    @property
    def batch_size(self): return self._batch_size
    @property
    def model(self):      return self._model

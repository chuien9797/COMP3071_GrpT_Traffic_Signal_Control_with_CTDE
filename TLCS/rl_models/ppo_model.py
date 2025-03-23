import timeit
import time
import numpy as np
import traci
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from torch.utils.tensorboard import SummaryWriter

# ================= PPO Model Definition =================
class PPOModel:
    def __init__(self, input_dim, output_dim, hidden_size=64, learning_rate=3e-4,
                 clip_ratio=0.2, update_epochs=10, batch_size=32, entropy_coef=0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.entropy_coef = entropy_coef

        self.actor, self.critic = self.build_actor_critic(hidden_size)
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)

    def build_actor_critic(self, hidden_size):
        inputs = layers.Input(shape=(self.input_dim,))
        common = layers.Dense(hidden_size, activation='relu')(inputs)

        actor = models.Model(inputs, layers.Dense(self.output_dim, activation='softmax')(layers.Dense(hidden_size, activation='relu')(common)))
        critic = models.Model(inputs, layers.Dense(1, activation='linear')(layers.Dense(hidden_size, activation='relu')(common)))

        return actor, critic

    def select_action(self, state):
        state = state.reshape(1, -1)
        probs = self.actor(state).numpy().flatten()
        action = np.random.choice(self.output_dim, p=probs)
        return action, probs[action]

    def train(self, states, actions, advantages, returns, old_probs):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)

        total_loss = 0.0
        for _ in range(self.update_epochs):
            with tf.GradientTape() as tape:
                new_probs = self.actor(states)
                masks = tf.one_hot(actions, self.output_dim)
                selected_new = tf.reduce_sum(new_probs * masks, axis=1)
                ratio = selected_new / (old_probs + 1e-10)
                clipped = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages))
                critic_loss = tf.reduce_mean(tf.square(returns - self.critic(states)))
                entropy = -tf.reduce_mean(new_probs * tf.math.log(new_probs + 1e-10))
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

            grads = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))
            total_loss = loss.numpy()

        return total_loss

# ================= PPO Simulation Loop =================
class PPOSimulation:
    def __init__(self, model, traffic_gen, sumo_cmd, gamma, max_steps,
                 green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self.model = model
        self.traffic_gen = traffic_gen
        self.sumo_cmd = sumo_cmd
        self.gamma = gamma
        self.max_steps = max_steps
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.num_states = num_states
        self.num_actions = num_actions
        self.training_epochs = training_epochs
        self.episode_rewards = []
        self.writer = SummaryWriter()

    def run_episode(self, episode):
        sim_start = timeit.default_timer()
        self.traffic_gen.generate_routefile(seed=episode)
        traci.start(self.sumo_cmd)

        state = get_state(self.num_states)
        ep_reward, steps, prev_action = 0.0, 0, None
        traj_states, traj_actions, traj_rewards = [], [], []
        old_wait = get_total_wait()

        while steps < self.max_steps:
            action, _ = self.model.select_action(state)
            switch = 1 if prev_action is not None and action != prev_action else 0
            set_green_phase(action)
            for _ in range(self.green_duration):
                traci.simulationStep()
                steps += 1

            next_state = get_state(self.num_states)
            new_wait = get_total_wait()
            queue_len = get_queue_length()
            emergency_penalty = sum(1 for v in traci.vehicle.getIDList() if traci.vehicle.getTypeID(v) == "emergency")
            reward = compute_reward(old_wait, new_wait, queue_len, switch, emergency_penalty)
            ep_reward += reward

            traj_states.append(state)
            traj_actions.append(action)
            traj_rewards.append(reward)

            state, old_wait, prev_action = next_state, new_wait, action

        traci.close()
        sim_time = round(timeit.default_timer() - sim_start, 2)
        self.episode_rewards.append(ep_reward)
        return traj_states, traj_actions, traj_rewards, ep_reward, sim_time

    def update(self, states, actions, rewards):
        returns = self.compute_returns(rewards)
        advantages = self.compute_advantages(returns, states)
        train_start = timeit.default_timer()
        loss = self.model.train(np.array(states), np.array(actions), advantages, returns, old_probs=np.ones(len(actions)))
        train_time = round(timeit.default_timer() - train_start, 2)

        # Logging
        self.writer.add_scalar("Loss/Total", loss, len(self.episode_rewards))
        return train_time

    def compute_returns(self, rewards):
        returns, G = [], 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return np.array(returns)

    def compute_advantages(self, returns, states):
        values = np.squeeze(self.model.critic.predict(np.array(states), verbose=0))
        return returns - values

# ================= Environment Utilities =================
def get_state(num_states):
    state = np.zeros(num_states)
    for car in traci.vehicle.getIDList():
        pos = 750 - traci.vehicle.getLanePosition(car)
        idx = min(int(pos // 7), 9)
        group = lane_group(traci.vehicle.getLaneID(car))
        if group >= 0:
            state[group * 10 + idx] = 1
    return state

def lane_group(lane_id):
    mapping = {"W2TL_0":0, "W2TL_1":0, "W2TL_2":0, "W2TL_3":1,
               "N2TL_0":2, "N2TL_1":2, "N2TL_2":2, "N2TL_3":3,
               "E2TL_0":4, "E2TL_1":4, "E2TL_2":4, "E2TL_3":5,
               "S2TL_0":6, "S2TL_1":6, "S2TL_2":6, "S2TL_3":7}
    return mapping.get(lane_id, -1)

def set_green_phase(action):
    traci.trafficlight.setPhase("TL", action*2)

def compute_reward(old_wait, new_wait, queue_len, switch, emergency):
    return (old_wait - new_wait) + 0.5*(old_wait - new_wait - queue_len) - 0.01*queue_len - 0.1*switch - 5*emergency

def get_total_wait():
    return sum(traci.vehicle.getAccumulatedWaitingTime(v) for v in traci.vehicle.getIDList()
               if traci.vehicle.getRoadID(v) in ["E2TL","N2TL","W2TL","S2TL"])

def get_queue_length():

    return sum(traci.edge.getLastStepHaltingNumber(edge) for edge in ["N2TL","S2TL","E2TL","W2TL"])

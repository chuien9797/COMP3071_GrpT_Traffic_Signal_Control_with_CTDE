# import timeit
# import time
# import numpy as np
# import traci
# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers
#
# # ================= PPO Model Definition =================
# class PPOModel:
#     def __init__(self, input_dim, output_dim, hidden_size=64, learning_rate=0.001,
#                  clip_ratio=0.2, update_epochs=10, batch_size=32):
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_size = hidden_size
#         self.learning_rate = learning_rate
#         self.clip_ratio = clip_ratio
#         self.update_epochs = update_epochs
#         self._batch_size = batch_size  # Added batch_size attribute
#
#         # Build the actor and critic networks
#         self.actor, self.critic = self.build_actor_critic()
#         self.optimizer = optimizers.Adam(learning_rate=self.learning_rate)
#
#     def build_actor_critic(self):
#         # Shared input layer
#         inputs = layers.Input(shape=(self.input_dim,))
#         common = layers.Dense(self.hidden_size, activation='relu')(inputs)
#
#         # Actor branch (outputs action probabilities)
#         actor_hidden = layers.Dense(self.hidden_size, activation='relu')(common)
#         action_probs = layers.Dense(self.output_dim, activation='softmax')(actor_hidden)
#         actor = models.Model(inputs=inputs, outputs=action_probs)
#
#         # Critic branch (outputs state value)
#         critic_hidden = layers.Dense(self.hidden_size, activation='relu')(common)
#         state_value = layers.Dense(1, activation='linear')(critic_hidden)
#         critic = models.Model(inputs=inputs, outputs=state_value)
#
#         return actor, critic
#
#     def select_action(self, state):
#         """
#         Given a state, sample an action according to the actor network's probabilities.
#         Returns the chosen action and its probability.
#         """
#         state = state.reshape(1, -1)
#         probs = self.actor(state).numpy().flatten()
#         action = np.random.choice(self.output_dim, p=probs)
#         return action, probs[action]
#
#     def predict_one(self, state):
#         """
#         Returns the action probability distribution for a given state.
#         This allows compatibility with simulation code that uses np.argmax on the result.
#         """
#         state = state.reshape(1, -1)
#         return self.actor(state).numpy().flatten()
#
#     def train(self, states, actions, advantages, returns, old_probs):
#         """
#         Update the actor and critic networks using the PPO clipped objective.
#         This function assumes that you've precomputed the advantages and returns.
#         """
#         # Convert inputs to tensors if they aren't already
#         states = tf.convert_to_tensor(states, dtype=tf.float32)
#         actions = tf.convert_to_tensor(actions, dtype=tf.int32)
#         advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
#         returns = tf.convert_to_tensor(returns, dtype=tf.float32)
#         old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
#
#         for _ in range(self.update_epochs):
#             with tf.GradientTape() as tape:
#                 new_probs = self.actor(states)
#                 action_masks = tf.one_hot(actions, self.output_dim)
#                 selected_new_probs = tf.reduce_sum(new_probs * action_masks, axis=1)
#                 ratio = selected_new_probs / (old_probs + 1e-10)
#                 clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
#                 actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
#                 critic_loss = tf.reduce_mean(tf.square(returns - self.critic(states)))
#                 total_loss = actor_loss + 0.5 * critic_loss
#
#             grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
#             self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))
#
#         return total_loss.numpy()
#
#     @property
#     def batch_size(self):
#         return self._batch_size
#
# # ============ Shared Environment Utility Functions ============
# # (These can be refactored into a separate file if desired.)
# def get_state(num_states):
#     """
#     Returns the state of the intersection as an occupancy grid.
#     Adjust the logic as needed.
#     """
#     state = np.zeros(num_states)
#     car_list = traci.vehicle.getIDList()
#     for car_id in car_list:
#         lane_pos = traci.vehicle.getLanePosition(car_id)
#         lane_id = traci.vehicle.getLaneID(car_id)
#         # Invert lane position (closer to 0 means near the intersection)
#         lane_pos = 750 - lane_pos
#         if lane_pos < 7:
#             lane_cell = 0
#         elif lane_pos < 14:
#             lane_cell = 1
#         elif lane_pos < 21:
#             lane_cell = 2
#         elif lane_pos < 28:
#             lane_cell = 3
#         elif lane_pos < 40:
#             lane_cell = 4
#         elif lane_pos < 60:
#             lane_cell = 5
#         elif lane_pos < 100:
#             lane_cell = 6
#         elif lane_pos < 160:
#             lane_cell = 7
#         elif lane_pos < 400:
#             lane_cell = 8
#         else:
#             lane_cell = 9
#
#         # Example mapping from lane_id to lane group (customize as needed)
#         if lane_id in ["W2TL_0", "W2TL_1", "W2TL_2"]:
#             lane_group = 0
#         elif lane_id == "W2TL_3":
#             lane_group = 1
#         elif lane_id in ["N2TL_0", "N2TL_1", "N2TL_2"]:
#             lane_group = 2
#         elif lane_id == "N2TL_3":
#             lane_group = 3
#         elif lane_id in ["E2TL_0", "E2TL_1", "E2TL_2"]:
#             lane_group = 4
#         elif lane_id == "E2TL_3":
#             lane_group = 5
#         elif lane_id in ["S2TL_0", "S2TL_1", "S2TL_2"]:
#             lane_group = 6
#         elif lane_id == "S2TL_3":
#             lane_group = 7
#         else:
#             lane_group = -1
#
#         if lane_group >= 0:
#             pos_index = lane_group * 10 + lane_cell  # Adjust factor as needed
#             if pos_index < num_states:
#                 state[pos_index] = 1
#     return state
#
# def set_green_phase(action):
#     """
#     Sets the green phase corresponding to the given action.
#     Modify phase codes as needed.
#     """
#     if action == 0:
#         traci.trafficlight.setPhase("TL", 0)  # NS_GREEN
#     elif action == 1:
#         traci.trafficlight.setPhase("TL", 2)  # NSL_GREEN
#     elif action == 2:
#         traci.trafficlight.setPhase("TL", 4)  # EW_GREEN
#     elif action == 3:
#         traci.trafficlight.setPhase("TL", 6)  # EWL_GREEN
#
# def compute_reward(old_total_wait, current_total_wait, emergency_penalty=0):
#     """
#     Computes reward as the reduction in waiting time minus any penalty.
#     """
#     return (old_total_wait - current_total_wait) - emergency_penalty
#
# def get_total_wait():
#     """
#     Sums waiting times on incoming roads.
#     """
#     incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
#     total_wait = 0
#     for veh in traci.vehicle.getIDList():
#         road_id = traci.vehicle.getRoadID(veh)
#         if road_id in incoming_roads:
#             total_wait += traci.vehicle.getAccumulatedWaitingTime(veh)
#     return total_wait
#
# # ================= PPO Simulation and Training Loop =================
# class PPOSimulation:
#     def __init__(self, model, traffic_gen, sumo_cmd, gamma, max_steps, green_duration,
#                  yellow_duration, num_states, num_actions, training_epochs):
#         self.model = model
#         self.traffic_gen = traffic_gen
#         self.sumo_cmd = sumo_cmd
#         self.gamma = gamma
#         self.max_steps = max_steps
#         self.green_duration = green_duration
#         self.yellow_duration = yellow_duration
#         self.num_states = num_states
#         self.num_actions = num_actions
#         self.training_epochs = training_epochs
#         self.episode_rewards = []  # For logging rewards
#
#     def run_episode(self, episode):
#         # Generate route file and start SUMO
#         self.traffic_gen.generate_routefile(seed=episode)
#         traci.start(self.sumo_cmd)
#         print("Simulating PPO episode:", episode)
#
#         state = get_state(self.num_states)
#         episode_reward = 0
#         steps = 0
#
#         trajectory_states = []
#         trajectory_actions = []
#         trajectory_rewards = []
#
#         old_total_wait = get_total_wait()
#         while steps < self.max_steps:
#             # Use PPO model's select_action method (stochastic sampling)
#             action, _ = self.model.select_action(state)
#             # Set the green phase using our shared utility
#             set_green_phase(action)
#             # Simulate for the duration of the green phase
#             self.simulate(self.green_duration)
#             next_state = get_state(self.num_states)
#             current_total_wait = get_total_wait()
#             reward = compute_reward(old_total_wait, current_total_wait)
#             episode_reward += reward
#
#             trajectory_states.append(state)
#             trajectory_actions.append(action)
#             trajectory_rewards.append(reward)
#
#             state = next_state
#             old_total_wait = current_total_wait
#             steps += self.green_duration
#
#         traci.close()
#         # Wait a bit to allow SUMO to shut down completely
#         import time
#         time.sleep(1)
#
#         # After the episode, compute returns and advantages
#         returns = self.compute_returns(trajectory_rewards)
#         advantages = self.compute_advantages(returns, trajectory_states)
#         # Update the policy for a number of epochs using collected trajectory data
#         for _ in range(self.training_epochs):
#             self.model.train(np.array(trajectory_states),
#                              np.array(trajectory_actions),
#                              advantages, returns,
#                              old_probs=np.ones(len(trajectory_actions)))
#         self.episode_rewards.append(episode_reward)
#         print("Episode", episode, "complete. Reward:", episode_reward)
#         return episode_reward
#
#     def simulate(self, steps_todo):
#         while steps_todo > 0:
#             traci.simulationStep()
#             steps_todo -= 1
#
#     def compute_returns(self, rewards):
#         returns = []
#         G = 0
#         for r in reversed(rewards):
#             G = r + self.gamma * G
#             returns.insert(0, G)
#         return np.array(returns)
#
#     def compute_advantages(self, returns, states):
#         values = self.model.critic.predict(np.array(states))
#         values = np.squeeze(values)
#         advantages = returns - values
#         return advantages

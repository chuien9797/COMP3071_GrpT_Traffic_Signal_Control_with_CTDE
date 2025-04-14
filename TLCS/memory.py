import random


class Memory:
    def __init__(self, size_max, size_min):
        """
        Initialize the replay memory.

        In a multi-agent scenario, each sample is stored as a tuple:
            (agent_id, state, action, reward, next_state, global_state)

        Parameters:
            size_max (int): Maximum number of samples to store.
            size_min (int): Minimum number of samples required before retrieval.
        """
        self._samples = []
        self._size_max = size_max
        self._size_min = size_min

    def add_sample(self, sample):
        """
        Add a sample into the memory.

        Parameters:
            sample (tuple): The sample to add. It is expected to be in the form:
                (agent_id, state, action, reward, next_state, global_state)
        """
        self._samples.append(sample)
        if self._size_now() > self._size_max:
            # Remove the oldest sample if capacity is exceeded.
            self._samples.pop(0)

    def get_samples(self, n):
        """
        Retrieve n random samples from the memory if there are at least the minimum required samples.

        Parameters:
            n (int): Number of samples to retrieve.

        Returns:
            list: A list of randomly selected samples. If there are fewer samples than requested,
                  returns all available samples.
        """
        if self._size_now() < self._size_min:
            return []
        available_samples = self._size_now()
        if n > available_samples:
            return random.sample(self._samples, available_samples)
        else:
            return random.sample(self._samples, n)

    def get_samples_by_agent(self, agent_id, n):
        """
        Retrieve n random samples specifically for the given agent id.

        This method encapsulates filtering of stored experiences to provide per-agent samples.

        Parameters:
            agent_id (int): The identifier of the agent for which to retrieve samples.
            n (int): The number of samples to retrieve.

        Returns:
            list: A list of randomly selected samples for that agent, or an empty list if
                  there are fewer than the minimum required samples.
        """
        agent_samples = [sample for sample in self._samples if sample[0] == agent_id]
        if len(agent_samples) < self._size_min:
            return []
        if n > len(agent_samples):
            return random.sample(agent_samples, len(agent_samples))
        else:
            return random.sample(agent_samples, n)

    def _size_now(self):
        """
        Get the current number of stored samples.

        Returns:
            int: The number of samples in memory.
        """
        return len(self._samples)

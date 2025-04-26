import random


class Memory:
    def __init__(self, size_max, size_min):

        self._samples = []
        self._size_max = size_max
        self._size_min = size_min

    def add_sample(self, sample):

        self._samples.append(sample)
        if self._size_now() > self._size_max:
            # Remove the oldest sample if capacity is exceeded.
            self._samples.pop(0)

    def get_samples(self, n):

        if self._size_now() < self._size_min:
            return []
        available_samples = self._size_now()
        if n > available_samples:
            return random.sample(self._samples, available_samples)
        else:
            return random.sample(self._samples, n)

    def get_samples_by_agent(self, agent_id, n):

        agent_samples = [sample for sample in self._samples if sample[0] == agent_id]
        if len(agent_samples) < self._size_min:
            return []
        if n > len(agent_samples):
            return random.sample(agent_samples, len(agent_samples))
        else:
            return random.sample(agent_samples, n)

    def _size_now(self):

        return len(self._samples)

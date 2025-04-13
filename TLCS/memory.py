import random


class Memory:
    def __init__(self, size_max, size_min):
        """
        Initialize the replay memory.

        This memory class is used as a shared buffer for a multi-intersection environment.
        Each sample stored is expected to be a tuple (state, action, reward, next_state)
        coming from one intersection (agent). If your simulation splits the overall state
        among several intersections, each intersection will add its own sample into the same memory.

        Parameters:
            size_max (int): Maximum number of samples to store.
            size_min (int): Minimum number of samples required before replay (experience sampling) occurs.
        """
        self._samples = []
        self._size_max = size_max
        self._size_min = size_min

    def add_sample(self, sample):
        """
        Add a sample into the memory.

        In a multi-agent (multi-intersection) scenario, each agent (or intersection)
        generates a sample (state, action, reward, next_state) that is added to the shared memory.

        Parameters:
            sample (tuple): The sample to add.
        """
        self._samples.append(sample)
        if self._size_now() > self._size_max:
            # Remove the oldest sample if the memory exceeds capacity.
            self._samples.pop(0)

    def get_samples(self, n):
        """
        Retrieve n random samples from memory.

        If the number of stored samples is less than the minimum threshold, returns an empty list.
        This function is independent of the fact that samples may originate from different intersections.

        Parameters:
            n (int): The number of samples to retrieve.

        Returns:
            list: A list of n randomly selected samples (or all samples if n is greater than the number available).
        """
        if self._size_now() < self._size_min:
            return []

        available_samples = self._size_now()
        if n > available_samples:
            return random.sample(self._samples, available_samples)
        else:
            return random.sample(self._samples, n)

    def _size_now(self):
        """
        Get the current number of stored samples.

        Returns:
            int: The number of samples in the memory.
        """
        return len(self._samples)

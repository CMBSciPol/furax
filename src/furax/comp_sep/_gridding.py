import jax
import itertools
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import interp1d
import logging
from tqdm import tqdm

logger = logging.getLogger('GRIDDING')


class DistributedGridSearch:
    def __init__(
        self,
        objective_fn,
        search_space,
        batch_size=None,
        memory_limit=0.6,
        log_every=0.1,
        progress_bar=True,
    ):
        """
        Initialize the grid search.

        Args:
            objective_fn: The objective function to be evaluated.
            search_space: A dictionary where keys are parameter names and values are lists
                of possible values.
            batch_size: The number of combinations to evaluate in each batch.
                If None, it is determined automatically.
            memory_limit: Fraction of device memory to use for determining batch size.
            verbose: Percentage (0.0 to 1.0) to control logging frequency.
                Logs every 'verbose' fraction of progress.
            use_tqdm: Whether to use tqdm for a progress bar.
        """
        keys, values = zip(*search_space.items())

        self.param_keys = keys
        self.search_space = search_space
        self.objective_fn = jax.jit(objective_fn)  # JIT compile the objective function
        # Create an iterator over all parameter combinations$
        self.combinations = list(itertools.product(*values))
        self.n_combinations = len(self.combinations)

        if self.n_combinations % jax.process_count() != 0:
            raise ValueError(
                f'Number of combinations ({self.n_combinations}) must be evenly divisible '
                f'by the number of processes ({jax.process_count()}).'
            )

        self.batch_size = batch_size
        self.log_every = log_every
        self.progress_bar = progress_bar

        # Automatically determine batch size if None
        if self.batch_size is None:
            if jax.devices()[0].platform == 'cpu':
                logger.warning("""
                Batch size not specified and automatic batch size
                determination is not supported on CPU.
                Falling back to default batch size of 64.
                """)
                self.batch_size = 64
            else:
                self.batch_size = int(self.suggest_batch_size() * memory_limit)
                print(f'Auto-determined maximum batch size: {self.batch_size}')

        if self.batch_size > (self.n_combinations // jax.process_count()):
            self.batch_size = self.n_combinations // jax.process_count()

        print(f'Selecting batch size of {self.batch_size}')

    def _measure_memory_usage(self, batch_size):
        """
        Measure memory usage of the objective function for a given batch size.

        Args:
            batch_size: The batch size to test.

        Returns:
            Estimated memory usage in bytes.
        """
        if jax.devices()[0].platform == 'cpu':
            raise ValueError('Memory measurement is not supported on CPU platform.')

        param_sample = {
            key: np.array([val[0]] * batch_size) for key, val in self.search_space.items()
        }

        # Analyze memory usage
        mem_analysis = (
            jax.jit(jax.vmap(self.objective_fn)).lower(**param_sample).compile().memory_analysis()
        )
        assert mem_analysis is not None

        return (
            mem_analysis.argument_size_in_bytes
            + mem_analysis.output_size_in_bytes
            + mem_analysis.temp_size_in_bytes
        )

    def suggest_batch_size(self):
        """
        Estimate the largest feasible batch size based on device memory constraints.

        Returns:
            The estimated maximum batch size.
        """
        if jax.devices()[0].platform == 'cpu':
            raise ValueError('Memory measurement is not supported on CPU platform.')

        memory_stats = jax.devices()[0].memory_stats()
        assert memory_stats is not None

        max_device_memory = memory_stats['bytes_limit'] - memory_stats['bytes_in_use']

        # Measure memory usage for progressively larger batch sizes
        test_batch_sizes = [2, 4, 8, 16, 32]
        memory_usages = []
        for batch_size in test_batch_sizes:
            try:
                memory_usages.append(self._measure_memory_usage(batch_size))
            except Exception as e:
                print(f'Error measuring memory for batch size {batch_size}: {e}')
                break

        # Ensure we have valid data points
        if len(memory_usages) < 2:
            raise ValueError('Not enough data points to interpolate memory usage.')

        # Interpolate to predict the maximum batch size
        interpolator = interp1d(
            memory_usages,
            test_batch_sizes[: len(memory_usages)],
            kind='linear',
            fill_value='extrapolate',
        )
        max_batch_size = int(interpolator(max_device_memory))
        return max_batch_size

    def _batch_generator(self, indx=0, size=1):
        """Generates batches of parameter combinations."""
        current_slice_combinations = self.combinations[
            indx * self.n_combinations // size : (indx + 1) * self.n_combinations // size
        ]
        n_batches = len(current_slice_combinations) // self.batch_size

        for i in range(n_batches):
            yield current_slice_combinations[i * self.batch_size : (i + 1) * self.batch_size]

    def run(self):
        """
        Run the grid search.

        Returns:
            A dict of arrays with parameter names and corresponding objective function values, sorted by value.
        """
        # Prepare a list to hold batches of results
        batch_results = {key: [] for key in self.param_keys}
        batch_results['value'] = []

        rank = jax.process_index()
        size = jax.process_count()
        assert self.batch_size is not None
        total_batches = len(self.combinations) // (self.batch_size * size)
        log_interval = max(1, int(self.log_every * total_batches)) if self.log_every > 0 else 0
        print(f'log_interval: {log_interval}')

        progress_bar = None
        if self.progress_bar:
            progress_bar = tqdm(total=total_batches, desc='Processing batches')

        for batch_idx, batch in enumerate(self._batch_generator(rank, size)):
            param_dicts = [dict(zip(self.param_keys, combo)) for combo in batch]
            param_arrays = {
                key: jnp.array([d[key] for d in param_dicts]) for key in self.param_keys
            }

            # Apply the objective function
            values = jax.vmap(lambda **kwargs: self.objective_fn(**kwargs))(**param_arrays)

            # Store the results as arrays
            for i, param_dict in enumerate(param_dicts):
                for key in param_dict:
                    batch_results[key].append(param_dict[key])
                batch_results['value'].append(values[i])

            # Log progress if verbose
            if self.log_every > 0:
                assert log_interval > 0
                if (batch_idx + 1) % log_interval == 0:
                    logger.info(f'Rank {rank}: Processed {batch_idx + 1}/{total_batches} batches.')

            # Update tqdm progress bar if enabled
            if progress_bar:
                progress_bar.update(1)

        if progress_bar:
            progress_bar.close()

        print('Done .. Stacking the results')

        # Stack the results into arrays along the new axis
        stacked_results = {key: jnp.stack(value, axis=0) for key, value in batch_results.items()}

        # Optional: Sort the results based on the "value"
        sorted_results = {
            key: value[stacked_results['value'].argsort()] for key, value in stacked_results.items()
        }

        return sorted_results

from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np

from averager import AllreduceAverager, GossipAverager, GradientPushAverager, MoshpitAverager, RandomGroupAverager


def run_iterative_averaging(seed, num_nodes, failure_prob, max_iterations, averaging_algo, target_precision=None):
    np.random.seed(seed)

    weights = np.random.normal(0, 1, num_nodes).astype(np.float64)
    history = np.zeros((max_iterations + 1,), dtype=np.float64)

    iter_num = 0
    history[iter_num] = cur_precision = weights.var()

    while iter_num < max_iterations and (target_precision is None or cur_precision >= target_precision):
        iter_num += 1

        averaging_nodes = np.random.uniform(0, 1, num_nodes) >= failure_prob
        weights = averaging_algo(weights, averaging_nodes, iter_num)
        assert weights.size == num_nodes

        history[iter_num] = cur_precision = weights.var()
    return history, iter_num


def run_experiment(num_nodes, failure_prob, max_iterations, averaging_algo, num_restarts, target_precision=None):
    experiment_result = partial(run_iterative_averaging, num_nodes=num_nodes, failure_prob=failure_prob,
                                max_iterations=max_iterations, averaging_algo=averaging_algo,
                                target_precision=target_precision)

    with Pool(processes=min(cpu_count(), num_restarts)) as pool:
        results = pool.map(experiment_result, range(num_restarts))

    trajectories, finish_iters = zip(*results)
    stacked_trajectories = np.maximum(np.stack(trajectories, axis=0), np.finfo(np.float64).eps)
    avg_iters = np.mean(np.stack(finish_iters))

    avg_trajectories = np.mean(stacked_trajectories, axis=0)
    return avg_trajectories, avg_iters


def run_experiment_setup(num_nodes, failure_prob, max_iterations, num_restarts, moshpit_args, target_precision=None):
    allreduce = AllreduceAverager()

    gossip = GossipAverager(num_nodes, num_surrounding_neighbors=1)
    sgp = GradientPushAverager(num_nodes, num_outgoing_edges=2)

    grid_size, grid_dims = moshpit_args
    group_averager = RandomGroupAverager(group_size=grid_size)
    moshpit = MoshpitAverager(num_nodes=num_nodes, grid_dims=grid_dims, grid_size=grid_size)

    algos = [allreduce, gossip, sgp, group_averager, moshpit]

    results = [run_experiment(num_nodes=num_nodes, failure_prob=failure_prob, max_iterations=max_iterations,
                              averaging_algo=algo, num_restarts=num_restarts, target_precision=target_precision) for
               algo in algos]
    return results

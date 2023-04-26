import numpy as np


class AllreduceAverager:
    def __init__(self):
        pass

    def __call__(self, weights, alive_mask, iter_num):
        if np.all(alive_mask):
            return np.repeat(weights.mean(), weights.size)
        return weights


class GossipAverager:
    def __init__(self, num_nodes, num_surrounding_neighbors):
        averaging_diag = np.full(num_nodes, 1 / num_surrounding_neighbors)
        diag = np.diag(averaging_diag)

        self.mixing_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        self.mixing_matrix += diag
        for shift in range(1, num_surrounding_neighbors + 1):
            self.mixing_matrix += np.roll(diag, shift, axis=1)
            self.mixing_matrix += np.roll(diag, -shift, axis=1)

    def __call__(self, weights, alive_mask, iter_num):
        matrix_for_iter = self.mixing_matrix.copy()

        # don't receive weights from failed peers
        matrix_for_iter[:, ~alive_mask] = 0

        # failed peers do not average their weights
        matrix_for_iter[~alive_mask, :] = 0

        sum_for_rows = np.sum(matrix_for_iter, axis=1, keepdims=True)
        np.divide(matrix_for_iter, sum_for_rows, where=sum_for_rows != 0, out=matrix_for_iter)

        return weights * (1 - alive_mask) + matrix_for_iter @ weights


class GradientPushAverager:
    def __init__(self, num_nodes, num_outgoing_edges):
        averaging_diag = np.full(num_nodes, 1 / num_outgoing_edges)
        diag = np.diag(averaging_diag)

        self.mixing_matrices = []

        for hop_power in range(int(np.floor(np.log2(num_nodes - 1)))):
            mixing_matrix = diag.copy()
            for shift in range(num_outgoing_edges):
                mixing_matrix += np.roll(diag, 2 ** (hop_power + shift))

            self.mixing_matrices.append(mixing_matrix)

    def __call__(self, weights, alive_mask, iter_num):
        matrix_for_iter = self.mixing_matrices[iter_num % len(self.mixing_matrices)].copy()

        # don't receive weights from failed peers
        matrix_for_iter[:, ~alive_mask] = 0

        # failed peers do not average their weights
        matrix_for_iter[~alive_mask, :] = 0

        sum_for_rows = np.sum(matrix_for_iter, axis=1, keepdims=True)
        np.divide(matrix_for_iter, sum_for_rows, where=sum_for_rows != 0, out=matrix_for_iter)

        return weights * (1 - alive_mask) + matrix_for_iter @ weights


class RandomGroupAverager:
    def __init__(self, group_size):
        self.group_size = group_size

    def __call__(self, weights, alive_mask, iter_num):
        averaging_inds = np.nonzero(alive_mask)

        group_count = int(np.floor(averaging_inds[0].size / self.group_size))

        np.random.shuffle(averaging_inds[0])
        groups = [averaging_inds[0][i * self.group_size:(i + 1) * self.group_size] for i in range(group_count)]

        for i in range(self.group_size * group_count, averaging_inds[0].size):
            groups[i % len(groups)] = np.append(groups[i % len(groups)], [averaging_inds[0][i]])

        new_weights = weights.copy()

        for group in groups:
            new_weights[group] = weights[group].mean()
        return new_weights


class OtherRandomGroupAverager:
    def __init__(self, num_nodes, group_size):
        self.num_nodes = num_nodes
        self.group_size = group_size
        self.group_count = int(np.floor(num_nodes / self.group_size))

    def __call__(self, weights, alive_mask, iter_num):
        averaging_inds = np.arange(self.num_nodes)
        np.random.shuffle(averaging_inds)

        groups = [averaging_inds[i * self.group_size:(i + 1) * self.group_size] for i in range(self.group_count + 1)]

        alive_mask = np.concatenate(
            [alive_mask, np.zeros_like(alive_mask, shape=(len(averaging_inds) - len(alive_mask)))])

        masked_groups = [group[alive_mask[averaging_inds][i * self.group_size:(i + 1) * self.group_size]]
                         for i, group in enumerate(groups)]

        new_weights = weights.copy()

        for group in masked_groups:
            if group.size > 1:
                new_weights[group] = weights[group].mean()
        return new_weights


class MoshpitAverager:
    def __init__(self, num_nodes, grid_dims, grid_size):
        self.chosen_inds = np.random.choice(grid_size ** grid_dims, size=num_nodes, replace=False)
        self.grid_dims = grid_dims
        self.grid_size = grid_size

        self.grid_shape = (self.grid_size,) * self.grid_dims

    def __call__(self, weights, alive_mask, iter_num):
        # find out the axis for averaging
        axis_to_average = iter_num % self.grid_dims

        # build a structure for averaged weights
        averaging_inds = np.nonzero(alive_mask)
        averaged_weights = weights[averaging_inds]
        weights_grid = np.full(self.grid_shape, fill_value=np.nan, dtype=np.float64)

        # fill and average
        alive_inds_in_grid = np.unravel_index(self.chosen_inds[averaging_inds], shape=self.grid_shape)
        weights_grid[alive_inds_in_grid] = averaged_weights
        weights_grid[:] = np.nanmean(weights_grid, axis=axis_to_average, keepdims=True)

        # build the result (keep non-alive, replace averaged with result)
        final_weights = weights * (1 - alive_mask)
        final_weights[averaging_inds] = weights_grid[alive_inds_in_grid]
        return final_weights

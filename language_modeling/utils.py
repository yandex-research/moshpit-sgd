from dataclasses import dataclass
from itertools import chain
from typing import Sequence, Dict

import torch
import hivemind
from hivemind.utils import nested_flatten, nested_pack
from transformers import Trainer


class SimpleAverager(hivemind.DecentralizedAverager):
    """ A daemon that runs decentralized averaging of model parameters and gradients """
    def __init__(self, trainer: Trainer, **kwargs):
        self.trainer = trainer  # note: this is a cyclic reference, we should to un-cycle it
        initialize_optimizer_state(self.trainer.optimizer)

        # averaged tensors are: (*model parameters, *model gradients)
        averaged_tensors = tuple(param.detach().cpu().float().clone() for param in self.trainer.model.parameters())
        averaged_tensors += tuple(torch.zeros_like(tensor) for tensor in averaged_tensors)

        super().__init__(averaged_tensors=averaged_tensors, **kwargs)

    def get_current_state(self):
        """
        Get current model/optimizer state and when requested by a newbie peer. executed in the host process.
        :returns: a tuple of (serializable_small_metadata, sequence of torch tensors)
        """
        with torch.no_grad():
            model_parameters = [x.cpu() for x in self.trainer.model.parameters()]
            optimizer_metadata, optimizer_tensors = dump_optimizer_state(self.trainer.optimizer)

        metadata = dict(step=self.trainer.state.global_step, group_bits=self.get_group_bits(),
                        optimizer_metadata=optimizer_metadata)
        return metadata, list(chain(model_parameters, optimizer_tensors))

    def load_state_from_peers(self, **kwargs):
        """ Attempt to download the latest optimizer state from peers and update trainer parameters/statistics """
        loadad_state = super().load_state_from_peers(**kwargs)
        if loadad_state is None:
            return

        metadata, flat_tensors = loadad_state
        num_params = len(list(self.trainer.model.parameters()))
        model_parameters, opt_tensors = flat_tensors[:num_params], flat_tensors[num_params:]
        with torch.no_grad():
            for local_param, loaded_param in zip(self.trainer.model.parameters(), model_parameters):
                local_param[...] = loaded_param
            load_optimizer_state(self.trainer.optimizer, metadata['optimizer_metadata'], opt_tensors)

        collaboration_step = metadata['step']
        while self.trainer.state.global_step < collaboration_step:
            self.trainer.state.global_step += 1
            self.trainer.lr_scheduler.step()


def initialize_optimizer_state(optimizer: torch.optim.Optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.grad is None:
                (0 * param.sum()).backward()
    optimizer.step()


def dump_optimizer_state(optimizer: torch.optim.Optimizer):
    with torch.no_grad():
        flat_metadata, flat_tensors = [], []
        for elem in nested_flatten(optimizer.state_dict()):
            if isinstance(elem, torch.Tensor):
                flat_metadata.append(dict(type='tensor', index=len(flat_tensors)))
                flat_tensors.append(elem.cpu())
            else:
                flat_metadata.append(dict(type='value', value=elem))
        return flat_metadata, flat_tensors


def load_optimizer_state(optimizer: torch.optim.Optimizer, flat_metadata: Dict, flat_tensors: Sequence[torch.Tensor]):
    flat_optimizer_state = []
    for elem in flat_metadata:
        if elem.get('type') == 'tensor' and isinstance(elem.get('index'), int):
            flat_optimizer_state.append(flat_tensors[elem['index']])
        elif elem.get('type') == 'value' and 'value' in elem:
            flat_optimizer_state.append(elem['value'])
    with torch.no_grad():
        return optimizer.load_state_dict(nested_pack(flat_optimizer_state, structure=optimizer.state_dict()))


class PerformanceEMA:
    """
    A running estimate of performance (operations/sec) using adjusted exponential moving average
    :param alpha: Smoothing factor in range [0, 1], [default: 0.1].
    """
    def __init__(self, alpha: float = 0.1, eps: float = 1e-20):
        self.alpha, self.eps, self.num_updates = alpha, eps, 0
        self.ema_seconds_per_sample, self.samples_per_second = 0, eps
        self.timestamp = hivemind.get_dht_time()

    def update(self, num_processed: int) -> float:
        """
        :param num_processed: how many items were processed since last call
        :returns: current estimate of performance (samples per second), but at most
        """
        assert num_processed > 0, f"Can't register processing {num_processed} samples"
        self.timestamp, old_timestamp = hivemind.get_dht_time(), self.timestamp
        seconds_per_sample = max(0, self.timestamp - old_timestamp) / num_processed
        self.ema_seconds_per_sample = self.alpha * seconds_per_sample + (1 - self.alpha) * self.ema_seconds_per_sample
        self.num_updates += 1
        adjusted_seconds_per_sample = self.ema_seconds_per_sample / (1 - (1 - self.alpha) ** self.num_updates)
        self.samples_per_second = 1 / max(adjusted_seconds_per_sample, self.eps)
        return self.samples_per_second




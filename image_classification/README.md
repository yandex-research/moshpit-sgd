The code is a modified copy of [Stochastic Gradient Push](https://github.com/facebookresearch/stochastic_gradient_push)
that implements Moshpit SGD as an option. All rights to the original code belong to the respective authors.

The trainer node can be launched via the following script:

```
python -u gossip_sgd.py \
    --batch_size 256 --lr 0.1 --num_dataloader_workers 8 \
    --num_epochs 90 --nesterov True --warmup True --moshpit True \
    --initial_peers DHT_COORDINATOR_ENDPOINT --target_group_size XXX --initial_group_bits YYY \
    --train_fast False --tag 'moshpit' --print_freq 100 --verbose False \
    --seed 1337 --schedule 30 0.1 60 0.1 80 0.1
```

The coordinator node (`DHT_COORDINATOR_ENDPOINT`) should point to a running DHT node:
either another trainer (`gossip_sgd.py`) or a dedicated starting peer (`../albert/run_first_peer.py`).

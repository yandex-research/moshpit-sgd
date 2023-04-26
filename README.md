# Moshpit SGD: Communication-Efficient Decentralized Training on Heterogeneous Unreliable Devices

![Illustration of Moshpit SGD](scheme.png)

This repository contains the official PyTorch implementation of experiments for
the NeurIPS 2021 paper ["Moshpit SGD: Communication-Efficient Decentralized Training on Heterogeneous Unreliable Devices"](https://arxiv.org/abs/2103.03239).

## Setup

To launch the code in this repository, you will need Python 3.8+ and PyTorch 1.7. Also, install the dependencies by
running `pip install -r requirements.txt`.

## Experiments

The links below contain the implementations of experiments in their respective directories:

* [Averaging quality](https://github.com/yandex-research/moshpit-sgd/blob/main/averaging_experiments/plot_results.ipynb)
* [Language modeling](https://github.com/yandex-research/moshpit-sgd/tree/main/language_modeling)

## References

```
@inproceedings{ryabinin2021moshpit,
 author = {Ryabinin, Max and Gorbunov, Eduard and Plokhotnyuk, Vsevolod and Pekhimenko, Gennady},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {18195--18211},
 publisher = {Curran Associates, Inc.},
 title = {Moshpit SGD: Communication-Efficient Decentralized Training on Heterogeneous Unreliable Devices},
 url = {https://proceedings.neurips.cc/paper_files/paper/2021/file/97275a23ca44226c9964043c8462be96-Paper.pdf},
 volume = {34},
 year = {2021}
}
```


This repository contains the code and data to reproduce the results from the paper:

**Strutz, D., Kiers, T., & Curtis, A. (2025).** *Single and Multi-Objective Optimization of Distributed Acoustic Sensing Cable Layouts for Geophysical Applications.* arXiv preprint arXiv:2510.07531. [https://arxiv.org/abs/2510.07531](https://arxiv.org/abs/2510.07531)

### Installation

The simplest way to install all dependencies is with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

You can also install the dependencies manually, for example in a virtual environment and install the required `dased` library using the following command (requires `pip`):

```bash
pip install git+https://github.com/dominik-strutz/dased.git@2025_10_DASED_Paper
```

The `dased` library source code and documentation can be found here: [dominik-strutz/dased](https://github.com/dominik-strutz/dased)

---
For further details, see the paper and the notebooks in this repository.

# 🏛️ A Professional Federated Learning Framework

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![PyYAML](https://img.shields.io/badge/PyYAML-6.0-red.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

This repository provides an advanced, object-oriented, and extensible framework for implementing and experimenting with Federated Learning (FL) algorithms in PyTorch. It is designed with professional software engineering principles to serve as a robust foundation for research and development.

## ✨ Core Design Principles

This framework is built upon several key design patterns to ensure scalability and maintainability:

1.  **Separation of Concerns**: Cleanly divided modules for data loading, model definition, client logic, server orchestration, and optimizers.

2. **Multi-Algorithm Support**: Switch between FedAvg, FedDANE, FedProx, and FedSGD with a single CLI flag.

3. **Reproducibility**: Fixed random seeds and configurable hyperparameters via a dataclass.

4. **Extensibility**: Add new architectures or optimizers.

## 📂 Framework Structure

The framework is organized for maximum clarity and extensibility:


```
federated_learning/
├── config.py           # Hyperparameters & experiment settings
├── data_loader.py      # MNIST download & client partitioning
├── model.py            # CNN architecture (BatchNorm, Dropout)
├── utils.py            # Model & gradient averaging helpers
├── optimizers.py       # Implementations of FedAvg, FedDANE, FedProx, FedSGD
├── client.py           # Encapsulates client-side training logic
├── server.py           # Federated learning orchestration
└── main.py             # CLI entry point to run experiments           
```


## 🛠️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/frezazadeh/federated-learning.git
    cd federated-learning
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run experiments via the command line using `main.py`:

```bash
python main.py --algo <algorithm_name>
```

### Selecting an Algorithm

Available options for `--algo`:
- `fedavg`
- `feddane`
- `fedprox`
- `fedsgd`

### Dataset Loading Behavior

Training automatically combines all supported datasets detected under `data/MNIST/raw`:
- Medical datasets (HeadCT, ChestCT, AbdomenCT, BreastMRI, Hand)
- CIFAR-10 (`cifar-10-batches-py`)
- CIFAR-100 (`cifar-100-python`)

If CIFAR folders are present, they are included automatically in training and class mapping.

## Algorithms

- **FedAvg:** Clients perform multiple local SGD epochs and the server averages model parameters.  
- **FedProx:** Adds a proximal term µ‖w−w_global‖² to local objectives to limit divergence.  
- **FedDANE:** Combines a Newton-like correction with a proximal term for faster convergence.  
- **FedSGD:** Clients compute gradients on their full local data and the server applies the averaged gradient once per round.

For details, see the respective classes in [`optimizers.py`](optimizers.py) and the orchestration logic in [`server.py`](server.py).


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


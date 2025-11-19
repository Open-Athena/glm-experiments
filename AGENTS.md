# AGENTS.md - Project Context for AI Assistants

This document provides essential context about the GLM Experiments project for AI assistants working on this codebase.

## Project Overview

This is a deep learning research project built on PyTorch Lightning and Hydra. It provides a clean, configurable framework for running machine learning experiments with minimal boilerplate.

## Key Technologies

- **PyTorch Lightning**: Framework for organizing PyTorch code, handles training loops, distributed training, etc.
- **Hydra**: Configuration management framework that enables dynamic hierarchical configuration composition
- **Python 3.13**: Required Python version
- **uv**: Modern Python package manager (replaces conda/pip for this project)

## Project Structure

```
├── configs/              # Hydra configuration files
│   ├── callbacks/        # Callback configs (checkpointing, early stopping, etc.)
│   ├── data/            # Data/datamodule configs
│   ├── debug/           # Debugging configs
│   ├── experiment/      # Experiment-specific configs (version controlled hyperparameters)
│   ├── hparams_search/  # Hyperparameter search configs (Optuna)
│   ├── hydra/           # Hydra framework configs
│   ├── local/            # Local machine-specific configs (gitignored)
│   ├── logger/           # Logger configs (wandb, tensorboard, etc.)
│   ├── model/           # Model configs
│   ├── paths/            # Path configuration
│   ├── trainer/          # Trainer configs (CPU, GPU, DDP, etc.)
│   ├── eval.yaml        # Main evaluation config
│   └── train.yaml       # Main training config
│
├── glm_experiments/      # Main package (renamed from src/)
│   ├── data/            # Data modules
│   ├── models/          # Lightning modules and model components
│   ├── utils/           # Utility functions
│   ├── eval.py          # Evaluation entry point
│   └── train.py         # Training entry point
│
├── tests/               # Test suite
├── logs/                # Generated logs (gitignored)
├── data/                # Data directory (gitignored)
└── notebooks/           # Jupyter notebooks
```

## Configuration System (Hydra)

### How It Works

All PyTorch Lightning modules are dynamically instantiated from module paths specified in config files. Example:

```yaml
# configs/model/mnist.yaml
_target_: glm_experiments.models.mnist_module.MNISTLitModule
lr: 0.001
net:
  _target_: glm_experiments.models.components.simple_dense_net.SimpleDenseNet
  input_size: 784
```

Using this config, objects are instantiated with:
```python
model = hydra.utils.instantiate(config.model)
```

### Main Config (`configs/train.yaml`)

The main config determines default training configuration. It uses Hydra's `defaults` to compose configs:
- Order matters: later configs override earlier ones
- Can be overridden from command line
- Supports experiment configs for version-controlled hyperparameters

### Experiment Configs

Located in `configs/experiment/`, these allow version control of specific hyperparameters for model/dataset combinations.

Example usage:
```bash
python glm_experiments/train.py experiment=example
```

### Command Line Overrides

Any config parameter can be overridden from command line:
```bash
python glm_experiments/train.py trainer.max_epochs=20 model.optimizer.lr=1e-4
```

Add new parameters with `+`:
```bash
python glm_experiments/train.py +model.new_param="value"
```

## Common Workflows

### Training

```bash
# CPU training
python glm_experiments/train.py trainer=cpu

# GPU training
python glm_experiments/train.py trainer=gpu

# Multi-GPU DDP
python glm_experiments/train.py trainer=ddp trainer.devices=4

# With experiment config
python glm_experiments/train.py experiment=example

# Override parameters
python glm_experiments/train.py trainer.max_epochs=20 data.batch_size=64
```

### Evaluation

```bash
python glm_experiments/eval.py ckpt_path="/path/to/checkpoint.ckpt"
```

### Debugging

```bash
# Fast dev run (1 epoch, debug-friendly config)
python glm_experiments/train.py debug=default

# Fast dev run with 1 batch
python glm_experiments/train.py debug=fdr

# Profiling
python glm_experiments/train.py debug=profiler

# Overfit to 1 batch
python glm_experiments/train.py debug=overfit
```

### Hyperparameter Search

```bash
# Grid search
python glm_experiments/train.py -m data.batch_size=32,64,128 model.lr=0.001,0.0005

# Optuna search
python glm_experiments/train.py -m hparams_search=mnist_optuna experiment=example
```

### Multiple Seeds

```bash
python glm_experiments/train.py -m seed=1,2,3,4,5 trainer.deterministic=True logger=csv tags=["benchmark"]
```

## Adding New Components

### Adding a New Model

1. Create model file: `glm_experiments/models/my_model.py`
2. Create config: `configs/model/my_model.yaml`
3. Use in training: `python glm_experiments/train.py model=my_model`

Model config should specify:
- `_target_`: Full module path to LightningModule class
- Model hyperparameters
- Optimizer and scheduler configs

### Adding a New Datamodule

1. Create datamodule: `glm_experiments/data/my_datamodule.py`
2. Create config: `configs/data/my_datamodule.yaml`
3. Use in training: `python glm_experiments/train.py data=my_datamodule`

### Adding a New Experiment Config

1. Create config: `configs/experiment/my_experiment.yaml`
2. Use: `python glm_experiments/train.py experiment=my_experiment`

Experiment configs should:
- Use `@package _global_` at the top
- Override defaults as needed
- Specify tags for identification

## Logging and Experiment Tracking

### Supported Loggers

- Tensorboard (`logger=tensorboard`)
- Weights & Biases (`logger=wandb`)
- MLFlow (`logger=mlflow`)
- Neptune (`logger=neptune`)
- Comet (`logger=comet`)
- CSV (`logger=csv`)
- Multiple loggers simultaneously (see `configs/logger/many_loggers.yaml`)

### Log Structure

Hydra creates new output directory for every run:
```
logs/
├── train/
│   ├── runs/
│   │   └── YYYY-MM-DD_HH-MM-SS/
│   │       ├── .hydra/          # Hydra logs
│   │       ├── checkpoints/     # Model checkpoints
│   │       ├── wandb/           # Logger outputs
│   │       └── ...
│   └── multiruns/               # Multi-run outputs
└── debugs/                      # Debug run outputs
```

### Logging Metrics

In LightningModule, log metrics with `/` separator for organization:
```python
self.log("train/loss", loss)
self.log("val/acc", accuracy)
```

## Testing

Tests are in `tests/` directory using pytest. Run with:
```bash
# All tests
pytest

# Exclude slow tests
pytest -k "not slow"

# Specific test file
pytest tests/test_train.py
```

Test fixtures are configured in `tests/conftest.py` to use minimal configs for speed.

## Important Patterns

### Task Wrapper

The `@task_wrapper` decorator handles:
- Exception logging
- Wandb cleanup
- Output directory logging

All main training/evaluation functions use this decorator.

### Root Utils

`rootutils.setup_root()` is called at the start of `train.py` and `eval.py` to:
- Add project root to PYTHONPATH
- Set PROJECT_ROOT environment variable
- Load `.env` file if present

### Config Instantiation

Use `hydra.utils.instantiate()` for all dynamic object creation:
- Models
- Datamodules
- Callbacks
- Loggers
- Trainers

### Tags

Experiments should be tagged for identification:
```bash
python glm_experiments/train.py tags=["experiment_name", "baseline"]
```

Tags are enforced if `extras.enforce_tags=True` (prompts user if missing).

## Best Practices

1. **Use torchmetrics**: For proper metric calculation, especially in multi-GPU setups
2. **Name metrics with `/`**: Helps organize in logger UIs
3. **Version control experiments**: Use experiment configs for hyperparameters
4. **Tag experiments**: Makes filtering and comparison easier
5. **Use dependency-groups**: Dev dependencies go in `[dependency-groups]` not `[project.optional-dependencies]`
6. **Python 3.13**: Project requires Python 3.13+
7. **uv for dependencies**: Use `uv sync` and `uv sync --group dev` for installation

## Package Structure

- Package name: `glm-experiments` (hyphenated for PyPI)
- Package directory: `glm_experiments/` (underscore for Python imports)
- Entry points: `train_command` and `eval_command` defined in `pyproject.toml`

## Development Tools

- **Pre-commit**: Code formatting, linting, type checking
- **pytest**: Testing framework
- **Makefile**: Common commands (`make train`, `make test`, etc.)

## Environment Variables

- `PROJECT_ROOT`: Set by rootutils, used in path configs
- `.env` file: Loaded automatically, use for private keys/paths (gitignored)

## Common Issues

1. **Import errors**: Make sure `rootutils.setup_root()` is called before importing local modules
2. **Config not found**: Check `_target_` paths match actual module structure
3. **DDP issues**: See known issues in template (some DDP modes may have problems)
4. **Path issues**: Use `${paths.root_dir}` or `${oc.env:PROJECT_ROOT}` in configs

## Modernization Notes

This project was modernized from a 2023 template:
- Replaced conda with `uv` for dependency management
- Converted to proper Python package with `pyproject.toml`
- Uses `dependency-groups` instead of deprecated `optional-dependencies`
- Python 3.13 requirement
- Package renamed from `src/` to `glm_experiments/`

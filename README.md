# **DynaMix: True Zero-Shot Inference of Dynamical Systems Preserving Long-Term Statistics (NeurIPS 2025)**

This repository provides a Python implementation of DynaMix, a foundation model architecture for zero-shot inference of dynamical systems that preserves long-term statistics.
Note there is also a julia version available at [https://github.com/DurstewitzLab/DynaMix-julia](https://github.com/DurstewitzLab/DynaMix-julia) (which was used for training the pretrained models).

## Theoretical Background

Dynamical systems are ubiquitous in nature and society, from weather patterns and climate to financial markets and biological systems. While traditional forecasting methods require extensive training data for each specific system, DynaMix takes a fundamentally different approach by learning universal dynamical representations that generalize across systems.

### Foundation Model for Dynamical Systems

DynaMix serves as a foundation model for dynamical systems. By training on a diverse collection of dynamical systems, DynaMix learns to extract and represent the underlying patterns and principles that govern temporal evolution across different domains. Unlike traditional methods that require retraining for each new system, DynaMix achieves zero-shot inference of a new system by only providing a short context window of observations.

### Key Innovations
- **Accurate Zero-Shot DSR**: DynaMix generalizes across diverse dynamical systems without fine-tuning, accurately capturing attractor geometry and long-term statistics.
- **Multivariate Dynamics Modeling**: The multivariate architecture captures dependencies across system dimensions and adapts flexibly to different dimensionalities and context lengths via embeddings.
- **Efficient and Lightweight**: Delivers high performance with a compact design, enabling orders-of-magnitude faster inference than traditional foundation models.
- **Interpretable Dynamics Composition**: Provides insights into the dynamical composition of reconstructed systems, revealing similarities across different dynamical systems.
- **General Time Series Forecasting**: Extends beyond DSR to general time series forecasting using adaptable embedding techniques.

![DynaMix Introduction](figures/Intro.png)

## DynaMix Demo
Try DynaMix instantly through our interactive Huggingface Space at [https://huggingface.co/spaces/DurstewitzLab/DynaMix](https://huggingface.co/spaces/DurstewitzLab/DynaMix). This demo lets you test the model's capabilities without installing any code or dependencies. Simply upload your time series data in the required format and generate long-term forecasts with zero-shot inference.

## Code Setup and Usage
The project is implemented in Python using PyTorch.

### Installation
Clone the repository and install the package with dependencies:

```bash
git clone https://github.com/DurstewitzLab/DynaMix-python
cd DynaMix-python
pip install -e .
```

## DynaMix Model Architecture

DynaMix is based on a sparse mixture of experts (MoE) architecture operating in latent space:

1. **Expert Networks**: Each expert is a specialized dynamical model (AL-RNN), given through

$$ z_{t+1}^j= A^jz_t^j + W^j \Phi^*(z_t^j) +h^j $$

2. **Gating Network**: Selects experts based on the provided context and current latent representation of the dynamics

By aggregating the expert weighting $w_{j,t}^{exp}$ with the expert prediction $z_t^i$ the next state is predicted

$$z_{t+1} = \sum_{j=1}^J w_{j,t}^{exp} z_t^i$$

![DynaMix Architecture](figures/architecture.png)

Model implementations:
- DynaMix → [`model/dynamix.py`](src/dynamix/model/dynamix.py), individual specifications for the architecture can be modified in the [`settings`](src/dynamix/training/settings/defaults.json) file.

## Evaluating DynaMix

### Zero-Shot Forecasting

DynaMix enables true zero-shot forecasting - the ability to generate accurate predictions for previously unseen dynamical systems without retraining or fine-tuning:

1. A short context window from the new system is provided
2. DynaMix infers the underlying dynamics from this context
3. The model generates long-term forecasts that preserve both trajectory accuracy and statistical properties

### Pretrained Model
Pre-trained DynaMix models can be accessed via [Huggingface](https://huggingface.co/DurstewitzLab/dynamix). The model has been trained on a diverse collection of dynamical systems and serves as a foundation model for zero-shot forecasting tasks.

To use a pretrained model, load it via:
```python
from dynamix.utilities.utilities import load_hf_model

# Load the pre-trained model
model = load_hf_model("dynamix-3d-alrnn-v1.0")

# Set model to evaluation mode
model.eval()
```

To successfully load a model, the name should match the model name in the Huggingface repository.


### Evaluation Pipeline

Example forecasting evaluations of dynamical systems and time series can be found in the [notebooks](notebooks/) folder. 

Given context data from the target system with shape ($T_C$, $S$, $N$) or ($T_C$, $N$) (where $T_C$ is the context length, $S$ the number of sequences that should get processed and $N$ the data dimensionality), generate forecasts by using the `DynaMixForecaster` class:

```python
import torch
from dynamix.model.forecaster import DynaMixForecaster

# Create a forecaster with the trained model
forecaster = DynaMixForecaster(model)

# Make prediction
with torch.no_grad():  # No gradient tracking needed for inference
    reconstruction = forecaster.forecast(
        context=context_tensor,
        horizon=prediction_length,
        preprocessing_method="delay_embedding",
        standardize=True,
        fit_nonstationary=False,
        initial_x=None
    )
```

The forecasting method requires the following inputs:

- *context*: Context data in the form of a tensor with shape ($T_C$, $S$, $N$) or ($T_C$, $N$)
- *horizon*: Forecast horizon, i.e. an integer specifying how many future steps to forecast

Optional arguments:
- *preprocessing_method*: for time series forecasting, choose between `pos_embedding`, `delay_embedding`, `delay_embedding_random` and `zero_embedding` as preprocessing method (default: `pos_embedding`)
- *standardize*: standardize data? `True`/`False` (default: `True`)
- *fit_nonstationary*: fit a non-stationary time series? `True`/`False` (default: `False`)
- *initial_x*: Optional initial condition for the model as tensor of shape ($S$, $N$), else last context value is used. Does not wirk with `fit_nonstationary=True` (default: `None`)

### Visualization

The package includes visualization functions for different types of systems requiring ground truth ($T$, $N$), context ($T_C$, $N$) and reconstruction ($T$, $N$):

```python
from dynamix.utilities.plotting_eval import plot_3D_attractor, plot_2D_attractor, plot_TS_forecast

# For 3D systems
fig = plot_3D_attractor(ground_truth, context, reconstruction)

# For 2D systems
fig = plot_2D_attractor(context, reconstruction)

# For time series data
fig = plot_TS_forecast(ground_truth, context, reconstruction, prediction_length)
```

### Metrics

To evaluate the performance of the model, several metrics are implemented (see [`metrics`](src/dynamix/metrics/)):

- **Geometrical Misalignment**: Measures how well the model captures the geometry of the attractor
- **Temporal Misalignment**: Measures how well the model captures the temporal patterns in the data
- **MASE**: Mean absolute scaled error, measures the accuracy of short-term predictions

```python
from dynamix.metrics.metrics import geometrical_misalignment, temporal_misalignment, MASE

dstsp = geometrical_misalignment(reconstruction_tensor, ground_truth_tensor, n_bins=30)
dh = temporal_misalignment(reconstruction_tensor, ground_truth_tensor, smoothing=20)
pe = MASE(ground_truth_tensor, reconstruction_tensor, steps=10)
```

## Training the Model

### Training Algorithm

DynaMix is trained using backpropagation through time with sparse teacher forcing (STF), a technique that balances stability and learning:

1. The model makes predictions over a sequence of time steps
2. At regular intervals (specified by `n_interleave`), the model's state is reset to match ground truth
3. This prevents error accumulation while still allowing the model to learn long-term dependencies

### Training the model
To train the model, see [`training_setup`](src/dynamix/training/training_setup.py) script for more details. Appropriate arguments can be parsed via the command line (or via changing the ones from the defaults in the [`settings`](src/dynamix/training/settings/defaults.json)):

```bash
python -m dynamix.training.training_setup 
        --latent_dim 30 --experts 10 --pwl_units 2 \
        --epochs 2000 --batch_size 256 --device cuda
```

### Model Saving
During training, the model is saved every `ssi` (scalar_saving_interval) epochs. The saved model state dictionary can be found in the specified `save_path` directory under the "checkpoints" subfolder. Additionally, training metrics are saved separately in a metrics.pt file.

## Citation

If you use DynaMix in your research, please cite our paper:

```
@misc{hemmer2025truezeroshotinferencedynamical,
      title={True Zero-Shot Inference of Dynamical Systems Preserving Long-Term Statistics}, 
      author={Christoph Jürgen Hemmer and Daniel Durstewitz},
      year={2025},
      eprint={2505.13192},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.13192}, 
}
```


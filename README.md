# Scion

Code accompanying the paper [Training Deep Learning Models with Norm-Constrained LMOs](https://arxiv.org/pdf/2502.07529).

## Repository structure

- [`scion.py`](scion.py): Contains the `Scion` and `ScionLight` reference implementation along with various norm choices. 
    `ScionLight` is a memory-efficient variant that reuses `p.grad`.
- [`examples/`](examples/): Example usage containing nanoGPT experiments with and without weight sharing.

## Notes

The `Scion` optimizer comes with a couple of hyperparameters:

- `momentum`: The parameter is `1-usual_momentum` of e.g. the PyTorch implementation of SGD with momentum. 
    A good default is 0.1. 
    Higher values seems to work better (e.g. 0.5) for short training runs with low noise as also supported by theory.
- `scale`: Controls the per-layer constraint radius factor. 
    The layerwise radius can be tuned on a small proxy model similarly to the input and output scaling factor of µP.
- `lr`: The learning rate can similarly be tuned on a small proxy model (corresponds to γ in the paper).
- `unconstrained`: When set to `False` the constrained variant of the Scion is used, which guarantees the iterates to stay bounded.
    The flag is useful for numerical stability in long training runs and to avoid overfitting.
    See [Section 3](https://arxiv.org/pdf/2502.07529) for a discussion on the connection with weight decay.

Architectural changes:

- Scale activation functions (ReLU, GELU) [by √2](https://github.com/LIONS-EPFL/scion/blob/main/examples/shallow-nanogpt/model.py#L104) to maintain the input variance.


## Examples

For runnable examples see [`examples/`](examples/).
Below are some pseudocode configurations for different architectures and domains (see [Appendix E.4](https://arxiv.org/pdf/2502.07529) for exact parameter choices):


- nanoGPT with weight sharing:

    ```python
    radius = 50.0
    optim_groups = [{
        'params': model.transformer.h.parameters(),
        'norm': 'Spectral',
        'norm_kwargs': {},
        'scale': radius,
    }, {
        'params': model.lm_head.parameters(),
        'norm': 'Sign',
        'norm_kwargs': {},
        'scale': radius*60.0,
    }]
    optimizer = Scion(optim_groups, lr=2**-12, momentum=0.1, unconstrained=False)
    ```

- MLP:

    ```python
    radius = 1.0
    optim_groups = [{
        'params': input_layer,
        'norm': 'Spectral',
        'norm_kwargs': {'max': True},
        'scale': radius,
    }, {
        'params': hidden_layers,
        'norm': 'Spectral',
        'norm_kwargs': {},
        'scale': radius,
    }, {
        'params': output_layer,
        'norm': 'Sign',
        'norm_kwargs': {'normalized': True},
        'scale': radius*2**10.0,
    }]
    optimizer = Scion(optim_groups, lr=2**-6, momentum=0.1)
    optimizer.init()
    ```

- CNN (see [`examples/airbench`](examples/airbench) for further details):

    ```python
    radius = 8.0
    optim_groups = [{
        'params': remaining_parameters,
        'norm': 'Auto', # Picks layerwise norm based on the parameter shape
        'norm_kwargs': {},
        'scale': radius,
    }, {
        'params': output_layer,
        'norm': 'Sign',
        'norm_kwargs': {'normalized': True},
        'scale': radius*16,
    }]
    optimizer = Scion(optim_groups, lr=2**-4, momentum=0.5)
    ```

## Changelog

- **2026-02-24 change of momentum initialization**: Instead of initializing the momentum buffer to zero, we initialize it with the gradient at initialization as required theoretically (see [Section 5](https://arxiv.org/pdf/2502.07529)), which leads to a consistent improvement experimentally.


## Citation

If you find this work useful, please cite it as follows:

```bibtex
@article{pethick2025training,
  title={Training Deep Learning Models with Norm-Constrained LMOs},
  author={Pethick, Thomas and Xie, Wanyun and Antonakopoulos, Kimon and Zhu, Zhenyu and Silveti-Falls, Antonio and Cevher, Volkan},
  journal={arXiv preprint arXiv:2502.07529},
  year={2025}
}
```

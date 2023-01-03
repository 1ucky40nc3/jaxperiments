# MNIST Classification

This is a simple example to train a MNIST example taken from [flax/examples/mnist](https://github.com/google/flax/tree/main/examples/mnist). Mind the original [license](https://github.com/google/flax/blob/main/LICENSE).

Big thanks to the authors for their great work!

## Usage

To execute the training just call the following command from the `jaxperiments` root directory.
```python
python3 projects/mnist/main.py --workdir /tmp/mnist --config projects/mnist/configs/default.py
```
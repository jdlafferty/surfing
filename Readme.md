# Codes for the experiments in the paper
---
## Fashion-MNIST dataset
The dataset is available [here]
(https://github.com/zalandoresearch/fashion-mnist).

Put the data in the directory modelname/data/fashion-mnist, modelname = ['VAE','DCGAN','WGAN','WGAN-GP']

## Files
* train.py  -- train each model.
* GD_regular.py/GD_surfing.py -- apply surfing and regular gradient descent to solve $f(x) = \|G(x)-G(x_*)\|^2$.
* compress.py -- apply surfing and regular gradient descent to solve $f(x) = \|AG(x)-Ay\|^2$, where $y$
is from test data.
* compressX.py -- apply surfing and regular gradient descent to solve $f(x) = \|AG(x)-AG(x_*)\|^2$.
* make-data.py -- generate data used to plot the surfaces in introduction.

## Reference
The following codes are from [https://github.com/hwalsuklee/tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections)

* VAE.py
* GAN.py
* WGAN.py
* WGAN-GP.py
* ops.py
* prior_factory.py
* utils.py

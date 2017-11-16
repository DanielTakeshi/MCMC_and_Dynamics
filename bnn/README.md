# Bayesian Neural Networks

This code is based on [Tianqi Chen's code for SGHMC][1].

A few pointers/questions:

- I'm not seeing any value of `m`, the number of Leapfrog steps. Looks like Chen
  just did one time, so `L=1` using the notation from (Neal, 2010).

- The mass matrix `M` is assumed some multiple of the identity matrix, so we can
  update momentum, and then add it to theta. In general we have to assume the
  step sizes absorb lots of constants.

- Hyper-parameters are indeed updated via Gibbs steps, which I know in detail.

- Chen combined the updates for SGLD and SGHMC into one updater.

- Not mentioned in the paper, but Chen uses weight decay on all the weights to
  impose the assumed Gaussian prior on the weights. Or, maybe it's common
  knowledge. (If confused, [see the CS 231n notes][2], and it's helpful to do
  the derivation.)

- Notation: `i_[...]` for inputs, `o_[...]` for outputs, `w` for weights, `g_w`
  for gradients, and `m_w` for momentums.

- TODO: utilize TensorFlow for gradients and get similar results.

[1]:https://github.com/tqchen/ML-SGHMC
[2]:http://cs231n.github.io/linear-classify/

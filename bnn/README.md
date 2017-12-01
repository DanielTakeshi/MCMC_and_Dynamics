# Bayesian Neural Networks with Hamiltonian Monte Carlo

Versions:

- Python 3.5.3
- TensorFlow 1.4.0

Example Usage:

```
python main.py --leapfrog_step 0.001 --temperature 10 --num_leapfrog 3 --seed 50
```

See my [accompanying blog post][2] for a description. (Note: I will fix it on
December 2 to reflect my exact implementation, but the basic idea should still
apply.)


This code is largely based on [Tianqi Chen's code for SGHMC][1] along with
Radford Neal's perspective on Bayesian Neural Networks using Hamiltonian Monte
Carlo.


## TODOs

- Implement the more sophisticated Metropolis-Hastings tests.

- Add a schedule for the temperature parameter.

- Add scripts to experiment with different hyperparameters, including adjusting
  the mass matrix.

- Adaptively adjust the leapfrog step size based on the acceptance rate.

- Benchmark with SGD, and if benchmarking with RMSProp and Adam, consider
  adjusting the leapfrog so that I use this information. (How?!?)

[1]:https://github.com/tqchen/ML-SGHMC
[2]:https://danieltakeshi.github.io/2017/11/26/basics-of-bayesian-neural-networks/

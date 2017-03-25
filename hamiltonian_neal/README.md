# MCMC Using Hamiltonian Dynamics

This section is about understanding the paper *MCMC Using Hamiltonian Dynamics*, by Radford Neal (2010).

## Simple Gaussian Results

### Univariate Gaussian

Section 5.2.1.3 example. Not much to say, except I really have to remember to use `astype('float32')` to avoid getting weird numpy errors with integers versus floats.

![simple_gaussians](draft_figures/univariate_gaussians.png?raw=true)

### Bivariate Gaussian, One Sample

Section 5.3.3.1 example. This is only for ONE sample to be drawn, which involves a 25-step leapfrog.

![bivariate_gaussian_1](draft_figures/bivariate_gaussians_one_sample.png?raw=true)

And the error is bounded, in fact here is a 200-step leapfrog:

![bivariate_gaussian_2](draft_figures/bivariate_gaussians_one_sample_200steps.png?raw=true)

Looks cool, right?

I also have gradient graphs not in this paper, for gradients of the position variables. However, I'm not sure what to make of them, they just oscillate.

I'm not sure what to think of this. However, understanding the figures that are like those in Neal's paper is easier because the update is just q = q + eps\*p. In other words, p helps to "guide" q to where it should go, which might be why it's called "momentum!" Think: if q is at the origin, and p is at [1,1], then the update will result in q = [eps,eps], moving towards the direction of p. Thus, the movement of q, I understand. The movement of p in the first place is a little harder to process. I need to think a little more about this.

### Bivariate Gaussian, Multiple Samples

TODO This is a WIP. I am trying to follow Neal's settings, but for some reason I am getting weird results. The acceptance rates are too low. I will double check the leapfrog code because the Hamiltonians shoudn't be fluctuating this much, right? If they follow the pattern from the one-sample case above, they should remain roughly at the 2.02 level (well, adjusting for the 0.98 covariance here). BTW, we can't do a covariance with diagonals greater than 1, since that wouldn't be a positive definite matrix.

I also need to get the random walk done ...

![bivariate_gaussian_many_1](draft_figures/bivariate_gaussians_many_samples.png?raw=true)

# MCMC Using Hamiltonian Dynamics

This section is about understanding the paper *MCMC Using Hamiltonian Dynamics*,
by Radford Neal (2010).

## Simple Gaussian Results

### Univariate Gaussian

Section 5.2.1.3 example. Not much to say, except I really have to remember to
use `astype('float32')` to avoid getting weird numpy errors with integers versus
floats.

![simple_gaussians](draft_figures/univariate_gaussians.png?raw=true)

### Bivariate Gaussian, One Sample

Section 5.3.3.1 example. This is only for ONE sample to be drawn, which involves
a 25-step leapfrog.

![bivariate_gaussian_1](draft_figures/bivariate_gaussians_one_sample.png?raw=true)

And the error is bounded, in fact here is a 200-step leapfrog:

![bivariate_gaussian_2](draft_figures/bivariate_gaussians_one_sample_200steps.png?raw=true)

Looks cool, right?

I also have gradient graphs not in this paper, for gradients of the position
variables. However, I'm not sure what to make of them, they just oscillate.

I'm not sure what to think of this. However, understanding the figures that are
like those in Neal's paper is easier because the update is just q = q + eps\*p.
In other words, p helps to "guide" q to where it should go, which might be why
it's called "momentum!" Think: if q is at the origin, and p is at [1,1], then
the update will result in q = [eps,eps], moving towards the direction of p.
Thus, the movement of q, I understand. The movement of p in the first place is a
little harder to process. I need to think a little more about this.

### Bivariate Gaussian, Multiple Samples

Hmmmm ... I can't quite replicate Neal's result. I'm getting acceptance rates of
about 0.5-0.6 using his reported settings. I had to tune the step size epsilon
and the number of leapfrog steps L, but I can get something that *looks* good,
and certainly looks better than a random walk. 

I wonder, how can we get an acceptance rate of 0.91 if this example is very
similar to the previous one, and there, Neal notes that the acceptance rate for
that last sample is 0.66 (see bottom of Section 5.3.3.1). Thus it seems odd that
we would always get high acceptance rates (i.e. close to one). 

I may double-check this leapfrog code later but I think it's exactly the same as
his pseudocode and in my earlier code. Even so, the Hamiltonians seem to be
fluctuating a bit too much for me, but who knows? 

Note to self: we can't do a covariance with diagonals greater than 1, since that
wouldn't be a positive definite matrix.

Fortunately, the random walk seems to work. I did 400 points and am showing
every 20th, like he did. Our rejection rates appear to match. (The number here
in parentheses are actually the *acceptance* rates.)

![bivariate_gaussian_many_1](draft_figures/bivariate_gaussians_many_samples.png?raw=true)

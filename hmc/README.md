# MCMC Using Hamiltonian Dynamics

This README will attempt to reproduce the plots from *MCMC Using Hamiltonian
Dynamics*, by Radford Neal (2010). I will also augment these with my own
extensions as I investigate tuning the parameters.

Contents:

- [Discretizing Hamiltonian Dynamics](##discretizing-hamiltonian-dynamics)
- [Bivariate Gaussians Example](#bivariate-gaussians)
    - [One Iteration of HMC](#figure-53-one-leapfrog-trajectory)
    - [Many Iterations ofHMC](#figure-53-one-leapfrog-trajectory)

## Discretizing Hamiltonian Dynamics

See `hamiltonian_univariate_gaussian.py` for the code I used.

We assume the following setup with 1-D position `q` and momentum `p` variables:

- `H(q,p) = U(q) + K(p)`
- `U(q) = q^2/2`
- `K(p) = p^2/2`

So the dynamics are:

- `dq/dt = p`
- `dp/dt = -q`

with solution

- `q(t) = r * cos(a+t)`
- `p(t) = -r * sin(a+t)`

First, here's the figure which reproduces Figure 1. It does so perfectly, as far
as I can tell. 

![simple_gaussians](figures/univariate_gaussians.png?raw=true)

Some extensions for the leapfrog method, where we test with different step
sizes. We can see that at `eps=2.0`, the trajectory diverges. (If this were
inside HMC, we'd be rejecting those samples, so it's not as bad as it looks ...
but it's still awful.)

![leapfrog_extensions](figures/univariate_gaussians_leapfrog_tests.png?raw=true)




## Bivariate Gaussians

### Figure 5.3, One Leapfrog Trajectory

See `bivariate_gaussian_one_leapfrog.py`.

I'm attempting to re-generate Figure 5.3, which is about ONE leapfrog
trajectory, so this would be done in one iteration of HMC to generate a sample.
Here it is below, reproduced exactly:

![bivariate_gaussian_1](figures/bivariate_gaussians_one_leapfrog_traj.png?raw=true)

Note that we'd normally not be interested in the intermediate steps, as all we
want to know is the FINAL Hamiltonian value (here it's 2.616) so that we'd
compare the energy difference with the starting value (of 2.205). So indeed, the
difference is about 0.41 (as Neal says!) so the probability of accepting the
final (position,momentum) = (q,p) coordinates here is `exp(-0.41) = 0.66`.

By the way, one full "momentum" step is really two half-steps that surround a
full position step. In normal code, we'd combine the two half-steps for the
intermediate momentum steps. I only split them here so that I could plot them
visually and get it to match Neal's plot exactly (as everything here is
deterministic).

Now let's see what happens when we vary epsilon. Indeed, as we get to 0.45, the
Hamiltonian diverges, as Neal confirms:

![bivariate_gaussian_2](figures/bivariate_gaussians_one_leapfrog_investigation.png?raw=true)

**Wow**, look at the difference between epsilon of 0.44 and 0.45. And by the
way, **the Hamiltonian value plots are not on the same scale!!** The positions
and momentums are on the same scale, for clarity. For instance, with a small
step size, the Hamiltonian values have a maximum difference bounded by 0.002.


### Figure 5.4, Running HMC

This time, we'll run many iterations of HMC, rather than just one as we did
earlier for replicating Figure 5.3.

TODO

# MCMC Using Hamiltonian Dynamics

This README will attempt to reproduce the plots from *MCMC Using Hamiltonian
Dynamics*, by Radford Neal (2010). I will also augment these with my own
extensions as I investigate tuning the parameters.

Contents:

- [Discretizing Hamiltonian Dynamics](#discretizing-hamiltonian-dynamics)
- [Bivariate Gaussians Example](#bivariate-gaussians)
    - [One Iteration of HMC](#figure-53-one-leapfrog-trajectory)
    - [Many Iterations of HMC](#figures-54-and-55-running-hmc)
- [A 100-Dimensional Distribution](#a-100-dimensional-distribution)

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



### Figures 5.4 and 5.5, Running HMC

This time, we'll run many iterations of HMC, rather than just one as we did
earlier for replicating Figure 5.3. Here's my attempt at reproducing Figure 5.4
using the same hyperparameters. For good measure, I also put the value of the
Hamiltonians after the end of the iteration (after we have our new (or re-used)
sample and the negated momentum) just for a sanity check. The Hamiltonian seems
to vary a lot, but at least the trajectory looks like it takes longer jumps than
random walks.

![many_samples](figures/bivariate_gaussians_many_samples.png?raw=true)

The number in the title is the acceptance rate. Neal reports 63% and 91%
acceptance rates for RW and HMC, respectively, and my numbers match. I ran this
an additional 20 times and here is the mean and standard deviation:

```
HMC accept rate: 0.910 +/- 0.073
RW accept rate: 0.637 +/- 0.021
```

I swear, I did not run this more times to get exactly a 91% acceptance rate
mean. Results are not deterministic since we re-sample the momentum variables
each time, and also I don't know where he set his starting position points, but
it's good that the results seem comparable.

Here's an attempt at replicating Figure 5.5 as well, and I also added in the
second coordinates:

![many_samples](figures/bivariate_gaussians_fig5-5.png?raw=true)

Yes, for both coordinates, there is a clear pattern in that the random walk
samples are more correlated. And recall, these are the position variables. And
also recall, yes we have 20 leapfrog steps, but we are also plotting *every* HMC
sample, whereas we're plotting every 20 **random-walk** steps, so we're really
comparing N **leapfrog steps** to N random walk steps (Neal does this to "match
computation time" which makes sense). HMC still wins out.


## A 100-Dimensional Distribution

(See the `gaussian_high_dim.py` script.)

Now we generate the last two figures in (Neal, 2010), Figures 5.6 and 5.7, based
on a 100-dimensional Gaussian with standard deviations of 0.01, 0.02, ..., 0.99,
1.0. (Neal, 2010) claim that this represents "more typical behavior" but again
this is a Gaussian so ... I'll take that with a grain of salt, knowing that Deep
Nets will be facing much harder problems. Remember, this 100-dimensional
Gaussian is what we want MCMC algorithms to *sample from*.

Anyway, let's generate the figures! Here's my reproduction of Figure 5.6:

![high_dim_fig5-6](figures/high_dim_gaussian_5-6.png?raw=true)

Couple of things:

- I actually added in the plot for the first coordinate as a sanity check. This
  one has a much smaller standard deviation, as expected, and you can't see the
  RW vs HMC difference in performance.

- The second row, which is what Neal actually reports, clearly shows the benefit
  of HMC over RW.

- Watch out! (Neal, 2010) reports the **standard deviations** of the various
  Gaussians, but we need the **covariance**, so be sure to square those values.
  I forgot to do that originally, and ended up getting absurdly high acceptance
  rates (essentially a 100% acceptance rate for HMC and 80% for RW). I fixed
  that, and as you can see my acceptance rates are comparable to Neal's reported
  values. Neal's values were 87% for HMC and 25% for RW.

- For the starting position, I randomly sampled it from a 100-dimensional
  Gaussian with the `100x100` identity matrix. I don't think it matters that
  much, though; you can clearly see that the sampling algorithm will correct for
  any initial deviations.

Now we get to the last figure in (Neal, 2010) where we plot the estimates of the
mean and standard deviations of each of the variables (i.e., coordinates) from
our sample mean and sample standard deviations. (This is the point of being a
Bayesian, so that we use our samples to approximate integrals, though in this
case we have closed form solutions but whatever ...  just pretend that these
means and standard deviations are hard to compute without sampling.)

Here we go:

![high_dim_fig5-7](figures/high_dim_gaussian_5-7.png?raw=true)

Yes, it's clear that our estimates with HMC are better. I wish we could say "our
posterior estimates" but of course it's cheating here as we know the form of the
"posterior" which is "this multivariate Gaussian we defined ...". 

So ... the next step, now that I've replicated all of the figures in (Neal,
2010), is to apply this to harder problems.

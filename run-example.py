#!/usr/bin/env python3

import numpy as np
import polars as pl
import plotnine as pn
import pypfilt
import scipy.stats


scenario_file = 'example.toml'
instances = {
    instance.scenario_id: instance
    for instance in pypfilt.load_instances(scenario_file)
}

# Generate simulated observations.
obs_tables = pypfilt.simulate_from_model(instances['simulate'])
rolls = obs_tables['roll']
rolls_file = 'simulated-rolls.ssv'
pypfilt.io.write_table(rolls_file, rolls, time_scale=pypfilt.Scalar())

# Generate prior samples from a Dirichlet distribution.
# See https://en.wikipedia.org/wiki/Categorical_distribution, in particular
# the "Bayesian inference using conjugate prior" section.
rng = np.random.default_rng(seed=12345)
prior_file = 'prior.ssv'
dist = scipy.stats.dirichlet(alpha=[1, 1, 1, 1, 1, 1])
num_particles = instances['example'].settings['filter']['particles']
prior_samples = dist.rvs(size=num_particles, random_state=rng)
np.savetxt(
    prior_file, prior_samples, header='p_1 p_2 p_3 p_4 p_5 p_6', comments=''
)

# Fit to the observations
context = instances['example'].build_context()
results = pypfilt.fit(context, filename=None)

# Collect the credible intervals for each outcome, over time.
cints = results.estimation.tables['model_cints']
df_cints = (
    pl.from_numpy(cints)
    .select(['time', 'name', 'prob', 'ymin', 'ymax'])
    .with_columns(
        name=pl.col('name').apply(str),
        ci=pl.lit(0).sub(pl.col('prob')),
    )
)

# Define the true probabilities, from which we simulated the observations.
df_true = pl.DataFrame(
    {
        'name': ['p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6'],
        'truth': [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
    }
)

# Calculate how often each outcome was observed.
df_rolls = pl.from_numpy(rolls)
df_fracs = (
    df_rolls.with_columns(name=pl.col('value').apply(lambda v: f'p_{v}'))
    .groupby('name')
    .agg(frac=pl.col('time').count().truediv(pl.lit(df_rolls.height)))
)

# Plot the credible intervals against the ground truth and observed outcomes.
plot = (
    pn.ggplot()
    + pn.geom_ribbon(
        df_cints,
        pn.aes(
            'time',
            ymin='ymin',
            ymax='ymax',
            fill='prob',
            colour='prob',
            group='ci',
        ),
    )
    + pn.geom_hline(df_true, pn.aes(yintercept='truth'))
    + pn.geom_hline(df_fracs, pn.aes(yintercept='frac'), linetype='dashed')
    + pn.facet_wrap('name', labeller=lambda s: f'Pr(Roll a {s[-1]})')
    + pn.xlab('Number of observations')
    + pn.ylab('Probability')
    + pn.scale_fill_continuous(
        name='CrI',
        breaks=[0, 50, 95],
        labels=['0%', '50%', '95%'],
    )
    + pn.scale_colour_continuous(
        name='CrI',
        breaks=[0, 50, 95],
        labels=['0%', '50%', '95%'],
    )
)

plot_file = 'example.png'
print(f'Saving {plot_file} ...')
plot.save(
    plot_file,
    width=8,
    height=6,
    units='in',
    dpi=300,
    verbose=False,
    metadata={'Software': None},
)

"""
An example model for categorical observations (die rolls).
"""

import numpy as np
import pypfilt
from pypfilt.io import time_field, read_fields


class Example(pypfilt.Model):
    """A model that describes the probability of each outcome."""

    def field_types(self, ctx):
        return [
            ('p_1', np.float_),
            ('p_2', np.float_),
            ('p_3', np.float_),
            ('p_4', np.float_),
            ('p_5', np.float_),
            ('p_6', np.float_),
        ]

    def can_smooth(self):
        # NOTE: allow the probabilities to be smoothed.
        return {'p_1', 'p_2', 'p_3', 'p_4', 'p_5', 'p_6'}

    def update(self, ctx, time_step, is_forecast, prev, curr):
        # NOTE: normalise the probabilities, so that they sum to unity after
        # post-regularisation is applied.
        denom = (
            prev['p_1']
            + prev['p_2']
            + prev['p_3']
            + prev['p_4']
            + prev['p_5']
            + prev['p_6']
        )
        for col in prev.dtype.names:
            prev[col] = prev[col] / denom
        # Simply copy the probabilities.
        curr[:] = prev[:]


class Roll(pypfilt.Obs):
    """An observation model for die rolls."""

    def __init__(self, obs_unit, settings):
        self.unit = obs_unit
        self.settings = settings

    def log_llhd(self, ctx, snapshot, obs):
        roll = obs['value']
        if roll < 0 or roll > 6:
            raise ValueError(f'Invalid roll: {roll}')
        probs = snapshot.state_vec.view((np.float_, 6))
        roll_probs = probs[:, roll - 1]
        if np.any(roll_probs == 0):
            raise ValueError('Probability cannot be zero')
        return np.log(roll_probs)

    def simulate(self, ctx, snapshot, rng):
        # Sample from the unit interval to select a value for each particle.
        probs = snapshot.state_vec.view((np.float_, 6))
        thresholds = rng.random(size=(probs.shape[0], 1))
        cum_probs = np.cumsum(probs, axis=-1)
        samples = 1 + np.sum(thresholds >= cum_probs, axis=-1)
        return samples

    def from_file(self, filename, time_scale):
        fields = [time_field('time'), ('value', np.int_)]
        return read_fields(time_scale, filename, fields)

    def row_into_obs(self, row):
        return {
            'time': row['time'],
            'value': row['value'],
            'unit': self.unit,
        }

    def obs_into_row(self, obs, dtype):
        return (obs['time'], obs['value'])

    def simulated_obs(self, ctx, snapshot, rng):
        values = self.simulate(ctx, snapshot, rng)
        return [{'time': snapshot.time, 'value': value} for value in values]

    def simulated_field_types(self, row):
        return [time_field('time'), ('value', np.int_)]

    def expect(self, ctx, snapshot):
        raise NotImplementedError()

    def quantiles(self, ctx, snapshot, probs):
        raise NotImplementedError()

    def llhd_in(self, ctx, snapshot, y0, y1):
        raise NotImplementedError()

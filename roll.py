import numpy as np
import pypfilt
from pypfilt.io import time_field, read_fields


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

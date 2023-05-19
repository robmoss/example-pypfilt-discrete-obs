import numpy as np
import pypfilt


class Die(pypfilt.Model):
    """A model that describes the probability of each 6-sided die outcome."""

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

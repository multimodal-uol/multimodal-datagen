#!/usr/bin/env python

from operator import itemgetter

from collections import OrderedDict
import numpy as np
import scipy.stats as stats

from algos_support.density_function import NUMERIC_PRECISION
from algos_support.density_function.error import DistributionFitError
from algos_support.density_function.probability_distribution import ProbabilityDistribution

import cexc

logger = cexc.get_logger(__name__)


class BetaDistribution(ProbabilityDistribution):
    """Wrapper around scipy.stats.beta"""

    Name = 'Beta'

    def __init__(self):
        super(BetaDistribution, self).__init__()
        self._alpha = None  # Alpha value of the distribution function
        self._beta = None  # Beta value of the distribution function

    def fit(self, data, distance=None):
        # Error out and do not fit Beta if all the values in the dataset are the same.
        # Normal distribution will be fit instead in auto mode.
        if np.all((data == data[0])):
            raise DistributionFitError(
                'Unable to fit a Beta distribution due to numerical errors. All values in the dataset are the same.'
            )
        self._min = data.min()
        self._max = data.max()

        # Beta distribution fits only in 0-1 range, scale the data between 0-1.
        normalized_data = (data - self._min) / (self._max - self._min)
        self._alpha, self._beta, self._mean, self._std = stats.beta.fit(normalized_data)
        # Error out and do not continue if the data is not shaped as a mirrored exponential distribution
        # self._beta must be less than 1, but we allow a 0.1 range of difference not to exclude any mirrored exponential like shapes.
        if not (self._beta <= 1.1 and self._alpha >= 1 and self._alpha != self._beta):
            raise DistributionFitError(
                'Not fitting a Beta distribution. The data is not shaped as a Mirrored Exponential distribution.'
            )
        self._other_params['Alpha'] = self._alpha
        self._other_params['Beta'] = self._beta
        self._cardinality = data.shape[0]
        if distance:
            self._metric = distance
            self._distance = self._get_distance(data, metric=distance)
            return self._distance

    @staticmethod
    def _make_boundary_ranges(lower_boundary, threshold):
        boundary_ranges = OrderedDict()
        if lower_boundary:
            lower_boundary = [round(lower, NUMERIC_PRECISION) for lower in lower_boundary]
            th = threshold.lower if threshold.lower is not None else threshold.threshold
            for thrshld, lower in zip(th, lower_boundary):
                boundary_ranges[thrshld] = [[-np.inf, lower, thrshld]]
        return boundary_ranges

    def apply(self, data, threshold, params):
        # (self._max-self._min) is never 0. In fit we error out if all data points are same. This guarantees self._min and self._max to be different.
        normalized_data = (data - self._min) / (self._max - self._min)
        dist = stats.beta(a=self._alpha, b=self._beta, loc=self._mean, scale=self._std)
        outliers = []
        lower = []
        if threshold.is_lower_upper():
            lower = [dist.ppf(l) for l in threshold.lower] if threshold.lower else None
            if lower:
                for l in lower:
                    tmp_outliers = np.zeros(normalized_data.shape[0])
                    tmp_outliers[(normalized_data < l)] = 1
                    outliers.append(tmp_outliers)
            else:
                for _ in threshold.upper:
                    tmp_outliers = np.zeros(normalized_data.shape[0])
                    outliers.append(tmp_outliers)
        else:
            for thrshld in threshold.threshold:
                lower.append(dist.ppf(thrshld))
                tmp_outliers = np.zeros(normalized_data.shape[0])
                # only set the outliers that are less than the lower threshold, do not set the ones as outliers which are greater than 1.
                tmp_outliers[normalized_data < lower[-1]] = 1
                outliers.append(tmp_outliers)
        # Rescale the normalized boundary value to the actual range
        lower = (
            [x * (self._max - self._min) + self._min for x in lower]
            if lower is not None
            else None
        )
        boundary_ranges = BetaDistribution._make_boundary_ranges(lower, threshold)
        samples = (
            self.sample_within_boundaries(
                data.size, [[list(p)] for p in zip([-np.inf] * len(lower), lower)]
            )
            if params.sample
            else None
        )

        densities = dist.pdf(normalized_data) if params.show_density else None
        full_samples = self.sample(data.size) if params.full_sample else None

        return BetaDistribution._create_result(
            outliers, boundary_ranges, densities, full_samples, samples
        )

    def sample(self, size):
        samples = stats.beta.rvs(
            a=self._alpha, b=self._beta, loc=self._mean, scale=self._std, size=size
        )
        # Rescale the sample values to the actual range
        samples = (samples * (self._max - self._min)) + self._min
        return samples

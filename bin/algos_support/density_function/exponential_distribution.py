#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
import scipy.stats as stats

from algos_support.density_function import NUMERIC_PRECISION
from algos_support.density_function.error import DistributionFitError
from algos_support.density_function.probability_distribution import ProbabilityDistribution


class ExponentialDistribution(ProbabilityDistribution):
    """Wrapper around scipy.stats.expon"""

    Name = 'Exponential'

    def fit(self, data, distance=None):
        # Error out and do not fit Exponential if all the values in the dataset are the same.
        # Normal distribution will be fit instead in auto mode.
        if np.all((data == data[0])):
            raise DistributionFitError(
                'Unable to fit an Exponential distribution due to numerical errors. All values in the dataset are the same.'
            )
        self._mean, self._std = stats.expon.fit(data)
        self._min = data.min()
        self._max = data.max()
        self._cardinality = data.shape[0]
        if distance:
            self._metric = distance
            self._distance = self._get_distance(data, metric=distance)
            return self._distance

    @staticmethod
    def _make_boundary_ranges(upper_boundary, threshold):
        boundary_ranges = OrderedDict()
        if upper_boundary:
            upper_boundary = [round(upper, NUMERIC_PRECISION) for upper in upper_boundary]
            th = threshold.upper if threshold.upper is not None else threshold.threshold
            for thrshld, upper in zip(th, upper_boundary):
                boundary_ranges[thrshld] = [[upper, np.inf, thrshld]]
        return boundary_ranges

    def apply(self, data, threshold, params):
        dist = stats.expon(loc=self._mean, scale=self._std)
        outliers = []
        if threshold.is_lower_upper():
            upper = [dist.ppf(1 - u) for u in threshold.upper] if threshold.upper else None
            if upper:
                for u in upper:
                    tmp_outliers = np.zeros(data.shape[0])
                    tmp_outliers[(data > u)] = 1
                    outliers.append(tmp_outliers)
            else:
                for _ in threshold.lower:
                    tmp_outliers = np.zeros(data.shape[0])
                    outliers.append(tmp_outliers)
        else:
            upper = []
            for thrshld in threshold.threshold:
                upper.append(dist.ppf(1 - thrshld))
                tmp_outliers = np.zeros(data.shape[0])
                tmp_outliers[(data > upper[-1])] = 1
                outliers.append(tmp_outliers)
        boundary_ranges = ExponentialDistribution._make_boundary_ranges(upper, threshold)
        densities = dist.pdf(data) if params.show_density else None
        full_samples = self.sample(data.size) if params.full_sample else None
        samples = (
            self.sample_within_boundaries(
                data.size, [[list(p)] for p in zip(upper, [np.inf] * len(upper))]
            )
            if params.sample
            else None
        )
        return ExponentialDistribution._create_result(
            outliers, boundary_ranges, densities, full_samples, samples
        )

    def sample(self, size):
        return stats.expon(loc=self._mean, scale=self._std).rvs(size)

from collections import OrderedDict
from operator import itemgetter
import numpy as np
import scipy.stats as stats

from algos_support.density_function import NUMERIC_PRECISION
from algos_support.density_function.probability_distribution import ProbabilityDistribution
from algos_support.density_function.error import DistributionFitError


class KDEDistribution(ProbabilityDistribution):
    """"Wrapper around scipy.stats.gaussian_kde"""

    Name = 'Gaussian KDE'

    def __init__(self, max_param_size):
        super(KDEDistribution, self).__init__()
        self._max_param_size = max_param_size  # cap on the number of KDE parameters

    def _check_data_size(self):
        """Check that the number of training points is more than 1"""
        if self._data.shape[0] <= 1:
            raise DistributionFitError(
                'Unable to fit a Gaussian KDE, too few training data points'
            )

    def fit(self, data, distance=None):
        self._data = (
            data
            if data.shape[0] <= self._max_param_size
            else np.random.choice(data, size=self._max_param_size)
        )
        # Error out and do not fit Gaussian KDE if all the values in the original or sampled datasets are the same.
        # The values in the down-sampled dataset being the same shows that the majority of the dataset contains the same value.
        # Instead Normal distribution will be fit in auto mode.
        if np.all((self._data == self._data[0])):
            raise DistributionFitError(
                'Unable to fit a Gaussian KDE distribution due to numerical errors. All values in the dataset are the same.'
            )
        self._cardinality = data.shape[0]
        self._min = self._data.min()
        self._max = self._data.max()
        self._mean = self._data.mean()
        self._std = self._data.std()
        self._check_data_size()
        try:
            kde = stats.gaussian_kde(self._data)
        except Exception as ex:
            raise DistributionFitError(
                'Unable to fit a Gaussian KDE due to numerical errors: {}'.format(ex)
            )

        self._other_params['bandwidth'] = kde.covariance_factor() * self._std
        self._other_params['parameter size'] = self._data.shape[0]
        if distance:
            self._metric = distance
            self._distance = self._get_distance(self._data, metric=distance)
            return self._distance

    def _find_anomaly_ranges(self, kde, threshold):
        """
        Search for ranges in the random variable domain, where the sum of
        areas under the curve of KDE is approximately equal to threshold.
        Formally speaking:
            1) Finds the smallest T, such that:
                - For all x_i in the domain (random variable) where PDF(x_i) < T,
                    sum{PDF(x) dx} ~= threshold

            2) Return all x_i as a set of continuous ranges.

        Args:
            kde (stats.gaussian_kde): Gaussian KDE object
            threshold (float): Anomaly threshold

        Returns:
            (list[(begin, end, area)]): List of ranges for anomalous areas of the univariate distribution,
            where each range is specified by beginning and end points. The third element of each rage is the
            area under the curve that's covered by that range.
        """
        # Steps:
        # 1. Divide the range of random variable to equally spaced slots and
        # get the PDF at division points.
        # 2. Imagine a horizontal line through the Y axis (Y axis is density of the KDE distribution).
        # Find the best placement of line (i.e., the smallest density value) so
        # that the sum of area under the curve for points whose density is below
        # the line, is close to threshold.

        # start from a small margin before min and after max
        x_range = np.linspace(self._min * 0.9, self._max * 1.1, 1000)
        densities = kde.pdf(x_range)
        upper = densities.max()
        lower = 0
        prev_area = None
        # Cap the number of iterations to 1000, so we don't end up
        # in an infinite loop in case for some reason convergence
        # does not happen fast enough
        for _ in range(1000):
            middle = (upper + lower) / 2
            items = KDEDistribution._get_auc(kde, x_range, densities, middle)
            area = sum(map(itemgetter(2), items))
            rounded_area = round(area, NUMERIC_PRECISION)
            if rounded_area == prev_area:
                # no progress
                break
            prev_area = rounded_area
            # if we're reasonably close to target threshold call it a success
            if threshold * 0.999 < area < threshold * 1.001:
                break
            elif area < threshold:
                lower = middle
            else:
                upper = middle

        # Arrange for the first and last boundary items so that if the first range is
        # on the left-most side of the distribution, then its opening point is -INF
        # and if the last item is on the right-most side of the distribution then its
        # closing point is INF
        if items:
            if items[0][0] <= self._min:
                items[0][0] = -np.inf
            if items[-1][1] >= self._max:
                items[-1][1] = np.inf
        return items

    def _find_anomaly_ranges_lower(self, kde, t_lower):
        """
            Returns anomaly range for the left tail only.
            Start with the whole area under the density curve and move the right end point towards left.
        """
        left = self._min * 0.9
        right = self._max * 1.1
        prev_area = None
        prev_right = right
        for _ in range(1000):
            items = [left, right, kde.integrate_box_1d(left, right)]
            lower_area = items[2]
            rounded_lower_area = round(lower_area, NUMERIC_PRECISION)
            if rounded_lower_area == prev_area:
                break
            prev_area = rounded_lower_area

            if t_lower * 0.999 < lower_area < t_lower * 1.001:
                break
            elif lower_area < t_lower:
                right = (right + prev_right) / 2
            else:
                prev_right = right
                right = (right + left) / 2
        items[0] = -np.inf
        return [items]

    def _find_anomaly_ranges_upper(self, kde, t_upper):
        """
            Returns anomaly range for the right tail only.
            Start with the whole area under the density curve. Move the left end point towards right.
        """
        left = self._min * 0.9
        right = self._max * 1.1
        prev_area = None
        prev_left = left
        for _ in range(1000):
            items = [left, right, kde.integrate_box_1d(left, right)]
            upper_area = items[2]
            rounded_upper_area = round(upper_area, NUMERIC_PRECISION)
            if rounded_upper_area == prev_area:
                break
            prev_area = rounded_upper_area
            if t_upper * 0.999 < upper_area < t_upper * 1.001:
                break
            elif upper_area < t_upper:
                left = (left + prev_left) / 2
            else:
                prev_left = left
                left = (left + right) / 2
        items[1] = np.inf
        return [items]

    @staticmethod
    def _get_auc(kde, x_range, densities, max_density):
        """ Find the areas (i.e., ranges on the X axis) under the density function curve,
          where the corresponding density of all points in these areas are <= target_density.

        Args:
            kde (stats.gaussian_kde): Gaussian KDE object
            x_range (numpy.array): Array of values on the X axis
            densities (numpy.array): Array of probability densities for values of x_range
            max_density (float): The density value that specifies which points in the X axis are selected

        Returns:
            (list[(begin, end, area)]): List of ranges for areas where density of X values are less than
            max_density. The third element of each range is the area under the curve that's covered by that range.

        """
        idx = densities < max_density
        begin = None
        prev = None
        items = []
        for f, i in zip(idx, x_range):
            if f and not begin:
                begin = i
            elif not f and begin:
                # We know that `x_range[i-1] > target_density` and `x_range[i] < target_density`
                # So, we take  `(x_range[i-1] + x_range[i] / 2)`, hoping to get a better approximation
                items.append([begin, i, kde.integrate_box_1d(begin, (prev + i) / 2)])
                begin = None
            prev = i
        if begin:
            items.append([begin, i, kde.integrate_box_1d(begin, i)])
        return items

    @staticmethod
    def _make_boundary_ranges(anomaly_ranges, threshold):
        # Round values of each tuple element
        anomaly_ranges = [
            list(
                map(
                    lambda x: [
                        round(x[0], NUMERIC_PRECISION),
                        round(x[1], NUMERIC_PRECISION),
                        round(x[2], NUMERIC_PRECISION),
                    ],
                    anomaly,
                )
            )
            for anomaly in anomaly_ranges
        ]
        boundary_ranges = OrderedDict()
        if threshold.is_lower_upper():
            if threshold.lower is not None and threshold.upper is not None:
                for thrshld, anomaly in zip(
                    threshold.lower, anomaly_ranges[: len(threshold.lower)]
                ):
                    boundary_ranges[thrshld] = anomaly
                for thrshld, anomaly in zip(
                    threshold.upper, anomaly_ranges[len(threshold.lower) :]
                ):
                    boundary_ranges[thrshld] = anomaly
            else:
                if threshold.lower is not None:
                    for thrshld, anomaly in zip(threshold.lower, anomaly_ranges):
                        boundary_ranges[thrshld] = anomaly
                if threshold.upper is not None:
                    for thrshld, anomaly in zip(threshold.upper, anomaly_ranges):
                        boundary_ranges[thrshld] = anomaly
        else:
            for thrshld, anomaly in zip(threshold.threshold, anomaly_ranges):
                boundary_ranges[thrshld] = anomaly
        return boundary_ranges

    @staticmethod
    def _update_outliers(data, anomaly_ranges, outliers, both_lower_upper):
        tmp_outliers = np.zeros(data.shape[0])
        # anomaly_ranges can be empty when there is a sharp end of the distribution and if the threshold is too small. Check if it is not empty first.
        if anomaly_ranges[-1]:
            beginning, end, _ = anomaly_ranges[-1][0]
            idx = (beginning < data) & (data < end)
            if both_lower_upper:
                for beginning, end, _ in anomaly_ranges[-1][1:]:
                    idx = idx | ((beginning < data) & (data < end))
            tmp_outliers[idx] = 1
        outliers.append(tmp_outliers)

    def apply(self, data, threshold, params):
        kde = stats.gaussian_kde(self._data)
        anomaly_ranges = []
        outliers = []
        # supporting lower threshold, upper threshold or both. If both are given multiple threshold does not apply.
        if threshold.is_lower_upper():
            if threshold.lower is not None and threshold.upper is not None:
                for t_lower, t_upper in zip(threshold.lower, threshold.upper):
                    anomaly_ranges.append(
                        [
                            self._find_anomaly_ranges_lower(kde, t_lower)[0],
                            self._find_anomaly_ranges_upper(kde, t_upper)[0],
                        ]
                    )
                    KDEDistribution._update_outliers(
                        data, anomaly_ranges, outliers, both_lower_upper=True
                    )
            elif threshold.lower is not None:
                for t_lower in threshold.lower:
                    anomaly_ranges.append(self._find_anomaly_ranges_lower(kde, t_lower))
                    KDEDistribution._update_outliers(
                        data, anomaly_ranges, outliers, both_lower_upper=False
                    )
            else:
                for t_upper in threshold.upper:
                    anomaly_ranges.append(self._find_anomaly_ranges_upper(kde, t_upper))
                    KDEDistribution._update_outliers(
                        data, anomaly_ranges, outliers, both_lower_upper=False
                    )
        else:
            for thrshld in threshold.threshold:
                anomaly_ranges.append(self._find_anomaly_ranges(kde, thrshld))
                tmp_outliers = np.zeros(data.shape[0])
                if anomaly_ranges[-1]:
                    beginning, end, _ = anomaly_ranges[-1][0]
                    idx = (beginning < data) & (data < end)
                    for beginning, end, _ in anomaly_ranges[-1][1:]:
                        idx = idx | ((beginning < data) & (data < end))
                    tmp_outliers[idx] = 1
                outliers.append(tmp_outliers)
        boundary_ranges = KDEDistribution._make_boundary_ranges(anomaly_ranges, threshold)
        densities = kde.pdf(data) if params.show_density else None
        full_samples = self.sample(data.size) if params.full_sample else None
        samples = (
            self.sample_within_boundaries(data.size, anomaly_ranges) if params.sample else None
        )

        return KDEDistribution._create_result(
            outliers, boundary_ranges, densities, full_samples, samples
        )

    def sample(self, size):
        return stats.gaussian_kde(self._data).resample(size)[0]

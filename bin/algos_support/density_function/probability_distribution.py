from collections import OrderedDict
import numpy as np
import scipy.stats as stats

from algos_support.density_function.distance_metric import DistanceMetric
from algos_support.density_function.error import SamplingError


class DistributionType(object):
    """Enum class for distribution types"""

    AUTO = 'auto'
    NORMAL = 'norm'
    EXPONENTIAL = 'expon'
    GAUSSIAN_KDE = 'gaussian_kde'
    BETA = 'beta'


class DistributionName(object):
    """Enum class for distribution names"""

    AUTO = 'Auto'
    NORMAL = 'Normal'
    EXPONENTIAL = 'Exponential'
    GAUSSIAN_KDE = 'Gaussian KDE'
    BETA = 'Beta'


class ProbabilityDistribution(object):
    """Interface and basic functionality for probability distribution types"""

    # Name of the distribution, to be over-ridden by subclasses
    Name = None

    # Dictionary keys of apply()'s return value
    DICT_KEY_OUTLIERS = 'Outliers'
    DICT_KEY_BOUNDARY_RANGES = 'BoundaryRanges'
    DICT_KEY_DENSITIES = 'ProbabilityDensities'
    DICT_KEY_FULL_SAMPLE = 'FullSampledValue'
    DICT_KEY_SAMPLE = 'SampledValue'
    # Small enough std value to replace the std when it is 0 in normal or exponential distributions.
    REPLACE_ZERO_STD = 0.000001

    def __init__(self, **kwargs):
        self._mean = None
        self._std = None
        self._min = None  # minimum value of the training data set
        self._max = None  # maximum value of the training data set
        # (string) metric type that is used to calculate the distance (1/goodness of fit)
        self._metric = None
        # (float) the distance between the sampled density function and the training dataset according to metric
        self._distance = None
        # Size of the dataset the distribution is fitted over
        self._cardinality = None
        # Dict, containing distribution-specific parameters.
        # In summary command this dictionary will be printed in
        # a "param1: val1, param2: val2, ..." format.
        self._other_params = {}
        # Data that the distribution was fitted over. Some distributions may leave this unused.
        self._data = None
        # Reserved field for future additions. Unused now.
        self._extension = {}

    def fit(self, data, distance=None):
        """Fit data and update parameters. Optionally return distance
        measure if a measure is specified

        Args:
            data (numpy.array): Array of numbers to fit the distribution on
            distance (DistanceMetric): If not None, return distance
                                             using the provided metric

        Returns:
            (float): distance
        """
        raise NotImplementedError()

    def apply(self, data, threshold, params):
        """Apply distribution to the data and find outliers

        Args:
            data (numpy.array): Array of numbers to apply the distribution on
            threshold (OutlierThreshold) Anomaly detection threshold
            params (ApplyParams object): Includes show_density and sample parameter values,
            if true probability densities and/or sampled values from the distribution are also shown.

        Returns:
            np.OrderedDict[numpy.array]: An OrderedDictionary where:
                - the first item is an array of 0's and 1's where 1's show anomalies.
                - the second item is the boundary ranges
                - the third item is an array of probability densities for input values (if show_density is True)
                - the fourth item is an array of sampled values within the inlier area of the density function (if sample is True)
            NOTE: Don't forget to update The DICT_KEY_* values if you change the above semantics
        """
        raise NotImplementedError()

    def sample_within_boundaries(self, size, outlier_ranges_list):
        """Generate and return samples from the inlier area on the density function within the inlier area.

        Args:
            size (int): Size of sample
            outlier_ranges_list (list): includes a list of calculated anomaly ranges of the distribution. Includes one or several:
                Normal: [[-inf, lower], [upper, inf]]
                Exponential: [[upper, inf]]
                Gaussian KDE: [[beginning_1, end_1, area_1], [beginning_2, end_2, area_2], ..., [beginning_n, end_n, area_n]]
                Beta: [[-inf, lower]]

        Returns:
            (numpy.array): Array of samples within inlier area
        """

        def generate_filtering(samples, outlier_ranges):
            """Traverse the anomaly_ranges and generate the filtering True/False array
            which includes True for outlier locations of the samples.

            Args:
                samples (numpy.array): Array includes the sampled values from the density function
                outlier_ranges (list): includes the calculated anomaly ranges of the distribution.

            Returns:
                filtering (numpy.array): True/False array to be used for filtering valid samples later.
                    -True for inliers, False for outliers.
            """
            filtering = np.zeros(len(samples), dtype=bool)
            for anomaly_range in outlier_ranges:
                # Make sure that samples are not falling into the outlier region (False) if the outlier region start or end point is None: lower_threshold=0.01.
                # Ex: ((-inf, 1.3),(None, inf)): no boundary region for upper threshold.
                # At the first loop the samples which are not in between -inf and 1.3 will be set as False. filtering | False = filtering.
                # At the second loop all samples will be set to False: left=False, right=True (everything is less than inf), (left & right) = False. filtering | False = filtering.
                left = False if anomaly_range[0] is None else (samples > anomaly_range[0])
                right = False if anomaly_range[1] is None else (samples < anomaly_range[1])
                filtering = filtering | (left & right)
            # invert filtering (True-to-False/False-to-True) to set inlier locations to True
            return ~filtering

        samples_list = []
        samples = self.sample(size)
        for outlier_ranges in outlier_ranges_list:
            if not outlier_ranges:
                samples_list.append(samples)
            # filtering assigns True to inlier sample locations
            filtering = generate_filtering(samples, outlier_ranges)

            # extract the inlier/valid sample locations
            samples_valid = np.extract(filtering, samples)
            num_loops = 0
            while len(samples_valid) < size and num_loops < 100:
                # generating 3 times of the invalid samples to avoid multiple looping
                samples_tmp = self.sample(3 * (len(samples) - len(samples_valid)))
                filtering = generate_filtering(samples_tmp, outlier_ranges)
                samples_valid = np.append(samples_valid, np.extract(filtering, samples_tmp))
                num_loops = num_loops + 1
            if len(samples_valid) < size:
                raise SamplingError(
                    "Can not sample from the density function within the boundary ranges. The boundary ranges are too small, likely because the specified thresholds are too large."
                )
            samples_list.append(samples_valid[:size])
        return samples_list

    def sample(self, size):
        """Return a numpy array of samples from the underlying distribution

        Args:
            size (int): Size of sample

        Returns:
            (numpy.array): Array of samples
        """
        raise NotImplementedError()

    def summary(self):
        """Return a dictionary, summarizing the distribution.
        N.B.: The format of the returned dictionary must be the
        following:

        {
            'type': distribution type name,
            'mean': mean of the distribution
            'std': standard deviation of the distribution
            'cardinality': the number of data points that the distribution is fitted to
            'distance': the distance metric and value between the sampled distribution and the training dataset
            'other': a string, indicating other distribution parameters
        }
        """
        other = (
            'N/A'
            if not self._other_params
            else ', '.join(['{}: {}'.format(k, v) for k, v in list(self._other_params.items())])
        )
        # Since _min and _max values are not added to the created models in older versions of MLTK (<=4.3),
        # we first check if a created model has these values. If the model does not have _min and _max we get the
        # approximate min and max from the sampled dataset (with 5000 data points) from the density function.
        min = getattr(self, '_min', None)
        max = getattr(self, '_max', None)
        if min is None and max is None:
            samples = self.sample(5000)
            min = samples.min()
            max = samples.max()
        return {
            'type': self.Name,
            'min': min,
            'max': max,
            'mean': self._mean,
            'std': self._std,
            'cardinality': self._cardinality,
            'distance': "metric: " + str(self._metric) + ", distance: " + str(self._distance),
            'other': other,
        }

    def _get_distance(self, data, metric):
        """Get distance (1/goodness-of-fit) based on given data and distribution

        Args:
            data (numpy.array): Array of numbers to fit the distribution on
            metric (DistanceMetric): Metric for distance (1/Goodness-of-fit)

        Returns:
            (float): Value of distance measure
        """

        samples = self.sample(len(data))
        if metric == DistanceMetric.WASSERSTEIN:
            return stats.wasserstein_distance(data, samples)
        elif metric == DistanceMetric.KOLMOGOROV_SMIRNOV:
            return stats.ks_2samp(data, samples)[0]
        else:
            RuntimeError('Invalid measure {}'.format(metric))

    @staticmethod
    def _create_result(outliers, boundary_ranges, densities, full_samples, samples):
        """Create output ndarray.
        Args:
            outliers (np.array): Array indicating outliers
            boundary_ranges (string): Outlier boundaries as a JSON string
            densities (np.array): Array of probability densities
            full_samples (np.array): Array of sampled values from the density function
            samples (np.array): Array of sampled values from the density function within the inlier area

        Returns:
            res (np.OrderedDict): Ordered Dictionary of resulting lists
        NOTE: Don't forget to update the TUPLE_IDX_* values if you change the order of items in the returned list
        """
        res = OrderedDict([('Outliers', outliers), ('BoundaryRanges', boundary_ranges)])
        if densities is not None:
            res['ProbabilityDensities'] = densities
        if full_samples is not None:
            res['FullSampledValue'] = full_samples
        if samples is not None:
            res['SampledValue'] = samples
        return res

    def get_name(self):
        return self.Name


class ApplyParams(object):
    """Class that holds the boolean apply parameters that determine if the respective results will be shown or not.
    """

    def __init__(self, show_density=False, full_sample=False, sample=False, show_options=None):
        self.show_density = show_density
        self.full_sample = full_sample
        self.sample = sample
        self.show_options = show_options

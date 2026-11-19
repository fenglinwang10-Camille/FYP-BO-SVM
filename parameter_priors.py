"""Bounded inverse-CDF parameter priors for Bayesian optimization."""

from dataclasses import dataclass

from scipy.stats import expon, gamma as gamma_distribution, norm


SUPPORTED_PRIORS = ("uniform", "normal", "exponential", "gamma")
CDF_EPSILON = 1e-6


@dataclass(frozen=True)
class ParameterBounds:
    low: float
    high: float

    @property
    def width(self):
        return self.high - self.low


PREPROCESSING_BOUNDS = {
    "f_low": ParameterBounds(0.5, 5.0),
    "f_high": ParameterBounds(30.0, 60.0),
    "window_size": ParameterBounds(1.0, 5.0),
}


def _distribution(prior, bounds):
    """Return the untruncated distribution used before boundary truncation."""
    if prior == "normal":
        return norm(loc=(bounds.low + bounds.high) / 2.0, scale=bounds.width / 4.0)
    if prior == "exponential":
        return expon(loc=bounds.low, scale=bounds.width / 3.0)
    if prior == "gamma":
        return gamma_distribution(a=2.0, loc=bounds.low, scale=bounds.width / 4.0)
    raise ValueError(f"Unsupported inverse-CDF prior: {prior}")


def latent_to_parameter(u, prior, bounds):
    """Map u in [0, 1] to a parameter constrained to [low, high]."""
    u = min(max(float(u), CDF_EPSILON), 1.0 - CDF_EPSILON)
    if prior == "uniform":
        return bounds.low + u * bounds.width

    distribution = _distribution(prior, bounds)
    lower_cdf = distribution.cdf(bounds.low)
    upper_cdf = distribution.cdf(bounds.high)
    probability = lower_cdf + u * (upper_cdf - lower_cdf)
    return float(distribution.ppf(probability))


def parameter_to_latent(value, prior, bounds):
    """Invert a bounded parameter value back to the BO latent coordinate."""
    value = min(max(float(value), bounds.low), bounds.high)
    if prior == "uniform":
        return (value - bounds.low) / bounds.width

    distribution = _distribution(prior, bounds)
    lower_cdf = distribution.cdf(bounds.low)
    upper_cdf = distribution.cdf(bounds.high)
    u = (distribution.cdf(value) - lower_cdf) / (upper_cdf - lower_cdf)
    return min(max(float(u), CDF_EPSILON), 1.0 - CDF_EPSILON)


def transform_preprocessing_parameters(values, prior):
    """Transform the three BO preprocessing coordinates to physical values."""
    if prior not in SUPPORTED_PRIORS:
        raise ValueError(f"prior must be one of {SUPPORTED_PRIORS}, got {prior!r}")
    return {
        name: latent_to_parameter(values[index], prior, bounds)
        for index, (name, bounds) in enumerate(PREPROCESSING_BOUNDS.items())
    }


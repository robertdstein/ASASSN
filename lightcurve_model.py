import numpy as np

def default(max_y):
    """Returns an array with typical values for each fit parameter,
    lying well within the strict parameter bounds

    :param y: max counts
    :return: Array containing starting values
    """
    return [max_y, 5 * 10 ** -4, 5, -5., 0.0]

def return_loose_bounds():
    """Return loose parameter bounds for the fit, based on purely
    on physical motivations.

    :return: Array containing bounds for parameters
    """
    return[(None,None), (10**-6, None), (2., 350),
           (None, -10**-6), (None, None)]


def logpeak(x, p):
    """Returns a value for the peak part of the lightcurve, which is a
    Gaussian that in logspace becomes a quadratic function

    :param x: Shifted Time (x - offset)
    :param p: Fit parameter array
    :return: Value of lightcurve at time x for peak law
    """
    model = p[0] - p[1]*(x**2)
    return model


def logcontinuity(p):
    """Function to find the point of intersection between the two models,
    and calculate the consequent gradient

    :param p: Fit Parameter Array
    :return: Values of Time, Light Curve and Gradient at transition
    """
    ytr = logpeak(p[2], p)
    xtr = p[2]
    gradtr = -2 * p[1] * p[2]
    return xtr, ytr, gradtr


def logpowerlaw(x, p):
    """Returns a value for the Power Law part of the lightcurve,
    which requires calculation of the transition ponit, in order to ensure
    continuity of the two components

    :param x: Shifted Time (x - offset)
    :param p: Fit parameter array
    :return: Value of Lightcurve at time x for power law
    """
    xtr, ytr, gradtr = logcontinuity(p)
    power = p[3]
    x0 = xtr - power/gradtr
    b = ytr - power*np.log(xtr-x0)
    return b + power*np.log(x-x0)


def fitfunc(x_unshifted, p):
    """Returns a value for the model at a time X_unshifted, using the
    parameter array p. Checks whether the point lies in the peak or power
    component, and returns the corresponding value

    :param x_unshifted: Unshifted time
    :param p: Fit parameter array
    :return: Value of Light Curve at x
    """
    x = np.array(x_unshifted+p[4])
    xtr, ytr, gradtr = logcontinuity(p)

    y = np.ones_like(x) * np.nan

    mask = x < xtr
    # other = np.array([bool(1-z) for z in mask])

    y[mask] = logpeak(x[mask], p)
    y[~mask] = logpowerlaw(x[~mask], p)
    return y

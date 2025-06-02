import warnings


class Waypoint:
    def __init__(self, lat=0, lon=0, alt=0, hold=10, relative_to=None):
        if relative_to is None:
            self.lat = lat
            self.lon = lon
            self.alt = alt
            self.hold = hold
            return
        else:
            self.x = lat - relative_to[0]
            self.y = lon - relative_to[1]
            self.z = alt
            self.hold = hold


def deprecated_method(func):
    """Decorator to mark methods as deprecated"""

    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Call to deprecated method {func.__name__}.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = f"[DEPRECATED] {func.__doc__}"
    return wrapper

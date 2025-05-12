import warnings
class Waypoint:
	def __init__(self, lat, lon, alt, hold=0, relative_to=None):
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

	def __repr__(self):
		return f"Waypoint(lat={self.x}, lon={self.y}, alt={self.z})"


def deprecated_method(func):
    """Decorator to mark methods as deprecated"""
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Call to deprecated method {func.__name__}.",
            category=DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = f"[DEPRECATED] {func.__doc__}"
    return wrapper


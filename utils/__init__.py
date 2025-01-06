if __name__ == 'utils':
    from .openpose import OPUtils
else:
    from openpose import OPUtils

__all__ = [
    "OPUtils"
]
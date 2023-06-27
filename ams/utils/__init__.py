from andes.utils.misc import elapsed

from ams.utils import paths  # NOQA
from ams.utils.paths import get_case  # NOQA


def timer(func):
    def wrapper(*args, **kwargs):
        t0, _ = elapsed()
        result = func(*args, **kwargs)
        _, s = elapsed(t0)
        return result, s
    return wrapper

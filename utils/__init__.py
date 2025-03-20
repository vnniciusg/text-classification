from functools import wraps
from time import perf_counter

from loguru import logger


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} took {perf_counter() - start:.2f} seconds")
        return result

    return wrapper

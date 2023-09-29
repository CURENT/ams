import logging

logger = logging.getLogger(__name__)


def dummify(input):
    if isinstance(input, str):
        return input
    else:
        return input

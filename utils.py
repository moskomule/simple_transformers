from functools import wraps
from typing import Callable, Optional

from homura import init_distributed, is_distributed


def distributed_ready_main(func: Callable = None,
                           backend: Optional[str] = None,
                           init_method: Optional[str] = None,
                           disable_distributed_print: str = False
                           ) -> Callable:
    """ Wrap a main function to make it distributed ready
    """

    if is_distributed():
        init_distributed(backend=backend, init_method=init_method, disable_distributed_print=disable_distributed_print)

    @wraps(func)
    def inner(*args, **kwargs):
        return func(*args, **kwargs)

    return inner

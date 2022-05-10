import functools
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, cast

_Fun = TypeVar("_Fun", bound=Callable[..., Any])


@dataclass(frozen=True)
class log_task_name_and_time:
    name: str

    def __call__(self, f: _Fun) -> _Fun:
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            logging.info(f"Running task {self.name}")
            start_time = time.process_time()
            value = f(*args, **kwargs)
            end_time = time.process_time()
            run_time = end_time - start_time
            logging.debug(f"CPU time elapsed {self.name} in {run_time:.4f} secs")
            return value

        return cast(_Fun, wrapped)

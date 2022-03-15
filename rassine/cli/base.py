from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from configpile import AutoName, Config, Param, types
from typing_extensions import Annotated


@dataclass(frozen=True)
class RassineConfig(Config):

    env_prefix_ = "RASSINE"

    #: Use the specified configuration files.
    #:
    #: Files can be separated by commas/the command can be invoked multiple times.
    config: Annotated[Sequence[Path], Param.config(env_var_name=AutoName.DERIVED)]

    #: Root path of the data, used as a base for other relative paths
    root: Annotated[Path, Param.store(types.path, env_var_name=AutoName.DERIVED)]

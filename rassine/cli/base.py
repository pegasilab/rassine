from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Sequence

from configpile import AutoName, Config, Param, ParamType, Result, errors, types
from typing_extensions import Annotated


@dataclass(frozen=True)
class RelPath:
    path: Path

    class _Type(types.ParamType["RelPath"]):
        def parse(self, arg: str) -> Result[RelPath]:
            return errors.wrap_exceptions(lambda: RelPath.from_str(arg))

    param_type: ClassVar[ParamType[RelPath]] = _Type()

    @staticmethod
    def from_str(s: str) -> RelPath:
        return RelPath(Path(s))

    def __rlshift__(self, other: Path) -> Path:
        return other / self.path


@dataclass(frozen=True)
class RassineConfig(Config):

    env_prefix_ = "RASSINE"

    #: Use the specified configuration files.
    #:
    #: Files can be separated by commas/the command can be invoked multiple times.
    config: Annotated[Sequence[Path], Param.config(env_var_name=AutoName.DERIVED)]

    #: Root path of the data, used as a base for other relative paths
    root: Annotated[Path, Param.store(types.path, env_var_name=AutoName.DERIVED)]

    #: Pickle protocol version to use
    pickle_protocol: Annotated[int, Param.store(types.int_, default_value="3")]


# TODO: change default pickle protocol to 4, default starting Python 3.8

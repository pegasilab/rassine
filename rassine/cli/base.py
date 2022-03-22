from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, Sequence, Union

import numpy as np
import pandas as pd
from configpile import AutoName, Config, Err, Param, ParamType, Res, Validator, types, userr
from typing_extensions import Annotated

from ..tybles import Row, Table


def _path_is_absolute(p: Path) -> bool:
    return p.is_absolute()


def _path_is_relative(p: Path) -> bool:
    return not p.is_absolute()


@dataclass(frozen=True)
class RelPath:
    path: Path

    @staticmethod
    def from_str(s: str) -> RelPath:
        return RelPath(Path(s))

    def __post_init__(self) -> None:
        assert not self.path.is_absolute(), "Relative paths cannot be absolute"


relPath: ParamType[RelPath] = types.path.validated(
    _path_is_relative, "Relative path cannot be absolute"
).map(RelPath)

_RelPathLike = Union[RelPath, str]


@dataclass(frozen=True)
class RootPath:
    path: Path

    def __post_init__(self) -> None:
        assert self.path.is_absolute(), "Root path must be absolute"

    def at(self, *rel_paths: _RelPathLike) -> Path:
        res: Path = self.path
        for r in rel_paths:
            if isinstance(r, RelPath):
                res = res / r.path
            elif isinstance(r, str):
                res = res / r
            else:
                raise ValueError(f"Cannot use type {type(r)}")
        return res


rootPath: ParamType[RootPath] = types.path.validated(
    _path_is_absolute, "Root path must be absolute"
).map(RootPath)


@dataclass(frozen=True)
class RassineConfig(Config):

    env_prefix_ = "RASSINE"

    #: Use the specified configuration files.
    #:
    #: Files can be separated by commas/the command can be invoked multiple times.
    config: Annotated[Sequence[Path], Param.config(env_var_name=AutoName.DERIVED)]

    #: Root path of the data, used as a base for other relative paths
    root: Annotated[RootPath, Param.store(rootPath, env_var_name=AutoName.DERIVED)]

    #: Pickle protocol version to use
    pickle_protocol: Annotated[int, Param.store(types.int_, default_value="3")]


@dataclass(frozen=True)
class RassineConfigBeforeStack(RassineConfig):

    #: Master table file path (CSV format)
    input_master_table: Annotated[
        RelPath,
        Param.store(relPath, env_var_name=AutoName.DERIVED, short_flag_name="-t"),
    ]

    def validate_input_master_table(self) -> Validator:
        if self.input_master_table is None:
            return None
        fn = self.root.at(self.input_master_table)
        if not fn.is_file():
            return Err.make(f"Meta table {fn} must exist")
        else:
            return None


# TODO: change default pickle protocol to 4, default starting Python 3.8


@dataclass(frozen=True)
class BasicInfo(Row):
    """
    Basic information about spectra given as input to RASSINE
    """

    #: Full filename including folders, folders may be outdated
    fileroot: str

    #: Name only of the file itself, excluding folders
    filename: str

    #: Observation date/time in MJD
    mjd: np.float64

    #: Optional RV shift correction
    model: Optional[np.float64]

    @classmethod
    def after_read_(cls, table: pd.DataFrame) -> pd.DataFrame:
        if "filename" not in table.columns:
            table["filename"] = [str(Path(f).name) for f in table["fileroot"]]
        return table

    @classmethod
    def before_write_(cls, table: pd.DataFrame) -> pd.DataFrame:
        table.drop("filename")
        return table

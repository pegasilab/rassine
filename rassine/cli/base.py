from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, Sequence

import numpy as np
import pandas as pd
from configpile import AutoName, Config, Err, Param, ParamType, Result, Validator, errors, types
from typing_extensions import Annotated

if TYPE_CHECKING:
    from dataclasses import dataclass as pdataclass
else:
    from pydantic.dataclasses import dataclass as pdataclass

from ..tybles import Row, Table


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


@dataclass(frozen=True)
class RassineConfigBeforeStack(RassineConfig):

    #: Master table file path (CSV format)
    input_master_table: Annotated[
        RelPath,
        Param.store(RelPath.param_type, env_var_name=AutoName.DERIVED, short_flag_name="-t"),
    ]

    def validate_input_master_table(self) -> Validator:
        if self.input_master_table is None:
            return None
        fn = self.root << self.input_master_table
        if not fn.is_file():
            return Err.make(f"Meta table {fn} must exist")
        else:
            return None


# TODO: change default pickle protocol to 4, default starting Python 3.8


@pdataclass(frozen=True)
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

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, fields
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Mapping,
    Sequence,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import pandas as pd

if TYPE_CHECKING:
    from dataclasses import dataclass as pdataclass
else:
    from pydantic.dataclasses import dataclass as pdataclass


def _is_optional(t: type) -> bool:
    """
    Returns whether the given type is Optional[X]
    """
    if get_origin(t) is not Union:
        return False
    return type(None) in get_args(t)


@pdataclass(frozen=True)
class Row(ABC):
    """
    Describes a typed row in a Pandas dataframe
    """

    @classmethod
    def fields_(cls) -> Mapping[str, bool]:
        """
        Returns a dictionary of fields along with whether they are optional
        """
        return {f.name: _is_optional(f.type) for f in fields(cls)}

    @classmethod
    def after_read_(cls, table: pd.DataFrame) -> pd.DataFrame:
        """
        Method called after reading a Pandas frame to perform optional processing

        Args:
            table: Table to process, is considered mutable

        Returns:
            The processed table
        """
        pass

    @classmethod
    def before_write_(cls, table: pd.DataFrame) -> pd.DataFrame:
        """
        Method called before writing a Pandas frame to perform optional processing

        Args:
            table: Table to process, is considered mutable

        Returns:
            The processed table
        """


R = TypeVar("R", bound=Row)  #: Row type


@dataclass(frozen=True)
class Table(Generic[R]):
    """
    Typed table
    """

    table: pd.DataFrame
    rowtype: Type[R]

    def nrows(self) -> int:
        """
        Returns the number of rows in the table

        Returns:
            Number of rows
        """
        return len(self.table.index)

    def all(self) -> Sequence[R]:
        """
        Returns all rows of this table
        """
        return [self[i] for i in self.table.index]

    @staticmethod
    def from_values(values: Sequence[R], rowtype: Type[R]) -> Table[R]:
        """

        Args:
            rowtype:
            values:

        Returns:

        """
        table = pd.DataFrame(values)
        return Table(table, rowtype)

    @staticmethod
    def read_csv(csv_file: Path, rowtype: Type[R]) -> Table[R]:
        """
        Reads a CSV file which represents a Table

        Args:
            csv_file: Path of file to read

        Returns:
            The table
        """
        table = pd.read_csv(csv_file, float_precision="round_trip")
        n = len(table.index)
        assert list(table.index) == list(
            range(n)
        ), "Indexing of table elements should start at 0 and be contiguous"
        rowtype.after_read_(table)
        return Table(table, rowtype)

    def write_csv(self, csv_file: Path) -> None:
        """

        Args:
            csv_file: Path of the file to write
        """
        to_write = self.table.copy(deep=True)
        self.rowtype.before_write_(to_write)
        to_write.to_csv(csv_file)

    def __getitem__(self, i: int) -> R:
        """
        Retrieves a row from the table

        Args:
            i: Row index

        Returns:
            A row dataclass
        """
        dt = self.rowtype
        data: Dict[str, Any] = self.table.iloc[i].to_dict()
        f = dt.fields_()
        for name, is_opt in f.items():
            if name not in data:
                if is_opt:
                    data[name] = None
                else:
                    raise KeyError(name)
        return dt(**{k: v for k, v in data.items() if k in f})

"""
Various dataclasses for standard RASSINE configuration parameters
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import configpile as cp
import tybles as tb

from . import io


@dataclass(frozen=True)
class NameRow:
    """
    Describes a table with a name column
    """

    #: Stacked spectrum name without path and extension
    name: str

    @staticmethod
    def schema() -> tb.Schema[NameRow]:
        return tb.schema(
            NameRow,
            order_columns=True,
            missing_columns="error",
            extra_columns="drop",
        )


@dataclass(frozen=True)
class PathPattern:
    """Path fragment, contains the relative path + filename pattern containing a {name} substring"""

    #: Relative path, possibly containing "{}"
    path: Path

    @staticmethod
    def parser() -> cp.Parser[PathPattern]:
        """Returns a configpile parser for a PathPattern"""
        return cp.parsers.path_parser.map(PathPattern)

    def to_path(self, root: Path, replacement: str) -> Path:
        """
        Returns a full path from this fragment

        Args:
            root: Absolute root path
            replacement: String to put in place of {name} in the fragment
        """
        assert root.is_absolute(), "Root path must be absolute"
        return root / Path(*[s.replace("{name}", replacement) for s in self.path.parts])


@dataclass(frozen=True)
class LoggingLevel:
    """Describes a logging level"""

    #: Logging level integer value, after parsing
    int_value: int

    @staticmethod
    def _parse(arg: str) -> cp.Res[LoggingLevel]:
        try:
            return LoggingLevel(int(arg))
        except ValueError:
            pass
        res = logging.getLevelName(arg)
        if isinstance(res, int):
            return LoggingLevel(res)
        return cp.Err.make(f"Cannot parse logging level {arg}")

    @classmethod
    def parser(cls) -> cp.Parser[LoggingLevel]:
        """Returns a parser for a logging level"""
        return cp.Parser.from_function(LoggingLevel._parse)

    def set(self, logger: Optional[logging.Logger] = None) -> None:
        """Sets the logging level in the provided logger

        Args:
            logger: Logger to update, or root logger (if :data:`None`)
        """
        if logger is None:
            logger = logging.getLogger()
        logger.setLevel(self.int_value)


@dataclass(frozen=True)
class PickleProtocol:
    """Describes a pickle protocol level"""

    level: int

    @classmethod
    def parser(cls) -> cp.Parser[PickleProtocol]:
        """Returns a parser for a Pickle protocol"""
        return cp.parsers.int_parser.map(PickleProtocol)

    def set(self) -> None:
        """Sets the pickle protocol level as parsed"""
        io.default_pickle_protocol = self.level

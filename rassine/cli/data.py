from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import configpile as cp

from .. import io


@dataclass(frozen=True)
class LoggingLevel:
    """Describes a logging level"""

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
        return cp.Parser.from_function(LoggingLevel._parse)

    def set(self, logger: Optional[logging.Logger] = None) -> None:
        """Sets the logging level in the provided logger

        Args:
            logger: Logger to update, or root logger (if :py:`None`)
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
        return cp.parsers.int_parser.map(PickleProtocol)

    def set(self) -> None:
        """Sets the pickle protocol level as parsed"""
        io.default_pickle_protocol = self.level

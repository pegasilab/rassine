from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import numpy
    import numpy.typing

    Float = Union[float, numpy.float64]
    NFArray = numpy.typing.NDArray[numpy.float64]
    Int = Union[int, numpy.int64]
    NIArray = numpy.typing.NDArray[numpy.int64]

absurd_minus_99_9: float = -99.9

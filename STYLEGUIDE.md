


## Floating point types

They should be equivalent on the platforms we run our code on. `float` and `np.float_` and
`np.float64`
are all double precision floating point values which are IEEE 788 64 bits.

We write `np.float64` explicitly when dealing with I/O.

We have a `Float` alias available to work interchangeably with `float` and `np.float64`.
Other floating-point types should be avoided, except `float32` for compact storage, and
of course complex numbers when relevant.
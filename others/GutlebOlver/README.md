# SparseVolterraNew

This directory contains a Julia 1.11.1 compatibility migration of the Gutleb--Olver / SparseVolterra core Volterra matrix construction code.

The goal is compatibility and fair benchmarking against the current VoltOpt implementation.  It is not an algorithmic refactor: the interpolation, sampling, projection, Clenshaw recurrence, and strict-banded matrix assembly flow are preserved from the existing SparseVolterra code and copied into `new/src/` before modification.

Run tests:

```bash
cd new
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.test()"
```

Run the smoke test:

```bash
julia --project=. scripts/smoke_run.jl
```

Run the construction benchmark:

```bash
julia --project=. scripts/bench_construct.jl
```

The correctness tests compare the migrated strict-banded construction against the original dense Clenshaw construction for the same projected smooth kernel `K(x, y) = exp(-x*y)`.  This checks the intended finite-dimensional banded approximation target, not machine-precision equality to the infinite-dimensional integral operator.

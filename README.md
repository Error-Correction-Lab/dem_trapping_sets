# Trapping-Set Enumeration in Circuit-Level Detector Error Models

This repository contains the trapping-set enumeration code used to study low-weight failure mechanisms in circuit-level detector error models (DEMs) of quantum LDPC codes.

The core search is implemented in C++ and exposed to Python through `pybind11`. The implementation enumerates leafless elementary trapping sets (LETSs) using the DPL expansion procedure, with optional enumeration of elementary trapping sets with leaves (ETSLs). The search is designed for the irregular Tanner graphs that arise from circuit-level DEMs and uses an external-memory implementation so that large intermediate frontiers do not need to be stored entirely in RAM.

The repository also contains a Python example that builds circuit-level DEMs for several bivariate-bicycle (BB) codes and runs the trapping-set search directly on the resulting detector matrix.

## Repository structure

The relevant files are organized as

```text
dem_trapping_sets/
├── cpp/
│   └── include/
│       ├── dpl-search.hpp
│       ├── dpl-search-binder.cpp
│       ├── tannergraph_parallelized_csr.hpp
│       └── setup-dpl-search.py
│
└── examples/
    ├── enumerate_dem_ts.py
    └── include_py/
        ├── build_decoding_matrices_layered.py
        ├── codes_q.py
        └── utils.py
```

The main components are:

- `dpl-search.hpp`: C++ implementation of the trapping-set search.
- `dpl-search-binder.cpp`: `pybind11` interface exposing the C++ enumerator to Python.
- `tannergraph_parallelized_csr.hpp`: Tanner-graph representation used by the C++ search.
- `setup-dpl-search.py`: build script for the Python extension.
- `enumerate_dem_ts.py`: example driver that builds a circuit-level DEM and enumerates trapping sets.
- `examples/include_py/`: helper routines for constructing the BB codes, syndrome-extraction circuits, and DEM matrices.

## Requirements

The example code requires Python 3.10 or newer.

The main Python dependencies are

```text
numpy
scipy
stim
pybind11
setuptools
```

They can be installed with

```bash
python -m pip install numpy scipy stim pybind11 setuptools
```

A C++17 compiler is also required.

On Linux, a recent GCC or Clang compiler is sufficient. For example, on Ubuntu:

```bash
sudo apt install build-essential
```

On macOS, the Xcode command-line tools provide a suitable Clang compiler:

```bash
xcode-select --install
```

On Windows, the extension can be built with the Microsoft Visual C++ Build Tools.

## Building the C++ extension

The Python extension is built in place inside `cpp/include`.

From the repository root, run

```bash
cd cpp/include
python setup-dpl-search.py build_ext --inplace
```

This produces a platform-dependent compiled module in the same directory, for example

```text
dpl_search.cpython-312-x86_64-linux-gnu.so
```

on Linux, or the corresponding `.pyd` file on Windows.

The example script automatically adds `cpp/include` to the Python search path, so no separate installation of `dpl_search` is required.

A quick import test can be performed from `cpp/include` with

```bash
python -c "import dpl_search; print(dpl_search.__doc__)"
```

The expected module description is

```text
External-memory DPL trapping-set enumerator for irregular Tanner graphs
```

If the C++ source has been changed, it is useful to perform a clean rebuild:

```bash
rm -rf build
rm -f dpl_search*.so
python setup-dpl-search.py build_ext --inplace
```

On Windows, simply delete the `build` directory and the previously generated `.pyd` file before rebuilding.

## Search features

### LETS enumeration

The main search enumerates leafless elementary trapping sets using DPL expansions. The implementation includes the three standard expansion types:

- dot expansion;
- path expansion;
- lollipop expansion.

The allowed expansions are determined by the class-dependent bounds \(b_{\max}^a\) used by Algorithm 1. The example Python script computes these bounds recursively before calling the C++ enumerator.

For a target search with maximum support size `amax` and final maximum `bmax`, intermediate classes may require larger values of \(b\). These auxiliary bounds are handled automatically through `bmax_by_a`.

### Irregular Tanner graphs

The implementation is not restricted to variable degree two or three. It supports irregular Tanner graphs and accepts an arbitrary set of positive variable-node degrees.

In the provided example, the variable-node degrees are read directly from the DEM matrix and passed to the C++ search:

```python
degrees = sorted(
    np.unique(np.asarray(H.sum(axis=0)).ravel().astype(int)).tolist()
)
```

This is necessary for circuit-level DEMs, whose Tanner graphs generally have irregular degree distributions.

### External-memory search

The enumeration uses a disk-backed external-memory implementation.

Candidate supports are accumulated in bounded in-memory buffers, sorted, written to temporary binary runs, and then merged and deduplicated on disk. Accepted intermediate structures are also maintained as disk-backed frontiers.

This avoids keeping the complete search frontier in RAM and makes searches with very large numbers of intermediate trapping sets possible.

The temporary working directory is created inside the requested output directory with a name of the form

```text
.dpl_tmp_...
```

It is removed automatically after a successful search. If the search terminates with an exception, the temporary directory is left in place and its location is printed, which can be useful for debugging.

The current public implementation is serial. Multithreaded candidate generation and anchored-search variants used during development have been removed.

### Optional ETS-with-leaves enumeration

The search can optionally be extended from LETSs to elementary trapping sets with leaves using repeated \(\mathrm{dot}_1\) expansions.

Enable this mode from the example script with

```bash
--etsl
```

By default, the ETSL stage is initialized from the LETSs found by the DPL search.

A more exhaustive singleton-seeded ETSL search can be enabled with

```bash
--etsl --singleton-etsl-seeds
```

In this mode, single variable nodes are also used as initial ETSL seeds. This can generate substantially more intermediate structures and can therefore be much more expensive.

### Disk output

Final trapping-set supports are streamed directly to CSV files instead of being returned to Python in memory.

For each nonempty \((a,b)\) class, the output file is named

```text
a_<a>_b_<b>.csv
```

For example,

```text
a_4_b_4.csv
a_5_b_3.csv
a_5_b_5.csv
```

Each row contains the variable-node indices of one trapping-set support, separated by commas.

The Python interface also returns a count matrix and several search statistics, including

- number of written supports;
- number of generated candidates;
- number of accepted intermediate structures;
- number of duplicate candidates;
- number rejected by support size;
- number rejected by the \(b\) bound;
- number rejected because the induced subgraph is not elementary.

### Maximum compiled support size

The current Python binder is compiled with

```cpp
COMPILED_AMAX = 32;
```

and therefore accepts searches with

```text
amax <= 32
```

This limit can be increased by changing `COMPILED_AMAX` in `dpl-search-binder.cpp` and rebuilding the extension.

## Bivariate-bicycle code presets

The example script includes the following BB codes:

| `--code` | Code | Default rounds |
|---|---|---:|
| `72` | `[[72,12,6]]` | 6 |
| `90` | `[[90,8,10]]` | 10 |
| `108` | `[[108,8,10]]` | 10 |
| `144` | `[[144,12,12]]` | 12 |
| `288` | `[[288,12,18]]` | 18 |

The aliases `gross` and `bb144` can be used for the `[[144,12,12]]` code. The aliases `two-gross`, `twogross`, and `bb288` can be used for the `[[288,12,18]]` code.

The number of syndrome-extraction rounds can be overridden with `--rounds`.

The example currently constructs the DEM at physical error rate

```python
PHYSICAL_ERROR_RATE = 1e-3
```

and uses Tanner girth

```python
GIRTH = 4
```

These values are defined near the top of `examples/enumerate_dem_ts.py`.

## Running the example

The general command is

```bash
python examples/enumerate_dem_ts.py \
    --code CODE \
    --output-dir OUTPUT_DIR \
    --amax AMAX \
    --bmax BMAX
```

The `[[144,12,12]]` code is used if `--code` is omitted.

### Small test

A relatively small test can be run on the `[[72,12,6]]` code with

```bash
python examples/enumerate_dem_ts.py \
    --code 72 \
    --output-dir results/bb72_test \
    --amax 3 \
    --bmax 3
```

The script will

1. construct the selected BB code;
2. build its syndrome-extraction circuit;
3. convert the circuit to a Stim detector error model;
4. construct the full detector matrix;
5. determine the variable-node degree set;
6. compute the class-dependent `bmax_by_a` bounds;
7. run the C++ DPL enumerator;
8. write the enumerated supports to `results/bb72_test/`.

At the end of the run, the script prints the number of trapping sets found in every nonempty \((a,b)\) class, the total number of written supports, and the elapsed search time.

To repeat the same search and replace an existing output directory, use

```bash
python examples/enumerate_dem_ts.py \
    --code 72 \
    --output-dir results/bb72_test \
    --amax 3 \
    --bmax 3 \
    --overwrite
```

### LETS + ETSL example

To include the optional ETS-with-leaves stage:

```bash
python examples/enumerate_dem_ts.py \
    --code 72 \
    --output-dir results/bb72_etsl \
    --amax 3 \
    --bmax 3 \
    --etsl
```

For the singleton-seeded ETSL search:

```bash
python examples/enumerate_dem_ts.py \
    --code 72 \
    --output-dir results/bb72_etsl_singletons \
    --amax 3 \
    --bmax 3 \
    --etsl \
    --singleton-etsl-seeds
```

## Command-line options

The example script exposes the following options:

```text
--code CODE
```

Select the BB code. Available presets are `72`, `90`, `108`, `144`, and `288`.

```text
--rounds R
```

Override the default number of syndrome-extraction rounds for the selected code.

```text
--output-dir PATH
```

Directory in which the trapping-set CSV files are written. This option is required.

```text
--amax A
```

Maximum trapping-set support size. The default is `5`.

```text
--bmax B
```

Maximum target value of \(b\). The default is `5`.

```text
--etsl
```

Enable the ETS-with-leaves stage.

```text
--singleton-etsl-seeds
```

Also initialize the ETSL stage from single variable nodes. This option requires `--etsl`.

```text
--overwrite
```

Delete and replace an existing nonempty output directory.

```text
--quiet
```

Disable progress messages from the C++ enumerator.

The complete command-line help can be displayed with

```bash
python examples/enumerate_dem_ts.py --help
```

## Calling the C++ enumerator directly from Python

The BB-code example is only a driver. The compiled module can also be called directly for another binary Tanner graph.

The input matrix should be in SciPy CSR format, with rows corresponding to check nodes and columns corresponding to variable nodes. The binder receives the CSR `indptr` and `indices` arrays.

A minimal call has the form

```python
result = dpl_search.enumerate(
    M,
    N,
    H.indptr.astype(np.int32, copy=False),
    H.indices.astype(np.int32, copy=False),
    g=4,
    amax=5,
    bmax_by_a=bmax_by_a,
    output_dir="results/my_search",
    allowed_variable_degrees=degrees,
    include_ets_with_leaves=False,
    include_singleton_etsl_seeds=False,
    verbose=True,
)
```

Here:

- `M` is the number of check nodes;
- `N` is the number of variable nodes;
- `g` is the Tanner-graph girth;
- `amax` is the maximum support size;
- `bmax_by_a[a]` is the maximum allowed intermediate \(b\) value at support size \(a\);
- `allowed_variable_degrees` specifies the variable degrees permitted in the search;
- `output_dir` is the directory used for the CSV output and temporary external-memory files.

An empty `allowed_variable_degrees` vector allows every positive variable-node degree.

## Notes on computational cost

Trapping-set enumeration is combinatorial. Runtime and disk usage can increase rapidly with `amax`, with the allowed intermediate \(b\) bounds, and with the size and density of the Tanner graph.

In particular:

- increasing `amax` can increase the number of intermediate structures by orders of magnitude;
- circuit-level DEMs can be substantially denser and more irregular than conventional LDPC Tanner graphs;
- enabling ETSL generally increases the search space;
- enabling singleton ETSL seeds can increase it very substantially.

For this reason, it is recommended to first test a new code or DEM with small values of `amax` and `bmax` before starting a larger enumeration.

## Citation

If you use this code in academic work, please cite the associated paper. Citation information will be added here when the paper is publicly available.

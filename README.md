# TuringBot Python Library

Python interface for [TuringBot](https://turingbotsoftware.com), a symbolic regression software that discovers mathematical formulas from data.

TuringBot uses simulated annealing to search the space of possible mathematical expressions, finding accurate and compact formulas that describe your data. It supports 15 search metrics, train/test splits, and a wide range of mathematical functions.

## Installation

```bash
pip install turingbot
```

You also need the TuringBot desktop application installed. Download it from [turingbotsoftware.com/download.html](https://turingbotsoftware.com/download.html).

## Quick start

```python
import time
from turingbot import simulation

sim = simulation()

# Start a symbolic regression search
# TuringBot is detected automatically if installed in the default location
sim.start_process(
    input_file="your_data.csv",  # CSV with input columns + target column
    search_metric=4,             # RMS error
    threads=4,
)

# Let TuringBot search for formulas
time.sleep(60)

# Read the best formulas found so far
sim.refresh_functions()
for f in sim.functions:
    print(f)

# Stop the search
sim.terminate_process()
```

Or using the context manager:

```python
import time
from turingbot import simulation

with simulation() as sim:
    sim.start_process(
        input_file="your_data.csv",
        search_metric=4,
        threads=4,
    )

    time.sleep(60)
    sim.refresh_functions()

    for f in sim.functions:
        print(f)
```

If TuringBot is installed in a non-default location, pass the path explicitly:

```python
sim.start_process(
    input_file="your_data.csv",
    path="/custom/path/to/TuringBot",
)
```

Default paths checked automatically:

| OS | Path |
|----|------|
| Windows | `C:\Program Files (x86)\TuringBot\TuringBot.exe` |
| macOS | `/Applications/TuringBot.app/Contents/MacOS/TuringBot` |
| Linux | `/usr/lib/turingbot/TuringBot` |

## Using numpy arrays and pandas DataFrames

You can pass numpy arrays or pandas DataFrames directly instead of file paths:

```python
import numpy as np
from turingbot import simulation

# Generate some data: y = 2*x1 + x2^2
data = np.column_stack([
    x1 := np.random.rand(100),
    x2 := np.random.rand(100),
    2 * x1 + x2 ** 2,
])

with simulation() as sim:
    sim.start_process(
        input_file=data,
        column_names=["x1", "x2", "y"],
    )
    # ...
```

If `column_names` is not provided, TuringBot will use its default names (`col1`, `col2`, ...).

Pandas DataFrames work the same way, and column names are preserved automatically:

```python
import pandas as pd
from turingbot import simulation

df = pd.read_csv("your_data.csv")

with simulation() as sim:
    sim.start_process(input_file=df)
    # ...
```

## Input format

The input file should be a CSV or TXT file where the last column is the target variable. Example:

```
x1,x2,y
0.1,0.2,0.24
0.3,0.4,0.55
0.5,0.6,0.81
```

## Search metrics

| ID | Metric |
|----|--------|
| 1  | Mean relative error |
| 2  | Classification accuracy |
| 3  | Mean error |
| 4  | RMS error (default) |
| 5  | F-score |
| 6  | Correlation coefficient |
| 7  | Hybrid (CC + RMS) |
| 8  | Maximum error |
| 9  | Maximum relative error |
| 10 | Nash-Sutcliffe efficiency |
| 11 | Binary cross-entropy |
| 12 | Matthews correlation coefficient |
| 13 | Residual sum of squares |
| 14 | Root mean squared log error |
| 15 | Percentile error |

## Parameters

See the full list of parameters in the `start_process` [docstring](https://turingbotsoftware.com/documentation.html), including:

- `threads` -- Number of threads
- `train_test_split` -- Train/test split ratio
- `maximum_formula_complexity` -- Max complexity of formulas
- `integer_constants` -- Restrict to integer constants
- `normalize_dataset` -- Normalize input data
- `allowed_functions` -- Restrict which math functions to use
- `custom_formula` -- Provide a formula template

## Documentation

Full documentation: [turingbotsoftware.com/documentation.html](https://turingbotsoftware.com/documentation.html)

## License

MIT

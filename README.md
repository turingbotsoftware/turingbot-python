# TuringBot Python Library

Python interface to [TuringBot](https://turingbotsoftware.com), a symbolic regression engine: it searches for explicit formulas that fit your data, and returns the most accurate formula found for each size.

The engine is compiled C++ and uses simulated annealing. There is no language runtime to set up and no GUI involved, so searches run the same in a notebook, over SSH or on a headless server. Input can be a CSV file, a numpy array or a pandas DataFrame.

Coming from PySR? [`TuringBotRegressor`](#coming-from-pysr) uses the same API as `PySRRegressor`.

## Installation

```bash
pip install turingbot
```

The engine comes with the TuringBot application: [download it](https://turingbotsoftware.com/download.html) and install it. The library needs Python 3.7 or newer and has no dependencies.

## Quickstart

```python
import time
import numpy as np
from turingbot import simulation

x = np.linspace(-3, 3, 100)
data = np.column_stack([x, 2.5 * np.sin(x) + x ** 2])  # the last column is the target

with simulation() as sim:
    sim.start_process(input_file=data, column_names=["x", "y"])
    time.sleep(10)
    sim.refresh_functions()
    for f in sim.functions:
        print(f)
```

```
[1, 3.27574, '3.0606060606060477']
[3, 1.79969, 'x*x']
[5, 1.03435, 'x*(0.8418349138360374+x)']
[8, 0.93293, 'asinh(x)+x*x']
[10, 0.0, '2.5*sin(x)+x*x']
```

Each line is `[size, error, formula]`, the Pareto front of size versus error, from the simplest formula to the most accurate. Here, the last one is the formula that generated the data.

The search runs in a background process. `refresh_functions()` reads the current front at any time, and the search stops at `terminate_process()` or when the `with` block exits.

`input_file` also takes the path of a CSV or TXT file, or a pandas DataFrame; the last column is the target. TuringBot is found in its default install location; otherwise, pass its path with `path=`. On a Linux server, that can be the AppImage, which needs no installation.

## Features

- **Custom search metrics**, which can reference dataset columns, for example a weight column `w`: `custom_metric="sum(w*pow(actual-predicted,2))/sum(w)"`.
- **Custom functions and constants**, used as building blocks by the search: `custom_functions="bump(x)=exp(-x*x)"`, `custom_constants="g=9.81,c=299792458"`.
- **Time series**: `delay`, `moving_average` and `decay` give formulas access to past values of the inputs, and with `allow_target_delay=True`, of the target itself.
- **GUI and command line**: the same engine runs in the TuringBot desktop app and from the command line, with the same options.

## Search options

```python
sim.start_process(
    input_file="data.csv",
    search_metric=4,                       # RMS error (the default)
    allowed_functions="+ * / sin cos exp",
    train_test_split=80,                   # 80% of the rows for training, 20% for testing
    threads=4,
)
```

With `train_test_split`, each formula also has its error on the test rows: `[size, error, test_error, formula]`.

Every option, with the IDs of the 16 search metrics: `help(simulation.start_process)`, or the [documentation](https://turingbotsoftware.com/documentation.html#python).

## Coming from PySR

`TuringBotRegressor` follows the scikit-learn API and uses `PySRRegressor`'s names where the two match: `model_selection` (`"best"`, `"accuracy"` or `"score"`, with the same definitions), `equations_`, `sympy()`, `max_evals` and `early_stop_condition`. It works in pipelines, cross-validation and grid searches.

```python
from turingbot import TuringBotRegressor

model = TuringBotRegressor(max_time=30).fit(X, y)
print(model.equations_)   # the formulas found, from the simplest to the most accurate
print(model.sympy())      # the chosen formula as a sympy expression
y_pred = model.predict(X_new)
```

| PySRRegressor | TuringBotRegressor |
|---|---|
| `niterations`, `timeout_in_seconds` | `max_time` (seconds) |
| `binary_operators`, `unary_operators` | `allowed_functions` |
| `maxsize` | `maximum_formula_complexity` |

It needs numpy. Its options are in `help(TuringBotRegressor)`.

## Citing

If you use TuringBot in your research, please cite it: [CITATION.cff](CITATION.cff) has the details, which GitHub also shows under "Cite this repository".

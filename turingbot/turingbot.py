"""
Python interface to TuringBot.

The simulation class runs the TuringBot executable and reads the formulas it
finds. TuringBotRegressor builds on it to provide the scikit-learn estimator
API: fit(), predict() and sympy() on numpy arrays or pandas DataFrames.
"""

import os
import platform
import shutil
import subprocess
import tempfile

if os.name != "nt":
    import signal

# Used by TuringBotRegressor only
import ast
import functools
import inspect
import keyword
import math
import numbers
import re
import sys
import time
import warnings

# numpy is needed by TuringBotRegressor only; the simulation class works without it.
try:
    import numpy as np
except ImportError:
    np = None

# scikit-learn is optional. Without it, TuringBotRegressor falls back to the
# minimal base classes defined further below.
try:
    from sklearn.base import BaseEstimator as _BaseEstimator
    from sklearn.base import RegressorMixin as _RegressorMixin
    from sklearn.exceptions import NotFittedError

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Running the TuringBot executable
# ---------------------------------------------------------------------------


_DEFAULT_PATHS = {
    "Windows": [
        r"C:\Program Files (x86)\TuringBot\resources\turingbot.exe",
        r"C:\Program Files\TuringBot\resources\turingbot.exe",
        r"C:\Program Files (x86)\TuringBot\TuringBot.exe",
        r"C:\Program Files\TuringBot\TuringBot.exe",
    ],
    "Darwin": [
        "/Applications/TuringBot.app/Contents/Resources/turingbot",
        "/Applications/TuringBot.app/Contents/MacOS/TuringBot",
    ],
    "Linux": [
        "/usr/lib/turingbot/resources/turingbot",
        "/usr/lib/TuringBot/resources/turingbot",
        "/usr/lib/turingbot/TuringBot",
        "/usr/local/lib/turingbot/TuringBot",
    ],
}


def find_executable():
    """Auto-detect the TuringBot executable path.

    Checks the default installation paths for the current OS, then
    falls back to looking for 'turingbot' or 'TuringBot' on PATH.

    Returns the path as a string, or None if not found.
    """
    system = platform.system()
    for p in _DEFAULT_PATHS.get(system, []):
        if os.path.isfile(p):
            return p

    # Fall back to PATH lookup
    for name in ("turingbot", "TuringBot"):
        found = shutil.which(name)
        if found:
            return found

    return None


def _save_array_to_csv(data, column_names=None):
    """Save a numpy array or pandas DataFrame to a temporary CSV file.

    Returns the path to the temporary file.
    """
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="turingbot_input_")
    try:
        # pandas DataFrame
        if hasattr(data, "to_csv"):
            os.close(fd)
            fd = None
            data.to_csv(path, index=False)
            return path

        # numpy array
        import numpy as np
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(
                f"Expected a 2D array, got {data.ndim}D. "
                "Reshape your data so that rows are samples and columns are variables."
            )

        os.close(fd)
        fd = None
        header = ""
        if column_names is not None:
            if len(column_names) != data.shape[1]:
                raise ValueError(
                    f"column_names has {len(column_names)} names but "
                    f"data has {data.shape[1]} columns."
                )
            header = ",".join(str(c) for c in column_names)

        np.savetxt(path, data, delimiter=",", header=header, comments="")
        return path
    except Exception:
        if fd is not None:
            os.close(fd)
        if os.path.exists(path):
            os.remove(path)
        raise


def _log_tail(log_file):
    try:
        with open(log_file, "r", errors="replace") as f:
            return f.read().strip()[-2000:]
    except OSError:
        return ""


def _remove_file(path):
    """Return whether path is gone."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        return False
    return True


class simulation:
    def __init__(self):
        self.functions = []
        self.process = None
        self._outfile = None
        self._outfile_is_temp = False
        self._infile = None
        self._infile_is_temp = False
        self._logfh = None
        self._log_file = None
        self._terminated = False
        self.info = None

    def parse_lines(self, lines):
        functions = []
        self.info = lines[0].strip()

        # Determine column layout from header (lines[1] is the actual header;
        # lines[0] is the "Formulas generated:" info line)
        header_fields = lines[1].strip().split()
        has_test_error = "Test_Error" in header_fields
        func_idx = 3 if has_test_error else 2

        for i in range(2, len(lines)):
            parts = lines[i].strip().split()
            if not parts or len(parts) <= func_idx:
                continue

            entry = [int(parts[0]), float(parts[1])]
            if has_test_error:
                entry.append(float(parts[2]))
            entry.append(parts[func_idx])

            functions.append(entry)

        return functions

    def start_process(
        self,
        path: str = None,
        input_file=None,
        config: str = None,
        column_names: list = None,
        threads: int = None,
        outfile: str = None,
        predictions_file: str = None,
        formulas_file: str = None,
        search_metric: int = None,
        train_test_split: int = None,
        test_sample: int = None,
        train_test_seed: int = None,
        bound_search_mode: int = None,
        maximum_formula_complexity: int = None,
        history_size: int = None,
        max_occurrences_per_variable: int = None,
        distinct_variables_min: int = None,
        distinct_variables_max: int = None,
        constants_min: int = None,
        constants_max: int = None,
        fscore_beta: float = None,
        percentile: float = None,
        integer_constants: bool = False,
        normalize_dataset: bool = False,
        allow_target_delay: bool = False,
        force_all_variables: bool = False,
        custom_formula: str = None,
        custom_constants: str = None,
        custom_functions: str = None,
        custom_metric: str = None,
        allowed_functions=None,
        function_size: str = None,
        log_file: str = None,
    ):
        """
        Start the process with the specified configuration and dataset.

        Parameters:
        -----------
        path : str, optional
            Path to the TuringBot executable. If not provided, the default
            installation path for your OS will be used automatically.
        input_file : str, os.PathLike, numpy.ndarray, or pandas.DataFrame
            The input data. Can be a file path, a 2D numpy array,
            or a pandas DataFrame. When an array or DataFrame is passed,
            it is saved to a temporary CSV file automatically.
        config : str, optional
            Path to a settings file. The options below override its
            settings; the True/False ones can only switch a setting on.
        column_names : list of str, optional
            Column names for the data. Only used when input_file is a
            numpy array. Ignored for DataFrames (which have their own
            column names) and file paths.
        threads : int, optional
            Number of threads to use.
        outfile : str, optional
            Output file path.
        predictions_file : str, optional
            File to store predictions.
        formulas_file : str, optional
            File with seed formulas to start the search from, such as
            the outfile of an earlier search.
        search_metric : int, optional
            Search metric to use. Default is 4 (RMS error).
            Options:
            1: Mean relative error
            2: Classification accuracy
            3: Mean error
            4: RMS error (default)
            5: F-score
            6: Correlation coefficient
            7: Hybrid (CC + RMS)
            8: Maximum error
            9: Maximum relative error
            10: Nash-Sutcliffe efficiency
            11: Binary cross-entropy
            12: Matthews correlation coefficient (MCC)
            13: Residual sum of squares (RSS)
            14: Root mean squared log error (RMSLE)
            15: Percentile error
            16: Custom metric (requires custom_metric parameter)
        train_test_split : int, optional
            Train/test split. Default is -1 (no test sample).
            Options:
            -1: No test sample (default)
            1-99: Percentage of data used for training (e.g., 80 for 80/20)
            Negative values: Absolute number of training rows (e.g., -200 for 200 rows)
        test_sample : int, optional
            How to select test samples. Default is 1 (random).
            Options:
            1: Chosen randomly (default)
            2: The last points
        train_test_seed : int, optional
            Random seed for train/test split generation. Default is -1 (no specific seed).
        bound_search_mode : int, optional
            Whether to use bound search mode. Default is 0 (deactivated).
            Options:
            0: Deactivated (default)
            1: Upper bound search
            2: Lower bound search
        maximum_formula_complexity : int, optional
            Maximum formula complexity. Default is 60.
        history_size : int, optional
            History size for the optimization process. Default is 20.
        max_occurrences_per_variable : int, optional
            Maximum occurrences per variable. Default is -1 (no limit).
        distinct_variables_min : int, optional
            Minimum number of distinct variables. Default is -1 (no limit).
        distinct_variables_max : int, optional
            Maximum number of distinct variables. Default is -1 (no limit).
        constants_min : int, optional
            Minimum number of constants. Default is -1 (no limit).
        constants_max : int, optional
            Maximum number of constants. Default is -1 (no limit).
        fscore_beta : float, optional
            Beta parameter for F-score. Default is 1.
        percentile : float, optional
            Percentile for the Percentile error metric. Default is 0.5.
        integer_constants : bool, optional
            Whether to use integer constants only. Default is False (disabled).
        normalize_dataset : bool, optional
            Whether to normalize the dataset before optimization. Default is False (no normalization).
        allow_target_delay : bool, optional
            Whether to allow the target variable in lag functions. Default is False (not allowed).
        force_all_variables : bool, optional
            Whether to force the solution to include all input variables. Default is False (not forced).
        custom_formula : str, optional
            Custom formula for the search. If not provided, the last column will be treated as the target variable.
        custom_constants : str, optional
            Custom constants, comma-separated name=value pairs. Add |cost
            after the value to set complexity (default: 1). Examples:
              "k=1.380649e-23,h=6.62607e-34"
              "sun=1.989e30|7" (complexity 7)
        custom_functions : str, optional
            Custom base functions, comma-separated. One argument = unary,
            two = binary. Add |cost after the expression to set complexity
            (default: 4). Examples:
              "bump(x)=exp(-x*x)"
              "mydist(x,y)=sqrt(x*x+y*y)"
              "gauss(x)=exp(-x*x)|2" (complexity 2)
        custom_metric : str, optional
            Custom search metric formula. Automatically sets search_metric
            to 16.
            'actual' and 'predicted' give per-row values and must be
            inside sum(), mean(), median(), maxval(), or minval():
              "mean(pow(actual-predicted,2))"
              "sqrt(mean(pow(actual-predicted,2)))"
              "maxval(abs(actual-predicted))"
            Built-in metrics (rms, mae, mre, maxerr, maxre, accuracy,
            correlation, nash, logloss, mcc_val, rss, rmsle, n) can
            be used directly:
              "0.5*rms+0.5*mae"
        allowed_functions : str or list of str, optional
            Allowed functions for the formula search, as a list of strings
            (e.g. ["+", "sin", "cos"]) or a single string (e.g. "+ sin cos").
            Default: all functions except the logical, history and constant
            ones (the default listed by "turingbot --help").
        function_size : str, optional
            Override size of base functions. Format: space-separated
            name:size pairs. Only overrides need to be specified; unspecified
            functions keep their defaults. Example: "sin:3 cos:3 pow:5 /:3"
        log_file : str, optional
            File that receives the program's console output. By default
            the output is discarded.
        """

        self.functions = []
        self.info = None

        # Terminate any previous process and clean up its temp files
        # before overwriting state with new paths
        self.terminate_process()

        if input_file is None:
            raise ValueError("input_file is required.")

        input_is_path = isinstance(input_file, (str, os.PathLike))
        if input_is_path:
            input_file = os.fspath(input_file)
            if not os.path.isfile(input_file):
                raise FileNotFoundError(f"Input file does not exist: {input_file}")

        if path is None:
            path = find_executable()
            if path is None:
                message = (
                    "TuringBot executable not found. Install TuringBot from "
                    "https://turingbotsoftware.com/download.html or pass the "
                    "path= argument explicitly."
                )
                if platform.system() == "Linux":
                    message += " For the AppImage, pass the .AppImage file as path=."
                raise FileNotFoundError(message)
        path = os.fspath(path)

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Executable path is incorrect: {path}")

        if config is not None and not os.path.isfile(config):
            raise FileNotFoundError(f"Configuration file does not exist: {config}")

        if not input_is_path:
            input_file = _save_array_to_csv(input_file, column_names)
        self._infile = input_file
        self._infile_is_temp = not input_is_path

        if outfile is None:
            fd, self._outfile = tempfile.mkstemp(suffix=".txt", prefix="turingbot_")
            os.close(fd)
            self._outfile_is_temp = True
        else:
            self._outfile = os.fspath(outfile)
            self._outfile_is_temp = False

        cmd = [path, input_file, "--outfile", self._outfile]
        if config is not None:
            cmd.append(os.fspath(config))

        if isinstance(allowed_functions, (list, tuple)):
            allowed_functions = " ".join(allowed_functions)
        options = [
            ("--search-metric", search_metric),
            ("--train-test-split", train_test_split),
            ("--test-sample", test_sample),
            ("--train-test-seed", train_test_seed),
            ("--bound-search-mode", bound_search_mode),
            ("--maximum-formula-complexity", maximum_formula_complexity),
            ("--history-size", history_size),
            ("--max-occurrences-per-variable", max_occurrences_per_variable),
            ("--distinct-variables-min", distinct_variables_min),
            ("--distinct-variables-max", distinct_variables_max),
            ("--constants-min", constants_min),
            ("--constants-max", constants_max),
            ("--fscore-beta", fscore_beta),
            ("--percentile", percentile),
            ("--custom-formula", custom_formula),
            ("--custom-constants", custom_constants),
            ("--custom-functions", custom_functions),
            ("--custom-metric", custom_metric),
            ("--allowed-functions", allowed_functions),
            ("--function-size", function_size),
            ("--threads", None if threads is None else int(threads)),
            ("--predictions-file", None if predictions_file is None else os.fspath(predictions_file)),
            ("--formulas-file", None if formulas_file is None else os.fspath(formulas_file)),
        ]
        for flag, value in options:
            if value is not None:
                cmd += [flag, str(value)]
        for flag, enabled in [
            ("--integer-constants", integer_constants),
            ("--normalize-dataset", normalize_dataset),
            ("--allow-target-delay", allow_target_delay),
            ("--force-all-variables", force_all_variables),
        ]:
            if enabled:
                cmd.append(flag)
        cmd += ["--library-parent-id", str(os.getpid())]

        self._terminated = False
        self._log_file = None if log_file is None else os.fspath(log_file)
        output = subprocess.DEVNULL
        try:
            if log_file is not None:
                # Appended, so _bound_log can empty it while the executable writes it (not on Windows)
                self._logfh = open(log_file, "w" if os.name == "nt" else "a")
                self._logfh.truncate(0)
                output = self._logfh
            if os.name == "nt":
                # Without a console of its own (pythonw, some IDEs), Python would get one opened for it
                self.process = subprocess.Popen(
                    cmd, stdout=output, stderr=output, creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                # Its own process group, so that terminate_process also stops what it starts
                self.process = subprocess.Popen(cmd, stdout=output, stderr=output, start_new_session=True)
        except BaseException:
            self.terminate_process()
            raise

    def refresh_functions(self):
        """Read the formulas found so far into self.functions.

        Raises RuntimeError if the executable exited with an error before writing any.
        """
        if not self._outfile:
            return
        try:
            with open(self._outfile, "r") as f:
                lines = f.readlines()
        except OSError:  # not written yet, or being replaced
            lines = []
        if len(lines) >= 2:
            self.functions = self.parse_lines(lines)

        if self.functions or self.process is None or self._terminated:
            return
        code = self.process.poll()
        if code is not None and code != 0:
            message = f"TuringBot exited with code {code} before writing any formulas."
            if self._log_file is None:
                message += " Pass log_file= to start_process to see its output."
            else:
                log = _log_tail(self._log_file)
                if log:
                    message += f" Its output was:\n{log}"
            raise RuntimeError(message)

    def terminate_process(self):
        """Stop the executable and remove the temporary files."""
        self._terminated = True
        try:
            self._stop_process()
        finally:
            if self._logfh is not None:
                try:
                    self._logfh.close()
                except OSError:
                    pass
                self._logfh = None
            if self._outfile_is_temp and _remove_file(self._outfile):
                self._outfile_is_temp = False
            if self._infile_is_temp and _remove_file(self._infile):
                self._infile_is_temp = False

    def _stop_process(self):
        process = self.process
        if process is not None and process.poll() is None:
            try:
                if os.name == "nt":
                    # /T: also what it started
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                else:
                    os.killpg(process.pid, signal.SIGTERM)
            except OSError:
                try:
                    process.kill()
                except OSError:
                    pass
            # Waited for before its files are removed, which it could otherwise write again
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate_process()

    def __del__(self):
        try:
            self.terminate_process()
        except Exception:  # at interpreter exit, modules can already be gone
            pass


# ---------------------------------------------------------------------------
# scikit-learn interface
# ---------------------------------------------------------------------------
#
# TuringBotRegressor wraps the simulation class in the scikit-learn estimator
# API. fit() runs a search until a stopping criterion is met and keeps the
# formulas found; predict() evaluates the selected formula in Python with the
# same arithmetic as the TuringBot executable, so it works on new data.


if not _HAS_SKLEARN:

    class NotFittedError(ValueError, AttributeError):
        """Raised when predict is called before fit."""

    class _BaseEstimator:
        @classmethod
        def _get_param_names(cls):
            parameters = inspect.signature(cls.__init__).parameters
            return sorted(name for name in parameters if name != "self")

        def get_params(self, deep=True):
            return {name: getattr(self, name) for name in self._get_param_names()}

        def set_params(self, **params):
            valid = self._get_param_names()
            for name, value in params.items():
                if name not in valid:
                    raise ValueError(f"Invalid parameter {name!r} for {type(self).__name__}.")
                setattr(self, name, value)
            return self

        def __repr__(self):
            params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
            return f"{type(self).__name__}({params})"

    class _RegressorMixin:
        def score(self, X, y):
            """Return the coefficient of determination (R^2) of the predictions."""
            y = np.asarray(y, dtype=float).ravel()
            residual = np.sum((y - self.predict(X)) ** 2)
            total = np.sum((y - y.mean()) ** 2)
            return 1.0 - residual / total


_POLL_INTERVAL = 0.1
# The log is only read when no formulas were written, so past this size it can be emptied
_LOG_LIMIT = 1 << 16
_FIRST_OUTPUT_GRACE = 30

_MODEL_SELECTIONS = ("best", "accuracy", "score")

_WORKDIR_PREFIX = "turingbot_fit_"

# scikit-learn's value of y for "validate X alone"
_NO_VALIDATION = "no_validation"

_NONDETERMINISTIC_CHECKS = {
    check: "the search is not deterministic"
    for check in (
        "check_fit_idempotent",
        "check_supervised_y_2d",
        "check_regressors_int",
        "check_regressor_data_not_an_array",
    )
}


class TuringBotRegressor(_RegressorMixin, _BaseEstimator):
    """Symbolic regression with TuringBot, behind the scikit-learn API.

    fit() runs the TuringBot executable on (X, y) until a stopping criterion
    is met and keeps the formulas it found: the most accurate formula for each
    complexity. predict() evaluates one of them in Python, so the model works
    on new data after the search has ended.

    Parameters
    ----------
    max_time : float or None, default=60
        Stop the search after this many seconds.
    max_evals : int or None, default=None
        Stop the search once this many formulas have been generated.
    early_stop_condition : float or None, default=None
        Stop the search once the error of the best formula is at most this.
        At least one of max_time, max_evals and early_stop_condition must be
        set.
    threads : int or None, default=None
        Number of threads. None uses every core.
    model_selection : {"best", "accuracy", "score"}, default="best"
        How predict() chooses a formula:
        "accuracy" takes the lowest error; "score" takes the largest drop in
        log error per unit of complexity; "best" takes the highest score among
        formulas whose error is at most 1.5 times the lowest.
    random_state : int, RandomState instance or None, default=None
        Seed for the train/test split when train_test_split is set.
        Repeated fits can still give different formulas.
    path : str or None, default=None
        Path to the TuringBot executable. None finds it automatically.

    The remaining parameters are passed to the executable; None and False
    keep its defaults. See simulation.start_process for details.

    search_metric, maximum_formula_complexity, allowed_functions,
    function_size, history_size, max_occurrences_per_variable,
    distinct_variables_min, distinct_variables_max, constants_min,
    constants_max, integer_constants, normalize_dataset, force_all_variables,
    bound_search_mode, fscore_beta, percentile, custom_constants,
    custom_functions, custom_metric, train_test_split, test_sample

    With train_test_split, TuringBot holds out part of the training data and
    reports a test error for each formula; the formulas are still selected by
    their training error.

    Attributes
    ----------
    equations_ : pandas.DataFrame or list of dict
        The formulas found, one per complexity, with the columns complexity,
        loss, test_loss (only with train_test_split), score and equation.
        A list of dicts when pandas is not installed.
    best_idx_ : int
        Position in equations_ of the formula chosen by model_selection.
    n_formulas_generated_ : int
        Number of formulas the search generated.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of str
        Feature names seen during fit, when X has string column names.
        Formulas use these names when they are valid identifiers, and
        x0, x1, ... otherwise.
    """

    def __init__(
        self,
        max_time=60,
        max_evals=None,
        early_stop_condition=None,
        threads=None,
        model_selection="best",
        random_state=None,
        path=None,
        search_metric=None,
        maximum_formula_complexity=None,
        allowed_functions=None,
        function_size=None,
        history_size=None,
        max_occurrences_per_variable=None,
        distinct_variables_min=None,
        distinct_variables_max=None,
        constants_min=None,
        constants_max=None,
        integer_constants=False,
        normalize_dataset=False,
        force_all_variables=False,
        bound_search_mode=None,
        fscore_beta=None,
        percentile=None,
        custom_constants=None,
        custom_functions=None,
        custom_metric=None,
        train_test_split=None,
        test_sample=None,
    ):
        self.max_time = max_time
        self.max_evals = max_evals
        self.early_stop_condition = early_stop_condition
        self.threads = threads
        self.model_selection = model_selection
        self.random_state = random_state
        self.path = path
        self.search_metric = search_metric
        self.maximum_formula_complexity = maximum_formula_complexity
        self.allowed_functions = allowed_functions
        self.function_size = function_size
        self.history_size = history_size
        self.max_occurrences_per_variable = max_occurrences_per_variable
        self.distinct_variables_min = distinct_variables_min
        self.distinct_variables_max = distinct_variables_max
        self.constants_min = constants_min
        self.constants_max = constants_max
        self.integer_constants = integer_constants
        self.normalize_dataset = normalize_dataset
        self.force_all_variables = force_all_variables
        self.bound_search_mode = bound_search_mode
        self.fscore_beta = fscore_beta
        self.percentile = percentile
        self.custom_constants = custom_constants
        self.custom_functions = custom_functions
        self.custom_metric = custom_metric
        self.train_test_split = train_test_split
        self.test_sample = test_sample

    def fit(self, X, y):
        """Search for formulas that predict y from X.

        If the search is interrupted, for example by KeyboardInterrupt or by
        a benchmark's SIGALRM handler, the formulas found so far are kept
        before the exception propagates, so predict() still works.

        Without a license, TuringBot loads only part of a large dataset; fit()
        then raises RuntimeError instead of keeping formulas for other data.
        """
        if np is None:
            raise ImportError("TuringBotRegressor requires numpy.")
        if self.max_time is None and self.max_evals is None and self.early_stop_condition is None:
            raise ValueError(
                "Set at least one of max_time, max_evals and early_stop_condition, "
                "otherwise the search never stops."
            )
        if self.model_selection not in _MODEL_SELECTIONS:
            raise ValueError(
                f"model_selection must be one of {_MODEL_SELECTIONS}, got {self.model_selection!r}."
            )
        X, y = self._validate_input(X, y, reset=True)
        context = _FormulaContext(self.custom_constants, self.custom_functions)
        names = self._formula_names(context)
        # A refit that finds nothing must not leave the previous fit's formulas
        for name in ("equations_", "best_idx_", "n_formulas_generated_", "_formula_names_", "_context_"):
            self.__dict__.pop(name, None)

        workdir = tempfile.mkdtemp(prefix=_WORKDIR_PREFIX)
        try:
            output, interruption = self._search(X, y, workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        if output is not None:
            self._store_results(output, names, context)
        if interruption is not None:
            raise interruption
        return self

    def predict(self, X, index=None):
        """Predict y with the selected formula, or with equations_ row `index`."""
        formula = self._formula(index)
        X = self._validate_input(X, reset=False)
        variables = {name: X[:, i] for i, name in enumerate(self._formula_names_)}
        with np.errstate(all="ignore"):
            values = _evaluate(_parse_expression(formula), variables, self._context_)
        return np.array(np.broadcast_to(values, (X.shape[0],)), dtype=float)

    def sympy(self, index=None):
        """Return the selected formula, or equations_ row `index`, as a sympy expression."""
        import sympy

        formula = self._formula(index)
        variables = {name: sympy.Symbol(name) for name in self._formula_names_}
        return _to_sympy(_parse_expression(formula), variables, self._context_, sympy)

    def _more_tags(self):  # scikit-learn < 1.6
        return {"non_deterministic": True, "_xfail_checks": _NONDETERMINISTIC_CHECKS}

    def __sklearn_tags__(self):  # scikit-learn >= 1.6
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags

    def _validate_input(self, X, y=_NO_VALIDATION, reset=True):
        """scikit-learn's validate_data, also without scikit-learn: y="no_validation" checks X alone."""
        if _HAS_SKLEARN:
            kwargs = {"reset": reset, "dtype": np.float64}
            if y is not None and not isinstance(y, str):
                kwargs["y_numeric"] = True
            try:
                from sklearn.utils.validation import validate_data  # scikit-learn >= 1.6
            except ImportError:
                return self._validate_data(X, y, **kwargs)
            return validate_data(self, X, y, **kwargs)

        if y is None:
            raise ValueError(
                f"{type(self).__name__} requires y to be passed, but the target y is None."
            )
        columns = getattr(X, "columns", None)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D array, got {X.ndim}D.")
        if not np.all(np.isfinite(X)):
            raise ValueError("Input X contains NaN or infinity.")
        if reset:
            self.n_features_in_ = X.shape[1]
            if columns is not None and all(isinstance(c, str) for c in columns):
                self.feature_names_in_ = np.asarray(columns, dtype=object)
            elif hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )
        if isinstance(y, str) and y == _NO_VALIDATION:
            return X
        y = np.asarray(y, dtype=float)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1D with one value per row of X.")
        if not np.all(np.isfinite(y)):
            raise ValueError("Input y contains NaN or infinity.")
        return X, y

    def _formula_names(self, context):
        default = [f"x{i}" for i in range(self.n_features_in_)]
        feature_names = getattr(self, "feature_names_in_", None)
        if feature_names is None:
            return default
        names = [str(name) for name in feature_names]
        reserved = context.reserved_names()
        usable = len(set(names)) == len(names) and all(
            name.isidentifier() and not keyword.iskeyword(name) and name not in reserved
            for name in names
        )
        if not usable:
            warnings.warn(
                "Some feature names are not valid formula variables (not identifiers, "
                "duplicated, or the name of a TuringBot function or constant); "
                "formulas use x0, x1, ... instead.",
                UserWarning,
            )
            return default
        return names

    def _train_test_seed(self):
        if self.train_test_split is None or self.random_state is None:
            return None
        if isinstance(self.random_state, numbers.Integral):
            return int(self.random_state)
        return int(self.random_state.randint(0, 2**31 - 1))

    def _search(self, X, y, workdir):
        """Return (last output read or None, exception that interrupted the search or None)."""
        outfile = os.path.join(workdir, "formulas.txt")
        log_file = os.path.join(workdir, "log.txt")

        # The last column is the target; _store_results renames x0, x1, ... to the feature names
        sim = simulation()
        sim.start_process(
            path=self.path,
            input_file=np.column_stack([X, y]),
            column_names=[f"x{i}" for i in range(X.shape[1])] + ["y"],
            outfile=outfile,
            log_file=log_file,
            threads=self.threads,
            search_metric=self.search_metric,
            train_test_split=self.train_test_split,
            test_sample=self.test_sample,
            train_test_seed=self._train_test_seed(),
            bound_search_mode=self.bound_search_mode,
            maximum_formula_complexity=self.maximum_formula_complexity,
            history_size=self.history_size,
            max_occurrences_per_variable=self.max_occurrences_per_variable,
            distinct_variables_min=self.distinct_variables_min,
            distinct_variables_max=self.distinct_variables_max,
            constants_min=self.constants_min,
            constants_max=self.constants_max,
            fscore_beta=self.fscore_beta,
            percentile=self.percentile,
            integer_constants=self.integer_constants,
            normalize_dataset=self.normalize_dataset,
            force_all_variables=self.force_all_variables,
            custom_constants=self.custom_constants,
            custom_functions=self.custom_functions,
            custom_metric=self.custom_metric,
            allowed_functions=self.allowed_functions,
            function_size=self.function_size,
        )

        output = None
        interruption = None
        demo_warning = None
        log_checked = False
        start = time.monotonic()
        try:
            while sim.process.poll() is None:
                time.sleep(_POLL_INTERVAL)
                output = _read_output(outfile, sim) or output
                if output is not None:
                    if not log_checked:
                        log_checked = True
                        demo_warning = _demo_warning(log_file)
                        if demo_warning is not None:
                            break
                    _bound_log(log_file)
                if self._should_stop(output, time.monotonic() - start):
                    break
        except BaseException as error:  # keep the formulas found so far, then re-raise
            interruption = error
        finally:
            exit_code = sim.process.poll()
            sim.terminate_process()
            try:
                sim.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sim.process.kill()
                sim.process.wait()

        if not log_checked:
            demo_warning = _demo_warning(log_file)
        if demo_warning is not None:
            # The formulas fit only part of the data, possibly with another column as the target
            if interruption is not None:
                return None, interruption
            raise RuntimeError(
                "TuringBot is not activated, and the dataset is larger than demo mode allows. "
                f"Its output was:\n{demo_warning}"
            )

        output = _read_output(outfile, sim) or output
        if output is None and interruption is None:
            log = _log_tail(log_file)
            if exit_code is not None:
                reason = f"TuringBot exited with code {exit_code} before writing any formulas."
            else:
                reason = "TuringBot did not write any formulas in time."
            raise RuntimeError(reason + (f" Its output was:\n{log}" if log else ""))
        return output, interruption

    def _should_stop(self, output, elapsed):
        if output is None:
            # Nothing to keep yet: wait for the first formulas, within a grace period
            return self.max_time is not None and elapsed >= self.max_time + _FIRST_OUTPUT_GRACE
        generated, rows = output
        if self.max_time is not None and elapsed >= self.max_time:
            return True
        if self.max_evals is not None and generated >= self.max_evals:
            return True
        best_error = min(row[1] for row in rows)
        return self.early_stop_condition is not None and best_error <= self.early_stop_condition

    def _store_results(self, output, names, context):
        generated, rows = output

        def rename(match):
            return names[int(match.group(1))]

        complexities = [row[0] for row in rows]
        losses = [row[1] for row in rows]
        scores = _formula_scores(complexities, losses)
        has_test = rows[0][2] is not None
        equations = []
        for row, score in zip(rows, scores):
            entry = {"complexity": row[0], "loss": row[1]}
            if has_test:
                entry["test_loss"] = row[2]
            entry["score"] = score
            entry["equation"] = re.sub(r"\bx(\d+)\b", rename, row[3])
            equations.append(entry)
        try:
            import pandas

            self.equations_ = pandas.DataFrame(equations)
        except ImportError:
            self.equations_ = equations
        self.best_idx_ = _select_formula(losses, scores, self.model_selection)
        self.n_formulas_generated_ = generated
        self._formula_names_ = names
        self._context_ = context

    def _formula(self, index):
        if not hasattr(self, "equations_"):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )
        if index is None:
            index = self.best_idx_
        if isinstance(self.equations_, list):
            return self.equations_[index]["equation"]
        return self.equations_["equation"].iloc[index]


# --- Search results --------------------------------------------------------


def _bound_log(log_file):
    """Only safe where the log is opened for appending, which is not the case on Windows."""
    if os.name == "nt":
        return
    try:
        if os.path.getsize(log_file) > _LOG_LIMIT:
            os.truncate(log_file, 0)
    except OSError:
        pass


def _demo_warning(log_file):
    """Return the executable's warning that demo mode loaded only part of the dataset, or None.

    It is the first output, so it is still in the log when the first formulas are written.
    """
    try:
        with open(log_file, "r", errors="replace") as f:
            head = f.read(4096)
    except OSError:
        return None
    for line in head.splitlines():
        if "Demo mode is active" in line:
            return line.strip()
    return None


def _read_output(path, sim):
    """Return (formulas_generated, [[size, error, test_error or None, formula], ...]) or None.

    The file can briefly be missing while it is replaced.
    """
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except OSError:
        return None
    if len(lines) < 3:
        return None
    rows = []
    for entry in sim.parse_lines(lines):
        if len(entry) == 3:
            size, error, formula = entry
            test_error = None
        else:
            size, error, test_error, formula = entry
        rows.append([size, error, test_error, formula])
    if not rows:
        return None
    match = re.search(r"(\d+)", lines[0])
    return (int(match.group(1)) if match else 0), rows


def _formula_scores(complexities, losses):
    """Score each formula by the drop in log loss per unit of complexity from the previous one."""
    scores = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, (complexity, loss) in enumerate(zip(complexities, losses)):
            if i == 0:
                scores.append(0.0)
            elif loss > 0:
                ratio = np.float64(loss) / losses[i - 1]
                scores.append(float(-np.log(ratio) / (complexity - complexities[i - 1])))
            else:
                scores.append(float("inf"))
    return scores


def _select_formula(losses, scores, model_selection):
    if model_selection == "accuracy":
        return losses.index(min(losses))
    if model_selection == "best":
        threshold = 1.5 * min(losses)
        candidates = [i for i, loss in enumerate(losses) if loss <= threshold]
    elif model_selection == "score":
        candidates = range(len(losses))
    else:
        raise ValueError(f"model_selection must be one of {_MODEL_SELECTIONS}, got {model_selection!r}.")

    def key(i):
        return scores[i] if not math.isnan(scores[i]) else -math.inf

    return max(candidates, key=key)

# --- Formula language ------------------------------------------------------
#
# Formulas as the executable writes them are valid Python expressions, so
# they are parsed with the ast module. These names and definitions are what a
# formula can refer to besides its variables.


_UNARY_FUNCTIONS = (
    "sin", "cos", "tan", "asin", "acos", "atan", "log", "log2", "log10", "exp",
    "asinh", "acosh", "atanh", "sinh", "cosh", "tanh", "tgamma", "lgamma",
    "floor", "ceil", "round", "erf", "sqrt", "abs", "sign",
)
_BINARY_FUNCTIONS = (
    "pow", "fmod", "smaller", "greater", "equal", "different", "logical_or",
    "logical_and", "max", "min",
)
_HISTORY_FUNCTIONS = ("delay", "moving_average", "decay")

_NAMED_CONSTANTS = {
    "pi": 3.14159265358979323846,
    "e": 2.71828182845904523536,
    "sqrt2": 1.41421356237309504880,
    "ln2": 0.69314718055994530941,
    "gamma": 0.57721566490153286060,
    "phi": 1.61803398874989484820,
}


# Python 3.7 parses numbers as ast.Num; later versions use ast.Constant.
_AST_NUM = ast.Num if sys.version_info < (3, 8) else None


def _parse_expression(text):
    """Formulas are Python expressions; ^ in custom function bodies has the precedence of **."""
    try:
        return ast.parse(text.strip().replace("^", "**"), mode="eval").body
    except SyntaxError:
        raise ValueError(f"Cannot parse formula {text!r}.") from None


def _split_top_level(text, separator=",", maxsplit=-1):
    parts = []
    depth = 0
    current = ""
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == separator and depth == 0 and len(parts) != maxsplit:
            parts.append(current)
            current = ""
        else:
            current += char
    if current:
        parts.append(current)
    return parts


def _strip_cost(text):
    head, separator, tail = text.rpartition("|")
    if separator and tail.strip().isdigit():
        return head.strip()
    return text.strip()


def _parse_custom_constants(text):
    constants = {}
    for entry in (text or "").split(","):
        name, separator, value = entry.partition("=")
        name = name.strip()
        if not separator or not name:
            continue
        try:
            constants[name] = float(_strip_cost(value))
        except ValueError:
            raise ValueError(f"Cannot parse custom constant {entry.strip()!r}.") from None
    return constants


def _parse_custom_functions(text):
    """Parse "name(x)=body[|cost],..." into a {name: (parameters, body)} dict."""
    functions = {}
    for entry in _split_top_level(text or ""):
        if not entry.strip():
            continue
        parts = _split_top_level(entry, "=", maxsplit=1)
        match = re.fullmatch(
            r"\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*(?:,\s*([A-Za-z_]\w*)\s*)?\)\s*",
            parts[0],
        )
        if len(parts) != 2 or match is None:
            raise ValueError(f"Cannot parse custom function {entry.strip()!r}.")
        parameters = [p for p in match.groups()[1:] if p is not None]
        functions[match.group(1)] = (parameters, _parse_expression(_strip_cost(parts[1])))
    return functions


class _FormulaContext:
    def __init__(self, custom_constants=None, custom_functions=None):
        self.constants = dict(_NAMED_CONSTANTS)
        self.constants.update(_parse_custom_constants(custom_constants))
        self.custom_functions = _parse_custom_functions(custom_functions)

    def reserved_names(self):
        return (
            set(_UNARY_FUNCTIONS) | set(_BINARY_FUNCTIONS) | set(_HISTORY_FUNCTIONS)
            | set(self.constants) | set(self.custom_functions)
        )


def _check_call(node, context):
    if not isinstance(node.func, ast.Name) or node.keywords:
        raise ValueError(f"Unsupported call in formula: {ast.dump(node)}")
    name = node.func.id
    if name in context.custom_functions:
        expected = len(context.custom_functions[name][0])
    elif name in _UNARY_FUNCTIONS:
        expected = 1
    elif name in _BINARY_FUNCTIONS or name in _HISTORY_FUNCTIONS:
        expected = 2
    else:
        raise ValueError(f"Unknown function {name!r} in formula.")
    if len(node.args) != expected:
        raise ValueError(f"{name} takes {expected} argument(s), got {len(node.args)}.")
    return name, node.args


def _number(node):
    if isinstance(node, ast.Constant):
        value = node.value
    elif _AST_NUM is not None and isinstance(node, _AST_NUM):
        value = node.n
    else:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


# --- Evaluation with numpy -------------------------------------------------


def _round_half_away(x):
    """C's round(): halves are rounded away from zero, unlike numpy.round."""
    truncated = np.trunc(x)
    return np.where(np.abs(x - truncated) >= 0.5, truncated + np.sign(x), truncated)


def _special_functions():
    try:
        from scipy import special

        return special.gamma, special.gammaln, special.erf
    except ImportError:

        def vectorize(function):
            def safe(value):
                try:
                    return function(value)
                except (ValueError, OverflowError):
                    return float("nan")

            return np.vectorize(safe, otypes=[float])

        return vectorize(math.gamma), vectorize(math.lgamma), vectorize(math.erf)


@functools.lru_cache(maxsize=None)
def _numpy_functions():
    tgamma, lgamma, erf = _special_functions()

    def indicator(test):
        return lambda x, y: np.asarray(test(x, y), dtype=float)

    return {
        ast.Add: np.add,
        ast.Sub: np.subtract,
        ast.Mult: np.multiply,
        ast.Div: np.true_divide,
        ast.Pow: np.power,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
        "log": np.log, "log2": np.log2, "log10": np.log10, "exp": np.exp,
        "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "tgamma": tgamma, "lgamma": lgamma, "erf": erf,
        "floor": np.floor, "ceil": np.ceil, "round": _round_half_away,
        "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
        "pow": np.power, "fmod": np.fmod,
        "smaller": indicator(np.less),
        "greater": indicator(np.greater),
        "equal": indicator(np.equal),
        "different": indicator(np.not_equal),
        "logical_or": indicator(lambda x, y: np.logical_or(x > 0, y > 0)),
        "logical_and": indicator(lambda x, y: np.logical_and(x > 0, y > 0)),
        "max": np.maximum, "min": np.minimum,
    }


def _nan_if_not_finite(result, *arguments):
    """NaN wherever an argument is NaN or infinite, the rule the executable applies to every operation."""
    finite = np.ones(np.shape(result), dtype=bool)
    for argument in arguments:
        finite &= np.isfinite(argument)
    return np.where(finite, result, np.nan)


def _evaluate(node, variables, context):
    """Literals stay scalars, so the result is a scalar when the formula has no variables."""
    number = _number(node)
    if number is not None:
        # float() also keeps integer literals out of numpy's integer power rules
        return float(number)
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in context.constants:
            return context.constants[node.id]
        raise ValueError(f"Unknown name {node.id!r} in formula.")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _evaluate(node.operand, variables, context)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp) and type(node.op) in _numpy_functions():
        left = _evaluate(node.left, variables, context)
        right = _evaluate(node.right, variables, context)
        return _nan_if_not_finite(_numpy_functions()[type(node.op)](left, right), left, right)
    if isinstance(node, ast.Call):
        name, argument_nodes = _check_call(node, context)
        if name in _HISTORY_FUNCTIONS and name not in context.custom_functions:
            raise NotImplementedError(
                f"{name}() reads earlier rows of the dataset, so the formula "
                "cannot be evaluated on independent samples."
            )
        arguments = [_evaluate(a, variables, context) for a in argument_nodes]
        if name in context.custom_functions:
            parameters, body = context.custom_functions[name]
            result = _evaluate(body, dict(zip(parameters, arguments)), context)
        else:
            result = _numpy_functions()[name](*arguments)
        return _nan_if_not_finite(result, *arguments)
    raise ValueError(f"Unsupported expression in formula: {ast.dump(node)}")


# --- Conversion to sympy ---------------------------------------------------


@functools.lru_cache(maxsize=None)
def _sympy_constants(sympy):
    return {
        "pi": sympy.pi, "e": sympy.E, "sqrt2": sympy.sqrt(2), "ln2": sympy.log(2),
        "gamma": sympy.EulerGamma, "phi": sympy.GoldenRatio,
    }


@functools.lru_cache(maxsize=None)
def _sympy_functions(sympy):
    def indicator(condition):
        return sympy.Piecewise((1, condition), (0, True))

    def fmod(x, y):
        # C's fmod truncates the quotient, so the result has the sign of x
        quotient = x / y
        return x - y * sympy.sign(quotient) * sympy.floor(sympy.Abs(quotient))

    return {
        ast.Add: lambda x, y: x + y,
        ast.Sub: lambda x, y: x - y,
        ast.Mult: lambda x, y: x * y,
        ast.Div: lambda x, y: x / y,
        ast.Pow: sympy.Pow,
        "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
        "asin": sympy.asin, "acos": sympy.acos, "atan": sympy.atan,
        "log": sympy.log,
        "log2": lambda x: sympy.log(x, 2),
        "log10": lambda x: sympy.log(x, 10),
        "exp": sympy.exp,
        "asinh": sympy.asinh, "acosh": sympy.acosh, "atanh": sympy.atanh,
        "sinh": sympy.sinh, "cosh": sympy.cosh, "tanh": sympy.tanh,
        "tgamma": sympy.gamma,
        "lgamma": lambda x: sympy.log(sympy.Abs(sympy.gamma(x))),
        "erf": sympy.erf,
        "floor": sympy.floor, "ceil": sympy.ceiling,
        "round": lambda x: sympy.sign(x) * sympy.floor(sympy.Abs(x) + sympy.Rational(1, 2)),
        "sqrt": sympy.sqrt, "abs": sympy.Abs, "sign": sympy.sign,
        "pow": sympy.Pow, "fmod": fmod,
        "smaller": lambda x, y: indicator(x < y),
        "greater": lambda x, y: indicator(x > y),
        "equal": lambda x, y: indicator(sympy.Eq(x, y)),
        "different": lambda x, y: indicator(sympy.Ne(x, y)),
        "logical_or": lambda x, y: indicator(sympy.Or(x > 0, y > 0)),
        "logical_and": lambda x, y: indicator(sympy.And(x > 0, y > 0)),
        "max": sympy.Max, "min": sympy.Min,
    }


def _to_sympy(node, variables, context, sympy):
    """Custom functions are expanded and custom constants become values; history functions stay undefined."""
    number = _number(node)
    if number is not None:
        # From repr, sympy keeps every digit needed to give back the same double
        return sympy.Integer(number) if isinstance(number, int) else sympy.Float(repr(number))
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in _sympy_constants(sympy):
            return _sympy_constants(sympy)[node.id]
        if node.id in context.constants:
            return sympy.Float(repr(context.constants[node.id]))
        raise ValueError(f"Unknown name {node.id!r} in formula.")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _to_sympy(node.operand, variables, context, sympy)
        return -operand if isinstance(node.op, ast.USub) else operand
    functions = _sympy_functions(sympy)
    if isinstance(node, ast.BinOp) and type(node.op) in functions:
        left = _to_sympy(node.left, variables, context, sympy)
        right = _to_sympy(node.right, variables, context, sympy)
        return functions[type(node.op)](left, right)
    if isinstance(node, ast.Call):
        name, argument_nodes = _check_call(node, context)
        arguments = [_to_sympy(a, variables, context, sympy) for a in argument_nodes]
        if name in context.custom_functions:
            parameters, body = context.custom_functions[name]
            return _to_sympy(body, dict(zip(parameters, arguments)), context, sympy)
        if name in _HISTORY_FUNCTIONS:
            return sympy.Function(name)(*arguments)
        return functions[name](*arguments)
    raise ValueError(f"Unsupported expression in formula: {ast.dump(node)}")

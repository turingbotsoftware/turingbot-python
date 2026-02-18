import os
import platform
import random
import shutil
import subprocess
import tempfile

if os.name != "nt":
    import signal

_DEFAULT_PATHS = {
    "Windows": [
        r"C:\Program Files (x86)\TuringBot\TuringBot.exe",
        r"C:\Program Files\TuringBot\TuringBot.exe",
    ],
    "Darwin": [
        "/Applications/TuringBot.app/Contents/MacOS/TuringBot",
    ],
    "Linux": [
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
        if os.path.exists(path):
            os.remove(path)
        raise


class simulation:
    def __init__(self):
        self.functions = []
        self.process = None
        self._outfile = None
        self._outfile_is_temp = False
        self._infile = None
        self._infile_is_temp = False
        self.info = None

    def parse_lines(self, lines):
        functions = []
        self.info = lines[0].strip()

        for i in range(2, len(lines)):
            functions.append(lines[i].strip().split())
            functions[-1][0] = int(functions[-1][0])
            functions[-1][1] = float(functions[-1][1])
            if len(functions[-1]) == 4:
                functions[-1][2] = float(functions[-1][2])

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
        allowed_functions: str = None,
    ):
        """
        Start the process with the specified configuration and dataset.

        Parameters:
        -----------
        path : str, optional
            Path to the TuringBot executable. If not provided, the default
            installation path for your OS will be used automatically.
        input_file : str, numpy.ndarray, or pandas.DataFrame
            The input data. Can be a file path (str), a 2D numpy array,
            or a pandas DataFrame. When an array or DataFrame is passed,
            it is saved to a temporary CSV file automatically.
        config : str, optional
            Path to the configuration file.
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
            File to store generated formulas.
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
        train_test_split : int, optional
            Train/test split. Default is -1 (no test sample).
            Options:
            -1: No test sample (default)
            50, 60, 70, 75, 80: Percentage split for training data
            100, 1000, 10000: Predefined row counts for training
            Negative values (e.g., -200): Use 200 rows for training
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
            1: Lower bound search
            2: Upper bound search
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
        allowed_functions : str, optional
            Allowed functions for the formula search. Default: "+ * / pow fmod sin cos tan asin acos atan exp log log2 log10 sqrt sinh cosh tanh asinh acosh atanh abs floor ceil round sign tgamma lgamma erf"
        """

        self.functions = []

        if input_file is None:
            raise ValueError("input_file is required.")

        # Convert numpy arrays and pandas DataFrames to a temp CSV
        if not isinstance(input_file, str):
            input_file = _save_array_to_csv(input_file, column_names)
            self._infile = input_file
            self._infile_is_temp = True
        else:
            self._infile = input_file
            self._infile_is_temp = False

        if path is None:
            path = find_executable()
            if path is None:
                raise FileNotFoundError(
                    "TuringBot executable not found. Install TuringBot from "
                    "https://turingbotsoftware.com/download.html or pass the "
                    "path= argument explicitly."
                )

        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Input file does not exist: {input_file}")

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Executable path is incorrect: {path}")

        if config is not None and not os.path.isfile(config):
            raise FileNotFoundError(f"Configuration file does not exist: {config}")

        path = f'"{path}"'
        input_file = f'"{input_file}"'
        if outfile is None:
            if os.name == "nt":
                self._outfile = tempfile.gettempdir() + r"\turingbot_%d.txt" % random.randint(1000000000, 9999999999)
            else:
                self._outfile = tempfile.gettempdir() + "/turingbot_%d.txt" % random.randint(1000000000, 9999999999)
            self._outfile_is_temp = True
        else:
            self._outfile = outfile
            self._outfile_is_temp = False

        if self.process is not None:
            self.process.kill()

        if os.name == "nt":
            cmd = f"{path} {input_file} --outfile {self._outfile}"
        elif os.name == "posix":
            cmd = f"exec {path} {input_file} --outfile {self._outfile} 1>/dev/null 2>/dev/null"
        else:
            cmd = f"exec stdbuf -oL {path} {input_file} --outfile {self._outfile} 1>/dev/null 2>/dev/null"

        if config is not None:
            cmd = f"{cmd} {config}"
        else:
            if search_metric is not None:
                cmd = f"{cmd} --search-metric {search_metric}"
            if train_test_split is not None:
                cmd = f"{cmd} --train-test-split {train_test_split}"
            if test_sample is not None:
                cmd = f"{cmd} --test-sample {test_sample}"
            if train_test_seed is not None:
                cmd = f"{cmd} --train-test-seed {train_test_seed}"
            if bound_search_mode is not None:
                cmd = f"{cmd} --bound-search-mode {bound_search_mode}"
            if maximum_formula_complexity is not None:
                cmd = f"{cmd} --maximum-formula-complexity {maximum_formula_complexity}"
            if history_size is not None:
                cmd = f"{cmd} --history-size {history_size}"
            if max_occurrences_per_variable is not None:
                cmd = f"{cmd} --max-occurrences-per-variable {max_occurrences_per_variable}"
            if distinct_variables_min is not None:
                cmd = f"{cmd} --distinct-variables-min {distinct_variables_min}"
            if distinct_variables_max is not None:
                cmd = f"{cmd} --distinct-variables-max {distinct_variables_max}"
            if constants_min is not None:
                cmd = f"{cmd} --constants-min {constants_min}"
            if constants_max is not None:
                cmd = f"{cmd} --constants-max {constants_max}"
            if fscore_beta is not None:
                cmd = f"{cmd} --fscore-beta {fscore_beta}"
            if percentile is not None:
                cmd = f"{cmd} --percentile {percentile}"
            if integer_constants:
                cmd = f"{cmd} --integer-constants"
            if normalize_dataset:
                cmd = f"{cmd} --normalize-dataset"
            if allow_target_delay:
                cmd = f"{cmd} --allow-target-delay"
            if force_all_variables:
                cmd = f"{cmd} --force-all-variables"
            if custom_formula is not None:
                cmd = f"{cmd} --custom-formula \"{custom_formula}\""
            if allowed_functions is not None:
                cmd = f"{cmd} --allowed-functions \"{allowed_functions}\""

        if threads is not None:
            cmd = f"{cmd} --threads {int(threads)}"

        if predictions_file is not None:
            cmd = f"{cmd} --predictions-file {predictions_file}"

        if formulas_file is not None:
            cmd = f"{cmd} --formulas-file {formulas_file}"

        cmd = f"{cmd} --library-parent-id {str(os.getpid())}"

        if os.name == "nt":
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            self.process = subprocess.Popen(
                cmd,
                shell=True,
                encoding="utf-8",
                preexec_fn=os.setsid
            )

    def refresh_functions(self):
        if self._outfile and os.path.isfile(self._outfile):
            with open(self._outfile, "r") as f:
                lines = f.readlines()

            if len(lines) > 0:
                self.functions = self.parse_lines(lines)

    def terminate_process(self):
        try:
            if self.process and self.process.poll() is None:
                if os.name == "nt":
                    self.process.terminate()
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

            if self._outfile_is_temp and self._outfile and os.path.exists(self._outfile):
                os.remove(self._outfile)
                self._outfile_is_temp = False

            if self._infile_is_temp and self._infile and os.path.exists(self._infile):
                os.remove(self._infile)
                self._infile_is_temp = False
        except Exception as error:
            print(error)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate_process()

    def __del__(self):
        self.terminate_process()

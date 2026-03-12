"""
July 2021

Ilwoo Lyu, ilwoolyu@postech.ac.kr
Seungbo Ha, mj0829@unist.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


class Logger(object):
    def __init__(self, path=None):
        self._path = path

    def write(self, values, log_format="12.6", overflow_margin=2, trunc_suffix="...", endline="\n"):
        """
        Format and write values to a log file or return as a string.

        The function supports single values, lists/tuples, and dictionaries. Numeric
        values are formatted according to the given log_format. Strings exceeding
        the specified width are truncated with a suffix. The formatted line can be
        optionally appended to a file.

        Parameters
        ----------
        values : int, float, str, list, tuple, or dict
            The value(s) to format and log.
        log_format : str
            Format string "width.decimal" for numeric values. Width is total field
            width; decimal is number of decimal places for floats.
        overflow_margin : int
            Margin to allow before switching to scientific notation for numeric values.
        trunc_suffix : str
            Suffix to append if a string value is truncated.
        endline : str
            String appended to the end of the formatted line (usually newline).

        Returns
        -------
        line : str
            The formatted line as a string (without the trailing endline character).

        Raises
        ------
        Exception
            If a value type is not supported (i.e., not int, float, str, or dict).
        """

        if not isinstance(values, Iterable):
            values = [values]

        width, decimal = map(int, log_format.split("."))
        width = max(width, decimal + 2, 6)

        line = ""
        if isinstance(values, dict):
            for key, val in values.items():
                line += f"{key} : {val}" + endline
        else:
            for v in values:
                if isinstance(v, (int, float)):
                    val = f"{v:^{log_format}f}" if isinstance(v, float) else f"{v:^{width}d}"
                    if len(val.strip()) <= width - overflow_margin:
                        line += val
                    else:
                        line += f"{v:^{width}.{max(0, min(decimal - 4, width - decimal - 1))}e}"
                elif isinstance(v, str):
                    if len(v) > width - overflow_margin:
                        v = v[: width - overflow_margin - len(trunc_suffix)] + trunc_suffix
                    line += f"{v:^{width}}"
                else:
                    raise Exception("LoggerError: unsupported type!")
            line += endline

        if self._path:
            open(self._path, "a").write(line)

        return line[: -len(endline)]

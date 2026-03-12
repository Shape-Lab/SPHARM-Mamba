"""
August 2025

Ilwoo Lyu, ilwoolyu@postech.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Pohang University of Science and Technology
"""

import os
import ast
import argparse
import configparser

from .. import __version__


def _cast_value(v):
    """
    Automatically cast string to int, float, or bool if possible.

    Parameters
    ----------
    v : str
        The string value to cast.

    Returns
    -------
    v : int, float, bool, list, or str
        The casted value. Returns the original string, otherwise.
    """

    if isinstance(v, str):
        v = v.strip()

        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass

        if "," in v:
            return [_cast_value(x) for x in v.split(",")]

        v_lower = v.lower()
        if v_lower in ["true", "yes"]:
            return True
        elif v_lower in ["false", "no"]:
            return False

    return v


def load_config(parser, fname, section_prefix=""):
    """
    Load defaults from a config file into an argparse parser.
    Supports prefixed section titles. If the config file is missing, parser defaults are preserved.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser containing arguments and groups to set defaults for.
    fname : str
        Path to the configuration file.
    section_prefix : str
        Prefix applied to section titles in the config file.

    Returns
    -------
    parser : argparse.ArgumentParser
        The parser with defaults updated from the config file, or unchanged if file is missing.
    """

    if fname and os.path.exists(fname):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(fname)

        defaults = {}
        for section in config.sections():
            if section_prefix and not section.startswith(section_prefix + "."):
                continue

            for k, v in config[section].items():
                if k in defaults:
                    raise ValueError(f"Duplicate argument '{k}' found in multiple sections of the config file")
                defaults[k] = _cast_value(v)

        parser.set_defaults(**defaults)

    return parser


def save_config(args, parser, fname, exclude_keys="", section_prefix=""):
    """
    Save argparse arguments to a config file, grouped by argument groups.
    Updates existing keys and preserves unrelated sections/values.
    Ungrouped arguments are saved under a default [config] section.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments to save.
    parser : argparse.ArgumentParser
        The parser used to parse args (for retrieving argument groups).
    fname : str
        File path to save the configuration.
    exclude_keys : list or str
        Argument names to exclude from saving.
    section_prefix : str
        Prefix applied to section titles in the config file.

    Returns
    -------
    None
        The function updates the config file on disk.
    """

    if isinstance(exclude_keys, str):
        exclude_keys = [exclude_keys]

    config = configparser.ConfigParser()
    config.optionxform = str
    if os.path.exists(fname):
        config.read(fname)

    dest_to_group = {}
    for group in parser._action_groups:
        for action in group._group_actions:
            dest_to_group[action.dest] = group.title

    if section_prefix:
        section_prefix += "."

    for key, val in vars(args).items():
        if key in exclude_keys or val is None:
            continue

        section = dest_to_group.get(key, None)
        if section is None:
            section = "optional arguments"

        section = f"{section_prefix}{section}"

        if section not in config:
            config[section] = {}

        if isinstance(val, bool):
            val = "yes" if val else "no"

        config[section][key] = str(val)

    with open(fname, "w") as f:
        f.write(
            f"; Auto-generated configuration file (SPHARM-DNNs v{__version__})\n"
            "; Edit the values below if needed.\n"
            "; Note: existing comments will not be preserved when saving with --save-config.\n\n"
        )
        config.write(f)


def parse_args(parser, section_prefix=""):
    """
    Parse command-line arguments with support for a config file.
    Handles required arguments automatically by temporarily disabling
    'required=True' until config defaults are loaded.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser with all arguments defined (including required ones).
    section_prefix : str
        Prefix applied to section titles in the config file.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with defaults applied from the config file.
    """

    required_flags = [a.dest for a in parser._actions if hasattr(a, "required") and a.required]

    for action in parser._actions:
        if hasattr(action, "required") and action.dest in required_flags:
            action.required = False

    parser = load_config(parser, getattr(parser.parse_known_args()[0], "config", None), section_prefix=section_prefix)

    for action in parser._actions:
        if hasattr(action, "required") and action.dest in required_flags and action.default is None:
            action.required = True
            continue

        if getattr(action, "default", None) in (argparse.SUPPRESS, None):
            continue

        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            continue

        if hasattr(action, "type") and action.type is not None:
            try:
                if getattr(action, "nargs", None) in ("+", "*"):
                    if not isinstance(action.default, list):
                        action.default = [action.default]
                    action.default = [action.type(v) for v in action.default]
                else:
                    action.default = action.type(action.default)
            except Exception as e:
                raise ValueError(f"Cannot cast config value for '{action.dest}': {action.default}") from e

    return parser.parse_args()

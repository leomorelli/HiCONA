"""Module to retrieve and change settings regarding package functionality.

The module contains custom objects used to retrieve the settings from a .json
file, so that they can be used in other parts of the code with an import.
Moreover, a public api is provided to visualize and modify those settings.
"""

from dataclasses import dataclass
import importlib.resources as imp_res
from os import path
import json

from . import resources


__all__ = ["display_settings", "update_settings", "restore_settings"]


_LINE_LENGTH = 79  # Max number of characters per printed line
_BACKUP_PATH = "backup.json"  # Backup position for packages settings
_DEFAULT_PATH = "defaults.json"  # Default position for package settings

# TODO: Look into #: to add attribute documentations to dataclasses

##############################################################################
########################### SETTINGS STORE OBJECTS ###########################
##############################################################################


class _SettingsBase:
    """Base class to implement string representation of a dataclass."""

    def __str__(self):
        out = self.__class__.__name__.replace("_Hicona", "").upper()
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                out += f"\n - {k}:"
                for k_in, v_in in v.items():
                    out += f"\n   - {k_in}: {v_in}"
            else:
                out += f"\n - {k}: {v}"
        return out


@dataclass
class _HiconaConventions(_SettingsBase):
    """Class to store arbitrary string/dict like objects used in HiCONA."""

    chrom_lists: dict
    dtype_conversion: dict
    table_columns: dict
    table_uri_template: str


@dataclass
class _HiconaParameters(_SettingsBase):
    """Class to store processing parameter (values you might want to tune)."""

    base_pix_chunk: int
    lock_delay: float
    max_annot_mods: int
    min_pix_chunk: int


@dataclass
class _HiconaRegexes(_SettingsBase):
    """Class to store the various regexes, especially for genomic regions."""

    region: str
    chromosome: str


class HiconaSettings:
    """Class to be instantiated and imported to obtain the settings."""

    def __init__(self, conventions, parameters, regexes):
        self.conventions = _HiconaConventions(**conventions)
        self.parameters = _HiconaParameters(**parameters)
        self.regexes = _HiconaRegexes(**regexes)

    def __str__(self):
        hyphen_row = "-" * _LINE_LENGTH
        main_title = "HICONA SETTINGS".center(_LINE_LENGTH)
        out = f"{hyphen_row}\n{main_title}\n{hyphen_row}"
        for v in self.__dict__.values():
            out += f"\n\n{v}"
        out += f"\n\n{hyphen_row}"

        return out

    def to_json(self) -> dict:
        """Convert object and all its attributes in a nested json."""

        return {k: v.__dict__ for k, v in self.__dict__.items()}

    def update(self, new_settings: dict):
        """Recursively update object attributes with provided ones."""

        for k, v in new_settings.items():
            for a, o in self.__dict__.items():
                if k in o.__dict__:
                    setattr(getattr(self, a), k, v)
                    break


##############################################################################
############################# BASIC SETTINGS OPS #############################
##############################################################################


def _read_settings_json(json_path):
    """Load json settings from a file."""

    folder = imp_res.files(resources)
    with open(path.join(folder, json_path), "r", encoding="utf-8") as handle:
        settings = json.load(handle)
    return settings


def _write_settings(json_path, settings_json):
    """Write json settings to a file."""

    folder = imp_res.files(resources)
    with open(path.join(folder, json_path), "w", encoding="utf-8") as handle:
        json.dump(settings_json, handle, indent=4)


def _load_settings(settings_path, backup_path):
    """Create settings obj from default location, fall back to backup path."""

    try:
        settings = _read_settings_json(settings_path)
    except FileNotFoundError:
        print("W: Unable to load default Hicona settings, using backup.")
        settings = _read_settings_json(backup_path)

    return HiconaSettings(**settings)


##############################################################################
############################ PUBLIC SETTINGS API #############################
##############################################################################


def display_settings():
    """Print settings to console."""

    print(HICONA_SETTINGS)


def update_settings(new_settings: dict) -> None:
    """Update current settings with the provided ones.

    Parameters
    ----------
    new_settings : dict
        Dictionary of the settings to update and write to disk.
    """

    HICONA_SETTINGS.update(new_settings)
    _write_settings(_DEFAULT_PATH, HICONA_SETTINGS.to_json())


def restore_settings():
    """Restore default settings, both on disk and in memory."""

    backup = _read_settings_json(_BACKUP_PATH)
    backup = {k: v for d in backup.values() for k, v in d.items()}
    update_settings(backup)


##############################################################################
########################## SETTINGS INITIALIZATION ###########################
##############################################################################

HICONA_SETTINGS = _load_settings(_DEFAULT_PATH, _BACKUP_PATH)

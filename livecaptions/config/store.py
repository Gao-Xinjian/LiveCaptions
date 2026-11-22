"""Load and save settings (TOML), term profiles and the default folders."""
import copy
import dataclasses
import os
import re
import tomllib

import platformdirs
import tomli_w

from livecaptions import APP_NAME
from livecaptions.config.schema import CONFIG_VERSION, Config


def get_config_dir():
    """Return the folder of config.toml and the term profiles."""
    return platformdirs.user_config_dir(APP_NAME, appauthor=False, roaming=True)


def get_config_file():
    """Return the path of config.toml."""
    return os.path.join(get_config_dir(), 'config.toml')


def get_models_dir(cfg):
    """Return the models folder of cfg, or the default one."""
    return cfg.model.models_dir or os.path.join(platformdirs.user_data_dir(APP_NAME, appauthor=False), 'models')


def get_save_dir(cfg):
    """Return the save folder of cfg, or Documents/LiveCaptions."""
    return cfg.save.dir or os.path.join(platformdirs.user_documents_dir(), APP_NAME)


def get_profiles_dir():
    """Return the folder of the term profiles (one .txt per profile)."""
    return os.path.join(get_config_dir(), 'terms')


def cfg2dict(cfg):
    """Convert a Config to a plain dict for TOML."""
    return dataclasses.asdict(cfg)


def _coerce(value, default):
    """Convert value to the type of default, or return default if it can't be."""
    try:
        if isinstance(default, bool):
            return bool(value)
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return str(value)
        if isinstance(default, list):
            return list(value)
    except (TypeError, ValueError):
        pass
    return default


def dict2cfg(data):
    """Build a Config from a dict, keeping defaults for missing or invalid keys and ignoring unknown ones."""
    data = migrate_dict(copy.deepcopy(data))
    cfg = Config()
    for sec in dataclasses.fields(cfg):
        if sec.name == 'version' or not isinstance(data.get(sec.name), dict):
            continue
        obj = getattr(cfg, sec.name)
        for f in dataclasses.fields(obj):
            if f.name in data[sec.name]:
                setattr(obj, f.name, _coerce(data[sec.name][f.name], getattr(obj, f.name)))
    return cfg


def migrate_dict(data):
    """Upgrade a settings dict from an older version to CONFIG_VERSION."""
    version = int(data.get('version', CONFIG_VERSION))
    # Version 1 is the first one; later versions add their steps here, e.g. `if version < 2: ...`.
    data['version'] = max(version, CONFIG_VERSION)
    return data


def load_config(path=None):
    """Load settings from path (default: config.toml); missing or broken files give the defaults."""
    path = path or get_config_file()
    if not os.path.exists(path):
        return Config()
    try:
        with open(path, 'rb') as f:
            return dict2cfg(tomllib.load(f))
    except (OSError, tomllib.TOMLDecodeError):
        return Config()


def save_config(cfg, path=None):
    """Save settings to path (default: config.toml), atomically."""
    path = path or get_config_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        tomli_w.dump(cfg2dict(cfg), f)
    os.replace(tmp_path, path)


def list_profiles():
    """Return the names of the term profiles."""
    pdir = get_profiles_dir()
    names = [os.path.splitext(n)[0] for n in os.listdir(pdir) if n.endswith('.txt')] if os.path.isdir(pdir) else []
    return sorted(set(names) | {'default'})


def get_profile_file(name):
    """Return the file of a term profile; the name is made safe for the file system."""
    safe = re.sub(r'[\\/:*?"<>|]', '_', name).strip() or 'default'
    return os.path.join(get_profiles_dir(), safe + '.txt')


def load_terms(name):
    """Return the terms text of a profile ('' if it has none)."""
    try:
        with open(get_profile_file(name), encoding='utf-8') as f:
            return f.read()
    except OSError:
        return ''


def save_terms(name, text):
    """Save the terms text of a profile."""
    os.makedirs(get_profiles_dir(), exist_ok=True)
    with open(get_profile_file(name), 'w', encoding='utf-8') as f:
        f.write(text)


def delete_profile(name):
    """Delete a term profile file if it exists."""
    try:
        os.remove(get_profile_file(name))
    except OSError:
        pass

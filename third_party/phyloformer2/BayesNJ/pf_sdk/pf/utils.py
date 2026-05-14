from collections.abc import Mapping
from enum import Enum
from functools import singledispatch


def instantiate_enum(key: str, enum):
    """Convert from string to Enum member"""
    try:
        return enum[key.upper()]
    except KeyError:
        names = [m.name for m in enum]  # type: ignore
        raise KeyError(
            f"'{key}' is not a valid member of {enum.__name__}, "
            f"valid members are: {names}"
        )


def val_to_toml(val) -> str:
    if isinstance(val, bool):
        return str(val).lower()
    elif isinstance(val, str):
        return f'"{val}"'
    elif isinstance(val, int) or isinstance(val, float):
        return f"{val}"
    elif isinstance(val, tuple) or isinstance(val, list):
        if len(val) == 0:
            return "[]"
        return f"[ {', '.join(val_to_toml(x) for x in val)} ]"
    elif isinstance(val, dict):
        if len(val) == 0:
            return "{}"
        inner = ", ".join(f"{k} = {val_to_toml(v)}" for k, v in val.items())
        return "{ " + inner + " }"
    elif isinstance(val, Enum):
        return f'"{val.name}"'
    else:
        raise TypeError(f"Value: {val} is of an unsupported type")


@singledispatch
def clean_collection(obj):
    """
    Recursively remove None items and empty
    lists/dicts from parent collection
    """
    if obj is not None:
        return obj


@clean_collection.register
def _(dct: dict):
    dct = {k: v_ for k, v in dct.items() if (v_ := clean_collection(v)) is not None}
    return dct if len(dct) > 0 else None


@clean_collection.register
def _(lst: list):
    lst = [x_ for x in lst if (x_ := clean_collection(x)) is not None]
    return lst if len(lst) > 0 else None


def update_nested_dict(base, updater):
    """Recursively update nested dictionnaries"""
    for k, v in updater.items():
        if isinstance(v, Mapping):
            base[k] = update_nested_dict(base.get(k, dict()), v)
        else:
            base[k] = v
    return base

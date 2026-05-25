#!/usr/bin/env python3

import argparse
import json
import shlex
import subprocess


NEGATED_BOOL_FLAGS = {
    "wandb_sync": "--no-wandb-sync",
}


def build_command_from_config(config: dict) -> list[str]:
    command = config.pop("command", "train")
    cmd = ["python", "third_party/phyloformer2/train.py", command]

    for key, value in config.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            else:
                neg = NEGATED_BOOL_FLAGS.get(key)
                if neg is not None:
                    cmd.append(neg)
        elif isinstance(value, list):
            cmd.append(flag)
            cmd.extend(str(x) for x in value)
        else:
            cmd.extend([flag, str(value)])

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run third_party/phyloformer2/train.py from a JSON config."
    )
    parser.add_argument("config", help="Path to PF2 JSON config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a JSON object.")

    cmd = build_command_from_config(dict(config))
    print("Running:\n" + " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

"""
Generate configurations from pyproject.toml.
"""

import toml
from datetime import datetime


current_date = datetime.now().strftime("%Y-%m-%d")
comment = f"# Generated on {current_date}.\n"


def write_cfg():
    """
    Write versioneer configuration from pyproject.toml to setup.cfg.
    """
    with open('pyproject.toml', 'r') as f:
        pyproject = toml.load(f)

    versioneer = pyproject['tool']['versioneer']

    with open('setup.cfg', 'w') as f:
        f.write(comment)
        f.write("[versioneer]\n")
        for key, value in versioneer.items():
            f.write(f"{key} = {value}\n")

    print("Versioneer configuration generated successfully.")


if __name__ == "__main__":
    write_cfg()

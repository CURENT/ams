"""
Generate configurations from pyproject.toml.
"""

import toml
from datetime import datetime


# Get the current date
current_date = datetime.now().strftime("%Y-%m-%d")
comment = f"# Generated on {current_date}.\n"


def write_req():
    """
    Write requirements from pyproject.toml to requirements.txt and requirements-extra.txt.
    """
    with open('pyproject.toml', 'r') as f:
        pyproject = toml.load(f)

    dependencies = pyproject['project']['dependencies']
    dev_dependencies = pyproject['project']['optional-dependencies']['dev']
    doc_dependencies = pyproject['project']['optional-dependencies']['doc']

    # Overwrite requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write(comment)
        for dep in dependencies:
            f.write(dep + '\n')

    # Overwrite requirements-extra.txt
    with open('requirements-extra.txt', 'w') as f:
        f.write(comment)
        max_len = max(len(dep) for dep in dev_dependencies + doc_dependencies)
        for dep in dev_dependencies:
            f.write(f"{dep.ljust(max_len)}  # dev\n")
        for dep in doc_dependencies:
            f.write(f"{dep.ljust(max_len)}  # doc\n")

    print("Requirements files generated successfully.")


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
    write_req()
    write_cfg()

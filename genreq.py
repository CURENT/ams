"""
Generate requirements from pyproject.toml.
"""

import toml
from datetime import datetime


def write_req():
    """
    Write requirements from pyproject.toml to requirements.txt and requirements-extra.txt.
    """
    with open('pyproject.toml', 'r') as f:
        pyproject = toml.load(f)

    dependencies = pyproject['project']['dependencies']
    dev_dependencies = pyproject['project']['optional-dependencies']['dev']
    doc_dependencies = pyproject['project']['optional-dependencies']['doc']

    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    comment = f"# Generated on {current_date}.\n"

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


if __name__ == "__main__":
    write_req()

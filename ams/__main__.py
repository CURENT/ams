"""
AMS main entry point Redirection to main.py
This makes the package callable with python -m andes
"""

from ams.cli import main

if __name__ == '__main__':
    main()

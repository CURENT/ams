"""
Utility functions for loading ams stock test cases,
mainly revised from ``andes.utils.paths``.
"""
import logging
import os
import pathlib
import tempfile

logger = logging.getLogger(__name__)


class DisplayablePath:
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = pathlib.Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = pathlib.Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))


def ams_root():
    """
    Return the root path to the ams source code.
    """

    dir_name = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(dir_name, '..'))


def cases_root():
    """
    Return the root path to the stock cases
    """

    dir_name = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(dir_name, '..', 'cases'))


def tests_root():
    """Return the root path to the stock cases"""
    dir_name = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(dir_name, '..', '..', 'tests'))


def get_case(rpath, check=True):
    """
    Return the path to a stock case for a given path relative to ``ams/cases``.

    To list all cases, use ``ams.list_cases()``.

    Parameters
    ----------
    check : bool
        True to check if file exists

    Examples
    --------
    To get the path to the case `kundur_full.xlsx` under folder `kundur`, do ::

        ams.get_case('kundur/kundur_full.xlsx')

    """
    case_path = os.path.join(cases_root(), rpath)
    case_path = os.path.normpath(case_path)

    if check is True and (not os.path.isfile(case_path)):
        raise FileNotFoundError(f'"{rpath}" is not a valid relative path to a stock case.')
    return case_path


def list_cases(rpath='.', no_print=False):
    """
    List stock cases under a given folder relative to ``ams/cases``
    """
    case_path = os.path.join(cases_root(), rpath)
    case_path = os.path.normpath(case_path)

    tree = DisplayablePath.make_tree(pathlib.Path(case_path))
    out = []
    for path in tree:
        out.append(path.displayable())

    out = '\n'.join(out)
    if no_print is False:
        print(out)
    else:
        return out


def get_config_path(file_name='ams.rc'):
    """
    Return the path of the config file to be loaded.

    Search Priority: 1. current directory; 2. home directory.

    Parameters
    ----------
    file_name : str, optional
        Config file name with the default as ``ams.rc``.

    Returns
    -------
    Config path in string if found; None otherwise.
    """

    conf_path = None
    home_dir = os.path.expanduser('~')

    # test ./ams.conf
    if os.path.isfile(file_name):
        conf_path = file_name
    # test ~/ams.conf
    elif os.path.isfile(os.path.join(home_dir, '.ams', file_name)):
        conf_path = os.path.join(home_dir, '.ams', file_name)

    return conf_path


def get_pycode_path(pycode_path=None, mkdir=False):
    """
    Get the path to the ``pycode`` folder.
    """

    if pycode_path is None:
        pycode_path = os.path.join(get_dot_ams_path(), 'pycode')

    if mkdir is True:
        os.makedirs(pycode_path, exist_ok=True)

    return pycode_path


def get_pkl_path():
    """
    Get the path to the picked/dilled function calls.

    Returns
    -------
    str
        Path to the calls.pkl file

    """
    pkl_name = 'calls.pkl'
    ams_path = get_dot_ams_path()

    if not os.path.exists(ams_path):
        os.makedirs(ams_path)

    pkl_path = os.path.join(ams_path, pkl_name)

    return pkl_path


def get_dot_ams_path():
    """
    Return the path to ``$HOME/.ams``
    """

    return os.path.join(str(pathlib.Path.home()), '.ams')


def get_log_dir():
    """
    Get the directory for log file.

    The default is ``<tempdir>/ams``, where ``<tempdir>`` is provided by ``tempfile.gettempdir()``.

    Returns
    -------
    str
        The path to the temporary logging directory
    """
    tempdir = tempfile.gettempdir()
    path = tempfile.mkdtemp(prefix='ams-', dir=tempdir)
    return path


def confirm_overwrite(outfile, overwrite=None):
    """
    Confirm overwriting a file.
    """

    try:
        if os.path.isfile(outfile):
            if overwrite is None:
                choice = input(f'File "{outfile}" already exist. Overwrite? [y/N]').lower()
                if len(choice) == 0 or choice[0] != 'y':
                    logger.warning(f'File "{outfile}" not overwritten.')
                    return False
            elif overwrite is False:
                return False
    except TypeError:
        pass

    return True


def get_export_path(system, fname, path=None, fmt='csv'):
    """
    Get the absolute export path and the derived file name for
    an AMS system export.

    This function is not intended to be used directly by users.

    Parameters
    ----------
    system : ams.system.System
        The AMS system to export. (Mocked in example)
    fname : str
        The descriptive file name, e.g., 'PTDF', or 'DCOPF'.
    path : str, optional
        The desired output path. For a directory, the file name will be generated;
        For a full file path, the base name will be used with the specified format;
        For None, the current working directory will be used.
    fmt : str, optional
        The file format to export, e.g., 'csv', 'json'. Default is 'csv'.

    Returns
    -------
    tuple : (str, str)
        - The absolute export path (e.g., '/home/user/project/data_Routine.csv').
        - The export file name (e.g., 'data_Routine.csv'), including the format extension.
    """
    # Determine the base name from system.files.fullname or default to "Untitled"
    if system.files.fullname is None:
        logger.info("Input file name not detected. Using `Untitled`.")
        base_name_prefix = f'Untitled_{fname}'
    else:
        base_name_prefix = os.path.splitext(os.path.basename(system.files.fullname))[0]
        base_name_prefix += f'_{fname}'

    target_extension = fmt.lower()  # Ensure consistent extension casing

    if path:
        abs_path = os.path.abspath(path)  # Resolve to absolute path early

        # Check if the provided path is likely intended as a directory
        if not os.path.splitext(abs_path)[1]:  # No extension implies it's a directory
            dir_path = abs_path
            final_file_name = f"{base_name_prefix}.{target_extension}"
            full_export_path = os.path.join(dir_path, final_file_name)
        else:
            # Path includes a filename. Use its directory, and its base name.
            dir_path = os.path.dirname(abs_path)
            # Use the provided filename's base, but enforce the target_extension
            provided_base_filename = os.path.splitext(os.path.basename(abs_path))[0]
            final_file_name = f"{provided_base_filename}.{target_extension}"
            full_export_path = os.path.join(dir_path, final_file_name)
    else:
        # No path provided, use current working directory
        dir_path = os.getcwd()
        final_file_name = f"{base_name_prefix}.{target_extension}"
        full_export_path = os.path.join(dir_path, final_file_name)

    return full_export_path, final_file_name

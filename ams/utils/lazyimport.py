"""
Lazy import utility for optional and deferred dependencies.

Notes
-----
Adapted from ``andes.utils.lazyimport``, which itself derives from
pyforest (https://github.com/8080labs/pyforest).

Original pyforest license (MIT):

    Copyright (c) 2019 8080 Labs

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions: The above copyright
    notice and this permission notice shall be included in all copies or
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""


class LazyImport:
    """
    Defer an import until the first attribute access or call.

    Usage::

        pd = LazyImport('import pandas')
        sps = LazyImport('import scipy.sparse as sps')

    The module is only imported when an attribute is first accessed
    (e.g. ``pd.DataFrame``) or when the object is called.

    Notes
    -----
    Uses ``__double_dunder_methods__`` to minimise name-collision risk with
    attributes of the imported module.
    """

    def __init__(self, import_statement):
        self.__import_statement__ = import_statement
        self.__imported_name__ = import_statement.strip().split()[-1]
        self.__complementary_imports__ = []
        self.__was_imported__ = False

    def __on_import__(self, lazy_import):
        self.__complementary_imports__.append(lazy_import)

    def __maybe_import_complementary_imports__(self):
        for lazy_import in self.__complementary_imports__:
            try:
                lazy_import.__maybe_import__()
            except Exception:  # NOQA
                pass

    def __maybe_import__(self):
        self.__maybe_import_complementary_imports__()
        exec(self.__import_statement__, globals())  # noqa: S102
        self.__was_imported__ = True

    def __dir__(self):
        self.__maybe_import__()
        return eval(f"dir({self.__imported_name__})")

    def __getattr__(self, attribute):
        self.__maybe_import__()
        return eval(f"{self.__imported_name__}.{attribute}")

    def __call__(self, *args, **kwargs):
        self.__maybe_import__()
        return eval(self.__imported_name__)(*args, **kwargs)

    def __repr__(self, *args, **kwargs):
        if self.__was_imported__:
            self.__maybe_import__()
            return f"active LazyImport of {eval(self.__imported_name__)}"
        else:
            return f"lazy LazyImport for '{self.__import_statement__}'"

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

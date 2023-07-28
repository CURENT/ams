"""
This file is used to generate reStructuredText tables for Model and Routine references.
"""

import os
import ams

if not (os.path.isfile('routineref.rst') and os.path.isfile('configref.rst')):

    ss = ams.load(ams.get_case('ieee14/ieee14_opf.xlsx'))

    # write the top-level index file

    out = """.. _routineref:

*******************
Routine reference
*******************

Use the left navigation pane to locate the group and model
and view details.

"""

    out += ss.supported_routines(export='rest')

    out += '\n'
    out += '.. toctree ::\n'
    out += '    :maxdepth: 2\n'
    out += '    :hidden:\n'
    out += '\n'

    file_tpl = '    typedoc/{}\n'

    for type in ss.types.values():
        out += file_tpl.format(type.class_name)

    with open('routineref.rst', 'w') as f:
        f.write(out)

    # write individual files

    os.makedirs('typedoc', exist_ok=True)

    for type in ss.types.values():
        with open(f'typedoc/{type.class_name}.rst', 'w') as f:
            f.write(type.doc_all(export='rest'))

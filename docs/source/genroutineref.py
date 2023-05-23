"""
This file is used to generate reStructuredText tables for Model and Routine references.
"""

import os
import ams

if not (os.path.isfile('routineref.rst') and os.path.isfile('configref.rst')):

    ss = ams.load(ams.get_case('ieee14/ieee14.xlsx'))

    # write the top-level index file

    out = """.. _routineref:

***************
Routine reference
***************

Use the left navigation pane to locate the group and model
and view details.

"""

    out += ss.supported_models(export='rest')

    out += '\n'
    out += '.. toctree ::\n'
    out += '    :maxdepth: 2\n'
    out += '    :hidden:\n'
    out += '\n'

    file_tpl = '    groupdoc/{}\n'

    for group in ss.groups.values():
        out += file_tpl.format(group.class_name)

    with open('routineref.rst', 'w') as f:
        f.write(out)

    # write individual files

    os.makedirs('catdoc', exist_ok=True)

    for group in ss.groups.values():
        with open(f'catdoc/{group.class_name}.rst', 'w') as f:
            f.write(group.doc_all(export='rest'))

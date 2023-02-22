"""Plot the dependencies of ams."""

import os
import subprocess
import ams
import logging

logger = logging.getLogger(__name__)

module_to_exclude = "ams ams.main ams.system ams.core ams.utils numpy"
ams_root = ams.utils.paths.ams_root()
pypower_path = os.path.join(ams_root, 'solver/pypower')
figfolder_path = os.path.normpath(os.path.join(ams_root, '../dev/notes/fig'))

module_to_dps = ["opf", "runopf", "opf_setup", "opf_model", "opf_execute", "pipsopf_solver"]

for module_name in module_to_dps:
    fig_path = os.path.join(figfolder_path, module_name + '.svg')
    file_path = os.path.join(pypower_path, module_name + '.py')
    if os.path.exists(fig_path):
        logger.warning(f"Figure {fig_path} already exists. Skipping...")
    else:
        logger.warning(f"Plotting dependencies of {module_name}...")
        cmd = f"pydeps {file_path} -o {fig_path} --exclude-exact {module_to_exclude} --rmprefix 'ams.solver.pypower.'"
        subprocess.run(cmd, shell=True)
        logger.warning(f"Done. Figure saved to {fig_path}.")

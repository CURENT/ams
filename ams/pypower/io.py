"""
Save and load PYPOWER cases.
"""

import logging
from copy import deepcopy

from os.path import basename, splitext, exists

from numpy import array, zeros, ones, c_

from scipy.io import loadmat, savemat


from sys import stderr

from os.path import basename

from numpy import array, c_, r_, any

from ams.pypower.routines.opffcns import run_userfcn

from ams.pypower.idx import IDX


logger = logging.getLogger(__name__)


def loadcase(casefile,
             return_as_obj=True, expect_gencost=True, expect_areas=True):
    """Returns the individual data matrices or an dict containing them
    as values.

    Here C{casefile} is either a dict containing the keys C{baseMVA}, C{bus},
    C{gen}, C{branch}, C{areas}, C{gencost}, or a string containing the name
    of the file. If C{casefile} contains the extension '.mat' or '.py', then
    the explicit file is searched. If C{casefile} containts no extension, then
    L{loadcase} looks for a '.mat' file first, then for a '.py' file.  If the
    file does not exist or doesn't define all matrices, the function returns
    an exit code as follows:

        0.  all variables successfully defined
        1.  input argument is not a string or dict
        2.  specified extension-less file name does not exist
        3.  specified .mat file does not exist
        4.  specified .py file does not exist
        5.  specified file fails to define all matrices or contains syntax
            error

    If the input data is not a dict containing a 'version' key, it is
    assumed to be a PYPOWER case file in version 1 format, and will be
    converted to version 2 format.

    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)
    """
    if return_as_obj == True:
        expect_gencost = False
        expect_areas = False

    info = 0

    # read data into case object
    if isinstance(casefile, str):
        lasterr = ''
        # check for explicit extension
        if casefile.endswith(('.py', '.mat')):
            rootname, extension = splitext(casefile)
            fname = basename(rootname)
        else:
            # set extension if not specified explicitly
            rootname = casefile
            if exists(casefile + '.mat'):
                extension = '.mat'
            elif exists(casefile + '.py'):
                extension = '.py'
            else:
                info = 2
            fname = basename(rootname)

        # attempt to read file
        if info == 0:
            if extension == '.mat':  # from MAT file
                try:
                    d = loadmat(rootname + extension, struct_as_record=True)
                    if 'ppc' in d or 'mpc' in d:  # it's a MAT/PYPOWER dict
                        if 'ppc' in d:
                            struct = d['ppc']
                        else:
                            struct = d['mpc']
                        val = struct[0, 0]

                        s = {}
                        for a in val.dtype.names:
                            s[a] = val[a]
                    else:  # individual data matrices
                        d['version'] = '1'

                        s = {}
                        for k, v in d.items():
                            s[k] = v

                    s['baseMVA'] = s['baseMVA'][0]  # convert array to float

                except IOError as e:
                    info = 3
                    lasterr = str(e)
            elif extension == '.py':  # from Python file
                try:
                    if PY2:
                        execfile(rootname + extension)
                    else:
                        exec(compile(open(rootname + extension).read(),
                                     rootname + extension, 'exec'))

                    try:  # assume it returns an object
                        s = eval(fname)()
                    except ValueError as e:
                        info = 4
                        lasterr = str(e)
                    # if not try individual data matrices
                    if info == 0 and not isinstance(s, dict):
                        s = {}
                        s['version'] = '1'
                        if expect_gencost:
                            try:
                                s['baseMVA'], s['bus'], s['gen'], s['branch'], \
                                    s['areas'], s['gencost'] = eval(fname)()
                            except IOError as e:
                                info = 4
                                lasterr = str(e)
                        else:
                            if return_as_obj:
                                try:
                                    s['baseMVA'], s['bus'], s['gen'], \
                                        s['branch'], s['areas'], \
                                        s['gencost'] = eval(fname)()
                                except ValueError as e:
                                    try:
                                        s['baseMVA'], s['bus'], s['gen'], \
                                            s['branch'] = eval(fname)()
                                    except ValueError as e:
                                        info = 4
                                        lasterr = str(e)
                            else:
                                try:
                                    s['baseMVA'], s['bus'], s['gen'], \
                                        s['branch'] = eval(fname)()
                                except ValueError as e:
                                    info = 4
                                    lasterr = str(e)

                except IOError as e:
                    info = 4
                    lasterr = str(e)

                if info == 4 and exists(rootname + '.py'):
                    info = 5
                    err5 = lasterr

    elif isinstance(casefile, dict):
        s = deepcopy(casefile)
    else:
        info = 1

    # check contents of dict
    if info == 0:
        # check for required keys
        if (s['baseMVA'] is None or s['bus'] is None
            or s['gen'] is None or s['branch'] is None) or \
            (expect_gencost and s['gencost'] is None) or \
                (expect_areas and s['areas'] is None):
            info = 5  # missing some expected fields
            err5 = 'missing data'
        else:
            # remove empty areas if not needed
            if hasattr(s, 'areas') and (len(s['areas']) == 0) and (not expect_areas):
                del s['areas']

            # all fields present, copy to ppc
            ppc = deepcopy(s)
            if not hasattr(ppc, 'version'):  # hmm, struct with no 'version' field
                if ppc['gen'].shape[1] < 21:  # version 2 has 21 or 25 cols
                    ppc['version'] = '1'
                else:
                    ppc['version'] = '2'

            if (ppc['version'] == '1'):
                # convert from version 1 to version 2
                ppc['gen'], ppc['branch'] = ppc_1to2(ppc['gen'], ppc['branch'])
                ppc['version'] = '2'

    if info == 0:  # no errors
        if return_as_obj:
            return ppc
        else:
            result = [ppc['baseMVA'], ppc['bus'], ppc['gen'], ppc['branch']]
            if expect_gencost:
                if expect_areas:
                    result.extend([ppc['areas'], ppc['gencost']])
                else:
                    result.extend([ppc['gencost']])
            return result
    else:  # error encountered
        if info == 1:
            logger.debug('Input arg should be a case or a string '
                         'containing a filename\n')
        elif info == 2:
            logger.debug('Specified case not a valid file\n')
        elif info == 3:
            logger.debug('Specified MAT file does not exist\n')
        elif info == 4:
            logger.debug('Specified Python file does not exist\n')
        elif info == 5:
            logger.debug('Syntax error or undefined data '
                         'matrix(ices) in the file\n')
        else:
            logger.debug('Unknown error encountered loading case.\n')

        return info


def ppc_1to2(gen, branch):
    # -----  gen  -----
    # use the version 1 values for column names
    if gen.shape[1] >= IDX.gen.APF:
        logger.debug('ppc_1to2: gen matrix appears to already be in '
                     'version 2 format\n')
        return gen, branch

    shift = IDX.gen.MU_PMAX - IDX.gen.PMIN - 1
    tmp = array([IDX.gen.MU_PMAX, IDX.gen.MU_PMIN, IDX.gen.MU_QMAX, IDX.gen.MU_QMIN]) - shift
    mu_Pmax, mu_Pmin, mu_Qmax, mu_Qmin = tmp

    # add extra columns to gen
    tmp = zeros((gen.shape[0], shift))
    if gen.shape[1] >= mu_Qmin:
        gen = c_[gen[:, 0:IDX.gen.PMIN + 1], tmp, gen[:, mu_Pmax:mu_Qmin]]
    else:
        gen = c_[gen[:, 0:IDX.gen.PMIN + 1], tmp]

    # -----  branch  -----
    # use the version 1 values for column names
    shift = IDX.branch.PF - IDX.branch.BR_STATUS - 1
    tmp = array([IDX.branch.PF, IDX.branch.QF, IDX.branch.PT,
                IDX.branch.QT, IDX.branch.MU_SF, IDX.branch.MU_ST]) - shift
    Pf, Qf, Pt, Qt, mu_Sf, mu_St = tmp

    # add extra columns to branch
    tmp = ones((branch.shape[0], 1)) * array([-360, 360])
    tmp2 = zeros((branch.shape[0], 2))
    if branch.shape[1] >= mu_St - 1:
        branch = c_[branch[:, 0:IDX.branch.BR_STATUS + 1], tmp, branch[:, IDX.branch.PF - 1:IDX.branch.MU_ST + 1], tmp2]
    elif branch.shape[1] >= IDX.branch.QT - 1:
        branch = c_[branch[:, 0:IDX.branch.BR_STATUS + 1], tmp, branch[:, IDX.branch.PF - 1:IDX.branch.QT + 1]]
    else:
        branch = c_[branch[:, 0:IDX.branch.BR_STATUS + 1], tmp]

    return gen, branch


def savecase(fname, ppc, comment=None, version='2'):
    """Saves a PYPOWER case file, given a filename and the data.

    Writes a PYPOWER case file, given a filename and data dict. The C{fname}
    parameter is the name of the file to be created or overwritten. Returns
    the filename, with extension added if necessary. The optional C{comment}
    argument is either string (single line comment) or a list of strings which
    are inserted as comments. When using a PYPOWER case dict, if the
    optional C{version} argument is '1' it will modify the data matrices to
    version 1 format before saving.

    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)
    """
    ppc_ver = ppc["version"] = version
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]
    areas = ppc["areas"] if "areas" in ppc else None
    gencost = ppc["gencost"] if "gencost" in ppc else None

    # modifications for version 1 format
    if ppc_ver == "1":
        raise NotImplementedError

    # verify valid filename
    l = len(fname)
    rootname = ""
    if l > 2:
        if fname[-3:] == ".py":
            rootname = fname[:-3]
            extension = ".py"
        elif l > 4:
            if fname[-4:] == ".mat":
                rootname = fname[:-4]
                extension = ".mat"

    if not rootname:
        rootname = fname
        extension = ".py"
        fname = rootname + extension

    indent = '    '  # four spaces
    indent2 = indent + indent

    # open and write the file
    if extension == ".mat":  # MAT-file
        ppc_mat = {}
        ppc_mat['version'] = ppc_ver
        ppc_mat['baseMVA'] = baseMVA
        ppc_keys = ['bus', 'gen', 'branch']
        # Assign non-scalar values as NumPy arrays
        for key in ppc_keys:
            ppc_mat[key] = array(ppc[key])
        if 'areas' in ppc:
            ppc_mat['areas'] = array(ppc['areas'])
        if 'gencost' in ppc:
            ppc_mat['gencost'] = array(ppc['gencost'])
        if "A" in ppc and len(ppc["A"]) > 0:
            ppc_mat["A"] = array(ppc["A"])
            if "l" in ppc and len(ppc["l"]) > 0:
                ppc_mat["l"] = array(ppc["l"])
            if "u" in ppc and len(ppc["u"]) > 0:
                ppc_mat["u"] = array(ppc["u"])
        if "N" in ppc and len(ppc["N"]) > 0:
            ppc_mat["N"] = array(ppc["N"])
            if "H" in ppc and len(ppc["H"]) > 0:
                ppc_mat["H"] = array(ppc["H"])
            if "fparm" in ppc and len(ppc["fparm"]) > 0:
                ppc_mat["fparm"] = array(ppc["fparm"])
            ppc_mat["Cw"] = array(ppc["Cw"])
        if 'z0' in ppc or 'zl' in ppc or 'zu' in ppc:
            if 'z0' in ppc and len(ppc['z0']) > 0:
                ppc_mat['z0'] = array(ppc['z0'])
            if 'zl' in ppc and len(ppc['zl']) > 0:
                ppc_mat['zl'] = array(ppc['zl'])
            if 'zu' in ppc and len(ppc['zu']) > 0:
                ppc_mat['zu'] = array(ppc['zu'])
        if 'userfcn' in ppc and len(ppc['userfcn']) > 0:
            ppc_mat['userfcn'] = array(ppc['userfcn'])
        elif 'userfcn' in ppc:
            ppc_mat['userfcn'] = ppc['userfcn']
        for key in ['x', 'f']:
            if key in ppc:
                ppc_mat[key] = ppc[key]
        for key in ['lin', 'order', 'nln', 'var', 'raw', 'mu']:
            if key in ppc:
                ppc_mat[key] = array(ppc[key])

        savemat(fname, ppc_mat)
    else:  # Python file
        try:
            fd = open(fname, "wb")
        except Exception as detail:
            logger.debug("savecase: %s.\n" % detail)
            return fname

        # function header, etc.
        if ppc_ver == "1":
            raise NotImplementedError
#            if (areas != None) and (gencost != None) and (len(gencost) > 0):
#                fd.write('function [baseMVA, bus, gen, branch, areas, gencost] = %s\n' % rootname)
#            else:
#                fd.write('function [baseMVA, bus, gen, branch] = %s\n' % rootname)
#            prefix = ''
        else:
            fd.write('def %s():\n' % basename(rootname))
            prefix = 'ppc'
        if comment:
            if isinstance(comment, str):
                fd.write('#%s\n' % comment)
            elif isinstance(comment, list):
                for c in comment:
                    fd.write('#%s\n' % c)
        fd.write('\n%s## PYPOWER Case Format : Version %s\n' % (indent, ppc_ver))
        if ppc_ver != "1":
            fd.write("%sppc = {'version': '%s'}\n" % (indent, ppc_ver))
        fd.write('\n%s##-----  Power Flow Data  -----##\n' % indent)
        fd.write('%s## system MVA base\n' % indent)
        fd.write("%s%s['baseMVA'] = %.9g\n" % (indent, prefix, baseMVA))

        # bus data
        ncols = bus.shape[1]
        fd.write('\n%s## bus data\n' % indent)
        fd.write('%s# bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin' % indent)
        if ncols >= IDX.bus.MU_VMIN + 1:  # opf SOLVED, save with lambda's & mu's
            fd.write('lam_P lam_Q mu_Vmax mu_Vmin')
        fd.write("\n%s%s['bus'] = array([\n" % (indent, prefix))
        if ncols < IDX.bus.MU_VMIN + 1:  # opf NOT SOLVED, save without lambda's & mu's
            for i in range(bus.shape[0]):
                fd.write('%s[%d, %d, %.9g, %.9g, %.9g, %.9g, %d, %.9g, %.9g, %.9g, %d, %.9g, %.9g],\n' %
                         ((indent2,) + tuple(bus[i, :IDX.bus.VMIN + 1])))
        else:  # opf SOLVED, save with lambda's & mu's
            for i in range(bus.shape[0]):
                fd.write('%s[%d, %d, %.9g, %.9g, %.9g, %.9g, %d, %.9g, %.9g, %.9g, %d, %.9g, %.9g, %.4f, %.4f, %.4f, %.4f],\n' % (
                    (indent2,) + tuple(bus[i, :IDX.bus.MU_VMIN + 1])))
        fd.write('%s])\n' % indent)

        # generator data
        ncols = gen.shape[1]
        fd.write('\n%s## generator data\n' % indent)
        fd.write('%s# bus Pg Qg Qmax Qmin Vg mBase status Pmax Pmin' % indent)
        if ppc_ver != "1":
            fd.write(' Pc1 Pc2 Qc1min Qc1max Qc2min Qc2max ramp_agc ramp_10 ramp_30 ramp_q apf')
        if ncols >= IDX.gen.MU_QMIN + 1:             # opf SOLVED, save with mu's
            fd.write(' mu_Pmax mu_Pmin mu_Qmax mu_Qmin')
        fd.write("\n%s%s['gen'] = array([\n" % (indent, prefix))
        if ncols < IDX.gen.MU_QMIN + 1:  # opf NOT SOLVED, save without mu's
            if ppc_ver == "1":
                for i in range(gen.shape[0]):
                    fd.write('%s[%d, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %d, %.9g, %.9g],\n' %
                             ((indent2,) + tuple(gen[i, :IDX.gen.PMIN + 1])))
            else:
                for i in range(gen.shape[0]):
                    fd.write('%s[%d, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %d, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g],\n' % (
                        (indent2,) + tuple(gen[i, :IDX.gen.APF + 1])))
        else:
            if ppc_ver == "1":
                for i in range(gen.shape[0]):
                    fd.write(
                        '%s[%d, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %d, %.9g, %.9g, %.4f, %.4f, %.4f, %.4f],\n' %
                        ((indent2,) + tuple(gen[i, : IDX.gen.MU_QMIN + 1])))
            else:
                for i in range(gen.shape[0]):
                    fd.write('%s[%d, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %d, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.4f, %.4f, %.4f, %.4f],\n' % (
                        (indent2,) + tuple(gen[i, :IDX.gen.MU_QMIN + 1])))
        fd.write('%s])\n' % indent)

        # branch data
        ncols = branch.shape[1]
        fd.write('\n%s## branch data\n' % indent)
        fd.write('%s# fbus tbus r x b rateA rateB rateC ratio angle status' % indent)
        if ppc_ver != "1":
            fd.write(' angmin angmax')
        if ncols >= IDX.branch.QT + 1:  # power flow SOLVED, save with line flows
            fd.write(' Pf Qf Pt Qt')
        if ncols >= IDX.branch.MU_ST + 1:  # opf SOLVED, save with mu's
            fd.write(' mu_Sf mu_St')
            if ppc_ver != "1":
                fd.write(' mu_angmin mu_angmax')
        fd.write('\n%s%s[\'branch\'] = array([\n' % (indent, prefix))
        if ncols < IDX.branch.QT + 1:  # power flow NOT SOLVED, save without line flows or mu's
            if ppc_ver == "1":
                for i in range(branch.shape[0]):
                    fd.write('%s[%d, %d, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %d],\n' %
                             ((indent2,) + tuple(branch[i, :IDX.branch.BR_STATUS + 1])))
            else:
                for i in range(branch.shape[0]):
                    fd.write(
                        '%s[%d, %d, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %d, %.9g, %.9g],\n' %
                        ((indent2,) + tuple(branch[i, : IDX.branch.ANGMAX + 1])))
        elif ncols < IDX.branch.MU_ST + 1:  # power flow SOLVED, save with line flows but without mu's
            if ppc_ver == "1":
                for i in range(branch.shape[0]):
                    fd.write(
                        '%s[%d, %d, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %d, %.4f, %.4f, %.4f, %.4f],\n' %
                        ((indent2,) + tuple(branch[i, : IDX.branch.QT + 1])))
            else:
                for i in range(branch.shape[0]):
                    fd.write('%s[%d, %d, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %d, %.9g, %.9g, %.4f, %.4f, %.4f, %.4f],\n' % (
                        (indent2,) + tuple(branch[i, :IDX.branch.QT + 1])))
        else:  # opf SOLVED, save with lineflows & mu's
            if ppc_ver == "1":
                for i in range(branch.shape[0]):
                    fd.write('%s[%d, %d, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %d, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f],\n' % (
                        (indent2,) + tuple(branch[i, :IDX.branch.MU_ST + 1])))
            else:
                for i in range(branch.shape[0]):
                    fd.write('%s[%d, %d, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %d, %.9g, %.9g, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f],\n' % (
                        (indent2,) + tuple(branch[i, :IDX.branch.MU_ANGMAX + 1])))
        fd.write('%s])\n' % indent)

        # OPF data
        if (areas is not None) and (len(areas) > 0) or (gencost is not None) and (len(gencost) > 0):
            fd.write('\n%s##-----  OPF Data  -----##' % indent)
        if (areas is not None) and (len(areas) > 0):
            # area data
            fd.write('\n%s## area data\n' % indent)
            fd.write('%s# area refbus\n' % indent)
            fd.write("%s%s['areas'] = array([\n" % (indent, prefix))
            if len(areas) > 0:
                for i in range(areas.shape[0]):
                    fd.write('%s[%d, %d],\n' % ((indent2,) + tuple(areas[i, :IDX.area.PRICE_REF_BUS + 1])))
            fd.write('%s])\n' % indent)
        if gencost is not None and len(gencost) > 0:
            # generator cost data
            fd.write('\n%s## generator cost data\n' % indent)
            fd.write('%s# 1 startup shutdown n x1 y1 ... xn yn\n' % indent)
            fd.write('%s# 2 startup shutdown n c(n-1) ... c0\n' % indent)
            fd.write('%s%s[\'gencost\'] = array([\n' % (indent, prefix))
            if len(gencost > 0):
                if any(gencost[:, IDX.cost.MODEL] == IDX.cost.PW_LINEAR):
                    n1 = 2 * max(gencost[gencost[:, IDX.cost.MODEL] == IDX.cost.PW_LINEAR,  IDX.cost.NCOST])
                else:
                    n1 = 0
                if any(gencost[:, IDX.cost.MODEL] == IDX.cost.POLYNOMIAL):
                    n2 = max(gencost[gencost[:, IDX.cost.MODEL] == IDX.cost.POLYNOMIAL, IDX.cost.NCOST])
                else:
                    n2 = 0
                n = int(max([n1, n2]))
                if gencost.shape[1] < n + 4:
                    logger.debug('savecase: gencost data claims it has more columns than it does\n')
                template = '%s[%d, %.9g, %.9g, %d'
                for i in range(n):
                    template = template + ', %.9g'
                template = template + '],\n'
                for i in range(gencost.shape[0]):
                    fd.write(template % ((indent2,) + tuple(gencost[i])))
            fd.write('%s])\n' % indent)

        # generalized OPF user data
        if ("A" in ppc) and (len(ppc["A"]) > 0) or ("N" in ppc) and (len(ppc["N"]) > 0):
            fd.write('\n%s##-----  Generalized OPF User Data  -----##' % indent)

        # user constraints
        if ("A" in ppc) and (len(ppc["A"]) > 0):
            # A
            fd.write('\n%s## user constraints\n' % indent)
            print_sparse(fd, prefix + "['A']", ppc["A"])
            if ("l" in ppc) and (len(ppc["l"]) > 0) and ("u" in ppc) and (len(ppc["u"]) > 0):
                fd.write('%slu = array([\n' % indent)
                for i in range(len(ppc["l"])):
                    fd.write('%s[%.9g, %.9g],\n' % (indent2, ppc["l"][i], ppc["u"][i]))
                fd.write('%s])\n' % indent)
                fd.write("%s%s['l'] = lu[:, 0]\n" % (indent, prefix))
                fd.write("%s%s['u'] = lu[:, 1]\n\n" % (indent, prefix))
            elif ("l" in ppc) and (len(ppc["l"]) > 0):
                fd.write("%s%s['l'] = array([\n" % (indent, prefix))
                for i in range(len(l)):
                    fd.write('%s[%.9g],\n' % (indent2, ppc["l"][i]))
                fd.write('%s])\n\n' % indent)
            elif ("u" in ppc) and (len(ppc["u"]) > 0):
                fd.write("%s%s['u'] = array([\n" % (indent, prefix))
                for i in range(len(l)):
                    fd.write('%s[%.9g],\n' % (indent2, ppc["u"][i]))
                fd.write('%s])\n\n' % indent)

        # user costs
        if ("N" in ppc) and (len(ppc["N"]) > 0):
            fd.write('\n%s## user costs\n' % indent)
            print_sparse(fd, prefix + "['N']", ppc["N"])
            if ("H" in ppc) and (len(ppc["H"]) > 0):
                print_sparse(fd, prefix + "['H']", ppc["H"])
            if ("fparm" in ppc) and (len(ppc["fparm"]) > 0):
                fd.write("%sCw_fparm = array([\n" % indent)
                for i in range(ppc["Cw"]):
                    fd.write('%s[%.9g, %d, %.9g, %.9g, %.9g],\n' %
                             ((indent2,) + tuple(ppc["Cw"][i]) + tuple(ppc["fparm"][i, :])))
                fd.write('%s])\n' % indent)
                fd.write('%s%s[\'Cw\']    = Cw_fparm[:, 0]\n' % (indent, prefix))
                fd.write("%s%s['fparm'] = Cw_fparm[:, 1:5]\n" % (indent, prefix))
            else:
                fd.write("%s%s['Cw'] = array([\n" % (indent, prefix))
                for i in range(len(ppc["Cw"])):
                    fd.write('%s[%.9g],\n' % (indent2, ppc["Cw"][i]))
                fd.write('%s])\n' % indent)

        # user vars
        if ('z0' in ppc) or ('zl' in ppc) or ('zu' in ppc):
            fd.write('\n%s## user vars\n' % indent)
            if ('z0' in ppc) and (len(ppc['z0']) > 0):
                fd.write('%s%s["z0"] = array([\n' % (indent, prefix))
                for i in range(len(ppc['z0'])):
                    fd.write('%s[%.9g],\n' % (indent2, ppc["z0"]))
                fd.write('%s])\n' % indent)
            if ('zl' in ppc) and (len(ppc['zl']) > 0):
                fd.write('%s%s["zl"] = array([\n' % (indent2, prefix))
                for i in range(len(ppc['zl'])):
                    fd.write('%s[%.9g],\n' % (indent2, ppc["zl"]))
                fd.write('%s])\n' % indent)
            if ('zu' in ppc) and (len(ppc['zu']) > 0):
                fd.write('%s%s["zu"] = array([\n' % (indent, prefix))
                for i in range(len(ppc['zu'])):
                    fd.write('%s[%.9g],\n' % (indent2, ppc["zu"]))
                fd.write('%s])\n' % indent)

        # execute userfcn callbacks for 'savecase' stage
        if 'userfcn' in ppc:
            run_userfcn(ppc["userfcn"], 'savecase', ppc, fd, prefix)

        fd.write('\n%sreturn ppc\n' % indent)

        # close file
        fd.close()

    return fname


def print_sparse(fd, varname, A):
    A = A.tocoo()
    i, j, s = A.row, A.col, A.data
    m, n = A.shape

    if len(s) == 0:
        fd.write('%s = sparse((%d, %d))\n' % (varname, m, n))
    else:
        fd.write('ijs = array([\n')
    for k in range(len(i)):
        fd.write('[%d, %d, %.9g],\n' % (i[k], j[k], s[k]))

    fd.write('])\n')
    fd.write('%s = sparse(ijs[:, 0], ijs[:, 1], ijs[:, 2], %d, %d)\n' % (varname, m, n))

import sys
from pathlib import Path

def read_params(file):
    if not Path(file).is_file():
        sys.exit(f"Cannot find file {file}")
    params = {}
    with open(file, "r") as f:
        for l in f:
            l = l.split("#", 1)[0].strip()
            if not l:
                continue
            bits = l.split()
            if len(bits) != 2:
                print("unrecognized line in params")
                print("   ", l)
                sys.exit(1)
            params[bits[0]] = bits[1]
    params['npoly'] = int(params['npoly'])
    params['nPBI'] = int(params['nPBI'])
    params['nIM'] = int(params['nIM'])
    params['nFe'] = int(params['nFe'])
    params['fe_charge'] = 0
    if 'fe_charge' in params:
        params['feCharge'] = int(params['fe_charge'])
    if 'nNO3' in params:
        params['nNO3'] = int(params['nNO3'])
    else:
        params['nNO3'] = 0
    if 'L0' in params:
        params['L0'] = float(params['L0'])
    if 'density' in params:
        params['density'] = float(params['density'])
    if not 'molecule' in params:
        sys.exit("molecule missing in params")
    if not 'acpath' in params:
        # OK so long as we already have prmtop
        params['acpath'] = None
    elif not Path(params['acpath']).is_dir():
        print("acpath", params['acpath'], "must exist")
        sys.exit(1)
    if 'target_hours' in params:
        params['target_hours'] = int(params['target_hours'])
    if 'total_steps' in params:
        params['total_steps'] = int(params['total_steps'])
    return params


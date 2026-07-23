import numpy as np

def movload(fname):
    # loads .mov files given the path of the file. The .mov
    # files have a specific custom format hence the need for a custom function.
    # Column count is inferred from the first data line (not hardcoded).
    A = []
    fid = open(fname, 'rt')
    if fid == -1:
        raise Exception('Could not open ' + fname)

    trial = 0
    n_cols = None
    for line in fid:
        if line[0] == 'T':
            print('Trial: ', line.split()[1])
            a = int(line.split()[1])
            trial += 1
            if a != trial:
                print('Trials out of sequence')
                trial = a
            A.append(np.empty((0, n_cols if n_cols is not None else 0)))
        else:
            lineData = line.strip().split('\t')
            if n_cols is None:
                n_cols = len(lineData)
                # Retroactively set width on the current trial's empty buffer
                A[trial - 1] = np.empty((0, n_cols))
            elif len(lineData) != n_cols:
                raise ValueError(
                    f"Inconsistent column count in {fname}: expected {n_cols}, "
                    f"got {len(lineData)} (trial {trial})"
                )
            a = np.array([float(x) for x in lineData], ndmin=2)
            A[trial - 1] = np.vstack((A[trial - 1], a))

    fid.close()
    return A

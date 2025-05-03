import json
import sys
import os
import glob
import numpy as np
import h5py

def file_len(fname):
    """Counts the number of lines in the file with name fname."""
    with open(fname, 'r', encoding='utf-8') as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

def process_file(fname, options):
    """Process a review JSON lines file and return a numpy array with the data."""
    print(f"Processing {fname} …")
    limit = options.nwords
    stop  = options.lines
    n_features = 4
    total_lines = min(file_len(fname), stop)
    all_data = np.zeros((total_lines, limit + n_features), dtype=np.int16)

    with open(fname, 'r', encoding='utf-8') as ifile:
        for i, line in enumerate(ifile):
            if i % 10000 == 0:
                print(f"  line {i}")
            if i >= stop:
                break
            data = json.loads(line)
            codes = data['text']
            if not options.keep_unknown:
                codes = [c for c in codes if c != 1]
            # metadata
            all_data[i, 0] = data['stars']
            all_data[i, 1] = data['useful']
            all_data[i, 2] = data['funny']
            all_data[i, 3] = data['cool']
            # text codes
            truncated = codes[:limit]
            all_data[i, n_features:n_features+len(truncated)] = truncated

    print(f"{fname} done")
    return all_data

def finalize(results, outfname):
    """Concatenate all results and write to the given HDF5 filename."""
    print("Concatenating arrays …")
    data = np.concatenate(results, axis=0)
    print(f"Final array shape: {data.shape}")
    print(f"Writing to HDF5 file: {outfname}")
    with h5py.File(outfname, 'w') as h5:
        h5.create_dataset('reviews', data=data)

def parse_args():
    from optparse import OptionParser
    from base import setopts

    usage = "usage: %prog [options] <file_pattern>"
    parser = OptionParser(usage=usage)
    setopts(parser)

    parser.add_option("-u", "--keep-unknown",
                      dest="keep_unknown", action="store_true", default=False,
                      help="keep unknown codes (default: drop them)")
    parser.add_option("-n", "--nwords",
                      dest="nwords", type="int", default=250,
                      help="max number of words per review (default: %default)")
    parser.add_option("-o", "--output",
                      dest="outfile", default="data.h5",
                      help="output HDF5 file name (default: %default)")

    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.print_usage()
        sys.exit(1)
    options.pattern = args[0]
    return options

if __name__ == '__main__':
    import parallelize

    options = parse_args()

    olddir = os.getcwd()
    os.chdir(options.datadir)

    fnames = glob.glob(options.pattern)
    nproc = len(fnames) if options.parallel else None

    results = parallelize.run(process_file, fnames, nproc, options)
    finalize(results, options.outfile)

    os.chdir(olddir)

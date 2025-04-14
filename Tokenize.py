#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import nltk
import sys
import re
import string

# Download 'punkt' tokenizer if not already available.
nltk.download('punkt')


def output_fname(input_fname):
    """
    Generate the output filename by appending '_tok' before the file extension.
    For example, "input.json" becomes "input_tok.json".
    """
    return os.path.splitext(input_fname)[0] + '_tok.json'


def process_file(fname, options):
    """
    Tokenize the review data in file 'fname' and write the output to a new file.

    Modifications:
      - The text is preprocessed by adding spaces around common punctuation marks
        so that tokens like "however,but" become "however , but".
      - After tokenization, each token has its apostrophes removed (so "don't" becomes "dont"),
        then tokens that are entirely punctuation or that contain any digit are filtered out.
      - The output is written to a JSON file with '_tok.json' appended to the original name.
    """
    print('Opening', fname)
    ofname = output_fname(fname)
    ifile = open(fname, 'r', encoding='utf-8', errors='replace')
    ofile = open(ofname, 'w', encoding='utf-8')

    for i, line in enumerate(ifile):
        if i % 1000 == 0:
            print(i)
        if i == options.lines:
            break
        # Convert the JSON line into a dictionary.
        data = json.loads(line)
        # Extract the review text (assume the field name is "text").
        text = data['text']

        # Preprocess the text:
        # 1. Insert spaces around punctuation (comma, period, exclamation, question marks).
        text = re.sub(r'([,.!?])', r' \1 ', text)
        # 2. Replace multiple spaces with a single space.
        text = re.sub(r'\s+', ' ', text)

        # Tokenize the text using nltk's word_tokenize.
        tokens = nltk.word_tokenize(text)
        # Process tokens:
        #   - Remove any apostrophes from each token.
        #   - Convert to lower case.
        #   - Filter out tokens that are solely punctuation or contain digits.
        words = []
        for token in tokens:
            # Remove apostrophes. For example, "don't" becomes "dont".
            token = token.replace("'", "")
            # If the token becomes empty after removal, skip it.
            if not token:
                continue
            # Check if token is composed entirely of punctuation or contains any digit.
            if all(ch in string.punctuation for ch in token) or re.search(r'\d', token):
                continue
            words.append(token.lower())

        # Update JSON data with the processed word list.
        data['text'] = words
        line_out = json.dumps(data)
        ofile.write(line_out + '\n')

    ifile.close()
    ofile.close()


def parse_args():
    """
    Parse command line arguments.

    This function uses OptionParser and a custom 'setopts' function (assumed to be defined in 'base').
    """
    from optparse import OptionParser
    from base import setopts
    usage = "usage: %prog [options] <file_pattern>"
    parser = OptionParser(usage=usage)
    setopts(parser)
    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.print_usage()
        sys.exit(1)
    # 'pattern' should match the files to process (e.g., 'xa?').
    pattern = args[0]
    return options, pattern


if __name__ == '__main__':
    import glob
    import parallelize
    from multiprocessing import Pool

    # Parse command-line arguments.
    options, pattern = parse_args()
    olddir = os.getcwd()
    os.chdir(options.datadir)

    fnames = glob.glob(pattern)

    nprocesses = len(fnames) if options.parallel else None
    results = parallelize.run(process_file, fnames, nprocesses, options)

    os.chdir(olddir)

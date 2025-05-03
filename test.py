#!/usr/bin/env python
import argparse
import h5py
import numpy as np
from Vocabulary import Vocabulary


def main():
    parser = argparse.ArgumentParser(
        description="Print first n rows of the HDF5 dataset and decode the first review."
    )

    parser.add_argument("h5file", help="HDF5 file which is in data/, e.g., data.h5")

    parser.add_argument("-l", "--preview", type=int, default=10,
                        help="Number of rows to preview, default is 10.")

    parser.add_argument("-v", "--vocab", default=None,
                        help="Path to vocabulary index file. If not specified, it is assumed to be in the same directory as the HDF5 file under the name 'index'")

    args = parser.parse_args()

    h5file_path = "dataset/"+args.h5file

    if args.vocab:
        vocab_path = args.vocab
    else:
        import os
        vocab_path = os.path.join(os.path.dirname(h5file_path), 'index')


    with h5py.File(h5file_path, 'r') as h5:
        # 假设感兴趣的数据集名称为 'reviews'
        d = h5['reviews']
        print("Dataset shape:", d.shape)
        print("Dataset dtype:", d.dtype)

        print(f"\nPreview of first {args.preview} rows:")
        #
        preview_rows = min(args.preview, d.shape[0])
        for i in range(preview_rows):
            print(f"Row {i}: {d[i]}")


        vocab = Vocabulary.load(vocab_path)

        rev1 = d[0, 1:]
        rev1 = rev1[rev1 != 0]
        decoded_review = ' '.join(vocab.decode(rev1))
        print("\nDecoded first review:")
        print(decoded_review)


if __name__ == '__main__':
    main()

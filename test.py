#!/usr/bin/env python
import argparse
import h5py
import numpy as np
from Vocabulary import Vocabulary


def main():
    parser = argparse.ArgumentParser(
        description="Print first n rows of the HDF5 dataset and decode the first review."
    )
    # 第一个位置参数是 HDF5 文件路径
    parser.add_argument("h5file", help="HDF5 file which is in data/, e.g., data.h5")

    parser.add_argument("-l", "--preview", type=int, default=10,
                        help="Number of rows to preview, default is 10.")
    # 如果词汇表文件不在HDF5文件相同目录下，可以用 -v 指定（默认使用 datadir + '/index'）
    parser.add_argument("-v", "--vocab", default=None,
                        help="Path to vocabulary index file. If not specified, it is assumed to be in the same directory as the HDF5 file under the name 'index'")

    args = parser.parse_args()

    h5file_path = "dataset/"+args.h5file
    # 如果没有指定词汇表路径，则采用 h5file 的同级目录下的 'index'
    if args.vocab:
        vocab_path = args.vocab
    else:
        import os
        vocab_path = os.path.join(os.path.dirname(h5file_path), 'index')

    # 打开 HDF5 文件
    with h5py.File(h5file_path, 'r') as h5:
        # 假设感兴趣的数据集名称为 'reviews'
        d = h5['reviews']
        print("Dataset shape:", d.shape)
        print("Dataset dtype:", d.dtype)

        print(f"\nPreview of first {args.preview} rows:")
        # 输出前 n 行（若文件行数少于预览数，则只输出全部行）
        preview_rows = min(args.preview, d.shape[0])
        for i in range(preview_rows):
            print(f"Row {i}: {d[i]}")

        # 加载词汇表（Vocabulary 类需自行实现 load() 方法）
        vocab = Vocabulary.load(vocab_path)
        # 对第一条评论进行解码
        # 假设数据集中每条评论为一行，评论的第一个元素可能为标记或其他信息，
        # 因此取 d[0, 1:]（并去除值为 0 的填充部分）
        rev1 = d[0, 1:]
        rev1 = rev1[rev1 != 0]
        decoded_review = ' '.join(vocab.decode(rev1))
        print("\nDecoded first review:")
        print(decoded_review)


if __name__ == '__main__':
    main()

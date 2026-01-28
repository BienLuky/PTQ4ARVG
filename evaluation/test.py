import sys, os
import argparse
import torch
from evaluator import test_fid_pc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_batch", help="path to reference batch npz file")
    parser.add_argument("sample_batch", help="path to sample batch npz file")
    args = parser.parse_args()
    test_fid_pc(args.ref_batch, args.sample_batch)


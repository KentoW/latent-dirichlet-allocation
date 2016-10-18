# -*- coding: utf-8 -*-
import sys
import argparse


def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optinon", dest="option", default=1, type=int, help="description")
    args = parser.parse_args()
    main(args)


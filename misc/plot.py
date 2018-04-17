
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import json
import os

def main(args):
    data = json.load(open(os.path.join(args.result_dir, 'log')))
    x, t, v = [], [], []
    for d in data:
        x.append(d[args.xlabel])
        t.append(d['main/{}'.format(args.scope)])
        v.append(d['validation/main/{}'.format(args.scope)])
    plt.plot(x, t, 'b-', label='train {}'.format(args.scope))
    plt.plot(x, v, 'r-', label='validation {}'.format(args.scope))
    plt.xlabel(args.xlabel)
    plt.ylabel(args.scope)
    plt.title(args.scope)
    if args.ylim:
        plt.ylim([args.ylim[0], args.ylim[1]])
    plt.legend()
    plt.savefig(os.path.join(args.result_dir, '{}.pdf'.format(args.scope)), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=str)
    parser.add_argument('--scope', '-s', type=str, default='loss')
    parser.add_argument('--ylim', '-y', type=float, nargs=2)
    parser.add_argument('--xlabel', '-x', type=str,
                        choices=['iteration', 'epoch'], default='epoch')
    args = parser.parse_args()
    main(args)

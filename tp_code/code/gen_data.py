#!/usr/bin/env python

import numpy as np
from itertools import product


def chunker(seq, size=None, number=None):
    if number:
        division = len(seq) / number
        return (seq[round(division * i):round(division * (i + 1))] for i in range(number))
    elif size:
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    else:
        raise ValueError


def func_a(x):
    assert len(x) == 4
    return (np.log(x[0] + x[1]) - np.sqrt(x[1] * x[2])) / np.sqrt(x[3])


def main_a():
    sp = np.linspace(1, 200, 10)
    l = []
    for x in product(sp, sp, sp, sp):
        l.append(f'{x[0]},{x[1]},{x[2]},{x[3]},{func_a(x)}\n')

    n_test = len(l)//5
    l_test = l[:n_test]
    for idx, a in enumerate(chunker(l_test, number=5)):
        with open(f'/tmp/func_a-test-{idx}.dat', mode='w') as f:
            f.writelines(a)

    l_train = l[n_test:]
    for idx, a in enumerate(chunker(l_train, number=5)):
        with open(f'/tmp/func_a-train-{idx}.dat', mode='w') as f:
            f.writelines(a)


if __name__ == '__main__':
    main_a()

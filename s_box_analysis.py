from numpy import (arange, array, asarray, column_stack, log2, ones, uint,
                   unique, unpackbits, zeros)
from numpy.typing import NDArray
from sympy import Matrix


def differential(s_box_: NDArray):
    s_box = s_box_.flat
    n = len(s_box)
    A = arange(n).repeat(n).reshape(n, n)
    difference_table = s_box[A ^ arange(n)] ^ s_box

    differential_cryptanalysis_table = zeros((n, n), dtype=uint)

    for diff in range(n):
        vlaues, counts = unique(difference_table[diff], return_counts=True)
        differential_cryptanalysis_table[diff][vlaues] = counts
    return differential_cryptanalysis_table


def linear(s_box_: NDArray):
    s_box = s_box_.flat
    n = len(s_box)

    linear_equations = arange(n**2, dtype=uint)
    linear_equations = linear_equations[
        linear_equations % n > 0
    ]  # filter out equations with no output bits
    linear_equations = linear_equations[
        linear_equations >= n
    ]  # filter out equations with no input bits

    plaintext_ciphertext_sequences = s_box + arange(n) * n

    approximations = (
        linear_equations.repeat(n).reshape(-1, n) & plaintext_ciphertext_sequences
    )

    validity = asarray(
        unpackbits(approximations.view("uint8"))
        .reshape(*approximations.shape, -1)
        .sum(axis=2)
        % 2,
        dtype=bool,
    )  # this does what the notebook does, in a more efficient way

    true_equations = n - validity.sum(axis=1)
    biases = abs(true_equations / n - 0.5)

    ordering = biases.argsort()[::-1]

    return linear_equations[ordering], true_equations[ordering]


def algebraic(s_box_: NDArray):
    s_box = s_box_.flat
    n = len(s_box)
    bit_count = int(log2(n))

    def x(i):
        return (arange(n, dtype=uint) & (ones(n, dtype=uint) << i)) >> i

    def y(i):
        return (s_box & (ones(n, dtype=uint) << i)) >> i

    # {1, x3, ... , xo, y3, ... , Yo, x3x2, x3x1, ... , x1xo, x3y3, x3y2, ... , xoyo, y3y2, y3y1, ... ,y1Yo}
    monomials = [
        ones(n, dtype=uint),  # 1
        *[x(i) for i in reversed(range(bit_count))],  # x3, ... , x0
        *[y(i) for i in reversed(range(bit_count))],  # y3, ... , y0
        *[
            x(i) * x(j) for i in reversed(range(bit_count)) for j in reversed(range(i))
        ],  # x3x2, x3x1, ... , x1x0
        *[
            x(i) * y(j)
            for i in reversed(range(bit_count))
            for j in reversed(range(bit_count))
        ],  # x3y3, x3y2, ... , x0y0
        *[
            y(i) * y(j) for i in reversed(range(bit_count)) for j in reversed(range(i))
        ],  # y3y2, y3y1, ... , y1y0
    ]
    space_matrix = column_stack(monomials)

    null_space = array(Matrix(space_matrix).nullspace())
    null_space /= min(abs(null_space)[null_space != 0])
    null_space %= 2
    null_space = null_space.reshape(-1, len(monomials))
    null_space

    return null_space

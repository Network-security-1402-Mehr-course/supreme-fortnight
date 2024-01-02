from numpy import arange, asarray, uint, unique, unpackbits, zeros
from numpy.typing import NDArray


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

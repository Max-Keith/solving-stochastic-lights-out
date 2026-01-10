import numpy as np

from lightsout.board import BoardState


def board_to_index_3x3(state: BoardState) -> int:
    """
    Encode a 3x3 board (BoardState) as an integer in [0, 512).
    We interpret the flattened bits as a little-endian binary number.
    """
    flat = state.to_flat().astype(np.uint8)  # length 9, values 0/1
    # idx = sum_j flat[j] * 2^j
    powers = 1 << np.arange(9, dtype=np.uint16)  # [1,2,4,...,256]
    return int(np.dot(flat, powers))


def array_to_index_3x3(bits: np.ndarray) -> int:
    """
    Same as above but for a flat (9,) 0/1 numpy array.
    """
    bits = bits.astype(np.uint8).reshape(-1)
    powers = 1 << np.arange(9, dtype=np.uint16)
    return int(np.dot(bits, powers))

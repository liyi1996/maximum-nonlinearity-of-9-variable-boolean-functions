#!/usr/bin/env python3
"""Self-contained exhaustive verifier for the two surviving cubic classes.
Source data: c_282 and c_290 and the 18-vector transversals in the source drafts.
Requires numpy.  No catalog and no result JSON are imported.
"""
from itertools import combinations
from collections import Counter
import numpy as np

N = 512
X = np.arange(N, dtype=np.int32)
pairs = [sum(1 << i for i in p) for p in combinations(range(9), 2)]
pair_to_index = {m: i for i, m in enumerate(pairs)}
quad = np.array([(X & m) == m for m in pairs], dtype=np.uint8)
faces = np.array([(X >> i) & 1 for i in range(9)], dtype=np.int32)
parity = np.array([x.bit_count() & 1 for x in range(N)], dtype=np.int32)

DATA = {
282: (
 [19,25,28,44,69,138,268,304,448],
 16777216,
 [36507222016,18253611008,268435456,134217728,33554432,8388608,
  2097152,1048576,524288,65536,16384,4096,4458496,536871168,128,64,16777232,4],
 (128640, 36960, 18816, 18144)),
290: (
 [19,22,26,28,37,67,76,131,134,137,259,268,304,448],
 17725652992,
 [2147483648,5368709120,268435456,671088640,33554432,12582912,
  1048576,2883584,25821184,65536,52076511232,16384,25174016,
  16781312,25167872,4294967808,128,524320],
 (126592, 43200, 34560, 8640)),
}

def rank(words):
    pivots = {}
    for word in words:
        while word:
            j = word.bit_length() - 1
            if j in pivots:
                word ^= pivots[j]
            else:
                pivots[j] = word
                break
    return len(pivots)

def dot(a, b):
    return (a & b).bit_count() & 1

def truth_quad(q):
    result = np.zeros(N, dtype=np.uint8)
    for j in range(36):
        if (q >> j) & 1:
            result ^= quad[j]
    return result

def fwht(batch):
    w = batch.astype(np.int32, copy=True)
    n = len(w)
    for j in range(9):
        half = 1 << j
        view = w.reshape(n, N // (2*half), 2, half)
        low = view[:, :, 0, :].copy()
        high = view[:, :, 1, :].copy()
        view[:, :, 0, :] = low + high
        view[:, :, 1, :] = low - high
    return w

def verify(index):
    masks, q0, basis, expected = DATA[index]
    c = np.zeros(N, dtype=np.uint8)
    for m in masks:
        assert m.bit_count() == 3
        c ^= ((X & m) == m).astype(np.uint8)
    base_weight = faces @ c.astype(np.int32)
    assert np.all(base_weight % 4 == 0)
    rhs = (base_weight // 4) % 2
    rows = [0] * 9
    for j in range(36):
        weight = faces @ (c ^ quad[j]).astype(np.int32)
        assert np.all(weight % 4 == 0)
        col = ((weight // 4) % 2) ^ rhs
        for i in range(9):
            rows[i] |= int(col[i]) << j
    translations = []
    for i in range(9):
        word = 0
        for m in masks:
            if m & (1 << i):
                word ^= 1 << pair_to_index[m ^ (1 << i)]
        translations.append(word)
    assert rank(rows) == rank(translations) == 9
    assert len(basis) == 18
    assert rank(translations + basis) == 27
    assert all(dot(r, q0) == int(b) for r,b in zip(rows, rhs))
    assert all(dot(r, w) == 0 for r in rows for w in translations+basis)
    init = c ^ truth_quad(q0)
    moves = [truth_quad(word) for word in basis]
    visited = odd_count = equality_count = 0
    peaks = Counter()
    for start in range(0, 1 << 18, 256):
        states = np.arange(start, start+256, dtype=np.uint32)
        batch = np.broadcast_to(init, (len(states), N)).copy()
        for j, m in enumerate(moves):
            selected = ((states >> j) & 1).astype(bool)
            batch[selected] ^= m
        visited += len(batch)
        assert np.all((batch.astype(np.int32) @ faces.T) % 8 == 0)
        good_wt = batch.sum(axis=1) % 16 == 8
        odd_count += int(good_wt.sum())
        if not np.any(good_wt):
            continue
        qspec = fwht(1-2*batch[good_wt].astype(np.int32))
        assert np.all(qspec % 32 == 16)
        equality = np.all((np.abs(qspec)==16) | (np.abs(qspec)==48), axis=1)
        qspec = qspec[equality]
        equality_count += len(qspec)
        if not len(qspec):
            continue
        assert np.all((np.abs(qspec)==16).sum(axis=1) == 448)
        forced_sign = np.where(qspec>0, -1, 1).astype(np.int32)
        f_walsh = fwht(forced_sign)
        peak = np.abs(f_walsh).max(axis=1)
        peaks.update(map(int, peak))
        # An independent, untransformed direct character-sum check
        witness = np.abs(f_walsh).argmax(axis=1).astype(np.int32)
        chars = 1-2*parity[witness[:,None] & X[None,:]]
        direct = np.abs((chars*forced_sign).sum(axis=1))
        assert np.array_equal(direct, peak)
        assert np.all(direct >= 40)
    assert visited == (1 << 18)
    result = (odd_count, equality_count, peaks[40], peaks[48])
    assert set(peaks) == {40,48}
    assert result == expected, (index, result, expected)
    print(f'class {index}: states={visited}, odd_parity={odd_count}, spectrum_A={equality_count}, forced_peaks={dict(peaks)}', flush=True)
    return result

def verify_lower_bound_witness():
    # Fixed 512-bit witness from the public 2026/2529 verification notebook.
    hex_string = (
        '68B7EF2DA03B0D3EA00DB6A96DD99AEAFDB9C842B6D5DC8C4526CE0DD29020DB'
        'B75FE3314568344E73688FF0CB2482E065231869E1AA4583765CC491F8A8DB12'
    )
    bits = np.array([int(bit) for digit in hex_string
                     for bit in f'{int(digit, 16):04b}'], dtype=np.int32)
    assert len(bits) == 512
    peak = int(np.abs(fwht((1-2*bits)[None, :])).max())
    assert peak == 28
    print('NL 242 witness: Walsh peak=28, nonlinearity=242', flush=True)

if __name__ == '__main__':
    verify_lower_bound_witness()
    for label in (282, 290):
        verify(label)
    print('PASS: both quotient enumerations, rank tests and direct rejecting sums.', flush=True)

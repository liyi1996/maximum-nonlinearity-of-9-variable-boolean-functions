"""Exact integer verification of the supplied 9-variable Boolean function.

Usage: python verify_nonlinearity.py [construction_242.json]
No third-party dependencies are required.
"""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path


def fwht(values):
    a = list(values)
    if not a or len(a) & (len(a) - 1):
        raise ValueError("Length must be a power of two")
    h = 1
    while h < len(a):
        for start in range(0, len(a), 2 * h):
            for j in range(start, start + h):
                u, v = a[j], a[j + h]
                a[j], a[j + h] = u + v, u - v
        h *= 2
    return a


def decode_hex(value):
    value = "".join(value.split())
    if len(value) != 128:
        raise ValueError("Expected exactly 128 hex digits (512 truth-table bits)")
    return [int(bit) for digit in value for bit in f"{int(digit, 16):04b}"]


def algebraic_degree(f):
    coefficients = list(f)
    for j in range((len(f) - 1).bit_length()):
        for x in range(len(f)):
            if x & (1 << j):
                coefficients[x] ^= coefficients[x ^ (1 << j)]
    return max((x.bit_count() for x, c in enumerate(coefficients) if c), default=0)


def verify(f):
    n = len(f)
    assert n == 512 and set(f) <= {0, 1}
    spectrum = fwht([1 - 2 * bit for bit in f])
    # Independent, direct Hamming-distance calculation over every affine function.
    best = n
    witness = None
    distances = []
    for a in range(n):
        d = sum(bit ^ ((a & x).bit_count() & 1) for x, bit in enumerate(f))
        for b, distance in ((0, d), (1, n - d)):
            distances.append(distance)
            if distance < best:
                best, witness = distance, {"a": a, "b": b}
        assert spectrum[a] == n - 2 * d
    assert sum(w * w for w in spectrum) == n * n
    assert fwht(spectrum) == [n * (1 - 2 * bit) for bit in f]
    assert best == n // 2 - max(map(abs, spectrum)) // 2
    autocorrelation_numerators = fwht([w * w for w in spectrum])
    assert all(a % n == 0 for a in autocorrelation_numerators)
    autocorrelation = [a // n for a in autocorrelation_numerators]
    return {
        "truth_table_length": n,
        "weight": sum(f),
        "algebraic_degree": algebraic_degree(f),
        "nonlinearity_fwht": n // 2 - max(map(abs, spectrum)) // 2,
        "nonlinearity_direct_1024_affine_functions": best,
        "closest_affine_witness": witness,
        "max_abs_walsh": max(map(abs, spectrum)),
        "signed_walsh_histogram": dict(sorted(Counter(spectrum).items())),
        "absolute_walsh_histogram": dict(sorted(Counter(map(abs, spectrum)).items())),
        "parseval_sum": sum(w * w for w in spectrum),
        "absolute_autocorrelation_indicator": max(map(abs, autocorrelation[1:])),
        "affine_distance_histogram": dict(sorted(Counter(distances).items())),
    }, spectrum


def local_obstruction(f, spectrum):
    """Exact combinatorial check, without assuming any input symmetry.

    For |W(a)|=28 let B_x contain a when sign(W(a))*(-1)^(f(x)+a.x)=-1.
    Improving NL to >=243 with three flips needs all three B_x pairwise
    disjoint. Four flips have even weight, so improvement needs NL>=244;
    this again requires all four B_x pairwise disjoint. If no disjoint pair
    exists and no B_x is empty, all distances 1 through 4 are excluded.
    At distance 5 or 6 each critical row can have at most two bad flips.
    Double-counting bad pairs can therefore also exclude those distances.
    The result concerns only the neighborhood of THIS truth table.
    """
    assert max(map(abs, spectrum)) == 28 and sum(f) % 2 == 0
    critical = [a for a, w in enumerate(spectrum) if abs(w) == 28]
    bad = []
    for x, bit in enumerate(f):
        mask = 0
        for i, a in enumerate(critical):
            c = (1 if spectrum[a] > 0 else -1) * (1 - 2 * (bit ^ ((a & x).bit_count() & 1)))
            if c < 0:
                mask |= 1 << i
        bad.append(mask)
    disjoint_pairs = 0
    minimum_pair_overlap = len(critical)
    for i, j in combinations(range(len(f)), 2):
        overlap = (bad[i] & bad[j]).bit_count()
        disjoint_pairs += overlap == 0
        minimum_pair_overlap = min(minimum_pair_overlap, overlap)
    empty_count = sum(mask == 0 for mask in bad)
    radius_four_excluded = empty_count == 0 and disjoint_pairs == 0
    excluded_radii = []
    count_bounds = {}
    for k in range(1, 7):
        # An odd number of flips gives odd weight and target max|W|<=26;
        # an even number gives even weight, so improvement needs max|W|<=24.
        target_max = 26 if k % 2 else 24
        max_bad_per_row = (target_max - 28 + 2 * k) // 4
        single_lower = k * min(map(int.bit_count, bad))
        single_upper = len(critical) * max_bad_per_row
        pair_lower = (k * (k - 1) // 2) * minimum_pair_overlap
        pair_upper = len(critical) * max_bad_per_row * (max_bad_per_row - 1) // 2
        excluded = single_lower > single_upper or pair_lower > pair_upper
        if excluded:
            excluded_radii.append(k)
        count_bounds[k] = {
            "max_bad_flips_per_critical_coefficient": max_bad_per_row,
            "single_incidence_lower": single_lower,
            "single_incidence_upper_if_feasible": single_upper,
            "pair_incidence_lower": pair_lower,
            "pair_incidence_upper_if_feasible": pair_upper,
            "excluded": excluded,
        }
    return {
        "target_nonlinearity": 243,
        "critical_coefficients_abs_28": len(critical),
        "minimum_bad_constraints_per_single_flip": min(map(int.bit_count, bad)),
        "empty_bad_sets": empty_count,
        "pairs_checked": len(f) * (len(f) - 1) // 2,
        "minimum_bad_set_pair_intersection": minimum_pair_overlap,
        "disjoint_bad_set_pairs": disjoint_pairs,
        "all_hamming_distances_1_through_4_excluded": radius_four_excluded,
        "excluded_hamming_distances": excluded_radii,
        "all_hamming_distances_1_through_6_excluded": excluded_radii == list(range(1, 7)),
        "double_counting_certificates": count_bounds,
        "is_global_optimality_proof": False,
    }


def self_check():
    assert fwht([1] * 512) == [512] + [0] * 511
    bent8 = [1 - 2 * (((x & 1) * ((x >> 1) & 1)) ^ (((x >> 2) & 1) * ((x >> 3) & 1)) ^ (((x >> 4) & 1) * ((x >> 5) & 1)) ^ (((x >> 6) & 1) * ((x >> 7) & 1))) for x in range(256)]
    assert set(map(abs, fwht(bent8))) == {16}
    assert max(map(abs, fwht(bent8 + bent8))) == 32


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("construction", nargs="?", default=str(Path(__file__).with_name("construction_242.json")))
    parser.add_argument("--output", default=str(Path(__file__).with_name("verification_results.json")))
    args = parser.parse_args()
    data = json.loads(Path(args.construction).read_text(encoding="utf-8-sig"))
    self_check()
    f = decode_hex(data["truth_table_hex"])
    result, spectrum = verify(f)
    result["local_obstruction"] = local_obstruction(f, spectrum)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

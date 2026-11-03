"""Independent exact audits of the cubic double-adjoint obstruction."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import random
import re

from verify_nonlinearity import decode_hex, fwht

BASE = Path(__file__).resolve().parent
TRIPLES = [m for m in range(512) if m.bit_count() == 3]


def adjoint(support):
    result = set()
    for a, b in combinations(sorted(support), 2):
        if not (a & b):
            c = 511 ^ (a | b)
            if c in result:
                result.remove(c)
            else:
                result.add(c)
    return result


def shear(support, i, j):
    """Top-form pullback under x_i -> x_i+x_j."""
    result = set(support)
    for m in support:
        if ((m >> i) & 1) and not ((m >> j) & 1):
            q = (m ^ (1 << i)) | (1 << j)
            if q in result:
                result.remove(q)
            else:
                result.add(q)
    return result


def anf(bits):
    coef = list(bits)
    for i in range(9):
        for x in range(512):
            if x & (1 << i):
                coef[x] ^= coef[x ^ (1 << i)]
    return coef


def evaluate(coef):
    return anf(coef)  # Boolean Moebius transform is self-inverse.


def perfect_matching_parity(support):
    return sum(a in adjoint(support) for a in support) % 2


def main():
    rng = random.Random(2026093027)
    random_weight_audits = 100
    for _ in range(random_weight_audits):
        coef = [rng.randrange(2) if m.bit_count() <= 3 else 0 for m in range(512)]
        top = {m for m in TRIPLES if coef[m]}
        dual = adjoint(top)
        h = evaluate(coef)
        for m in TRIPLES:
            weight = sum(h[x] for x in range(512) if (x & m) == m)
            assert weight % 2 == 0
            assert ((weight // 2) % 2) == (m in dual)

    covariance_audits = 1000
    for _ in range(covariance_audits):
        top = {m for m in TRIPLES if rng.randrange(2)}
        i, j = rng.sample(range(9), 2)
        # Adjoint is CONTRAVARIANT: transpose the elementary shear.
        assert adjoint(shear(top, i, j)) == shear(adjoint(top), j, i)
        assert adjoint(adjoint(shear(top, i, j))) == shear(adjoint(adjoint(top)), i, j)

    catalog = []
    text = (BASE / 'fresh_cubic9_forms.txt').read_text(encoding='utf8', errors='replace')
    for line in text.splitlines():
        match = re.match(r'\s*(\d+) h=.*?  (.*)', line)
        if not match:
            continue
        top = {sum(1 << (int(c)-1) for c in word)
               for word in re.findall(r'\+([1-9]{3})', match[2])}
        catalog.append({'index': int(match[1]),
                        'matching_parity': perfect_matching_parity(top),
                        'double_adjoint_fixed': adjoint(adjoint(top)) == top})
    assert len(catalog) == 349
    fixed0 = [v['index'] for v in catalog if v['double_adjoint_fixed'] and v['matching_parity'] == 0]
    fixed1 = [v['index'] for v in catalog if v['double_adjoint_fixed'] and v['matching_parity'] == 1]
    assert fixed0 == [282, 290, 349]
    assert fixed1 == [320, 326, 329]

    pool = json.loads((BASE / 'small_peak_obstructions_phase_pool.json').read_text())['phases']
    explicit_phase_audits = []
    for record in pool:
        h = decode_hex(record['phase_truth_table_hex'])
        hcoef = anf(h)
        top = {m for m in TRIPLES if hcoef[m]}
        q = fwht([1-2*b for b in h])
        assert Counter(map(abs, q)) == {16: 448, 48: 64}
        aligned = [int(w > 0) for w in q]  # minus the Walsh sign
        fcoef = anf(aligned)
        six_dual = {511 ^ m for m, b in enumerate(fcoef) if b and m.bit_count() == 6}
        assert six_dual == adjoint(adjoint(top))
        fwalsh = fwht([1-2*b for b in aligned])
        explicit_phase_audits.append({'phase_hex': record['phase_truth_table_hex'],
                                     'double_adjoint_equals_degree6_dual': True,
                                     'double_adjoint_equals_original_top': six_dual == top,
                                     'aligned_max_abs_walsh': max(map(abs, fwalsh))})

    # Audit the residue-eight L1 majorant over every possible scaled Walsh value.
    for q in range(-63, 64, 4):
        assert 16*abs(q) <= 3*q*q-2*q+15
    report = {'random_six_face_weight_formula_audits': random_weight_audits,
              'random_GL_contravariant_shear_audits': covariance_audits,
              'catalog_entries': len(catalog),
              'catalog_scope': 'Local 2003 representative file; external coverage must be independently sourced.',
              'matching_parity_zero_double_adjoint_fixed_indices': fixed0,
              'matching_parity_one_double_adjoint_fixed_indices': fixed1,
              'explicit_zero_free_equality_phase_audits': explicit_phase_audits,
              'residue8_L1_bound': 10048,
              'residue8_majorant_checked_all_possible_scaled_coefficients': True,
              'global_nonexistence_proof_by_this_script_alone': False}
    (BASE / 'audit_phase_double_adjoint.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'explicit_zero_free_equality_phase_audits'}))


if __name__ == '__main__':
    main()

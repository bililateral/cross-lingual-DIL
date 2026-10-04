from fractions import Fraction as F
import json

X = [[F(-1, 2), F(0), F(1, 2)], [F(1), F(1), F(1)]]
Y = [[F(-3, 10), F(-2, 5), F(7, 10)], [F(1), F(1), F(1)]]
def right_transpose_product(A, B):
    return [[sum(a * b for a, b in zip(row_a, row_b)) for row_b in B] for row_a in A]
XX = right_transpose_product(X, X)
YX = right_transpose_product(Y, X)
delta = [y - x for x, y in zip(X[0], Y[0])]
Xp_delta = [sum(x * d for x, d in zip(row, delta)) for row in X]
result = {
    'environment': 'web scratch pure Python exact rational arithmetic; not project Linux/BGE evidence',
    'X': [[str(v) for v in row] for row in X],
    'Y': [[str(v) for v in row] for row in Y],
    'XX_transpose': [[str(v) for v in row] for row in XX],
    'YX_transpose': [[str(v) for v in row] for row in YX],
    'delta_p': [str(v) for v in delta],
    'X_delta_p': [str(v) for v in Xp_delta],
    'old_order': sorted(range(3), key=lambda j: X[0][j]),
    'new_order': sorted(range(3), key=lambda j: Y[0][j]),
    'T_is_identity_for_any_positive_epsilon': XX == YX,
    'all_variable_features_inside_tanh_range': all(-1 < v < 1 for v in X[0] + Y[0]),
}
assert XX == YX
assert Xp_delta == [0, 0]
assert result['old_order'] != result['new_order']
print(json.dumps(result, ensure_ascii=False, indent=2))

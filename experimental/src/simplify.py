from datetime import datetime
from sympy import Symbol, Quaternion, simplify, Matrix, symbols

reference = [Symbol(k) for k in ["w", "x", "y", "z"]]

r = Quaternion(*reference)

r_mat = r.to_rotation_matrix()


print("Start")

start = datetime.now()

result = simplify(r_mat)

end = datetime.now()

print('Runtime', end - start)

reference = Matrix([symbols(['a', 'b', 'c']), symbols(['d', 'e', 'f']), symbols(['g', 'h', 'i'])])

r = Quaternion.from_rotation_matrix(reference)

r_mat = r.to_rotation_matrix()

print("Start 2")

start = datetime.now()

result = simplify(r_mat)

end = datetime.now()

print('Runtime 2', end - start)

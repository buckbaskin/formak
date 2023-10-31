from datetime import datetime
from sympy import Symbol, Quaternion, simplify, Matrix, symbols

reference = Matrix(
    [symbols(["a", "b", "c"]), symbols(["d", "e", "f"]), symbols(["g", "h", "i"])]
)

r = Quaternion.from_rotation_matrix(reference)

r_mat = r.to_rotation_matrix()

for i in range(3):
    for j in range(3):
        print(i, j, "\n", r_mat[i, j])

print("Start 2")

1 / 0

start = datetime.now()

result = simplify(r_mat)

end = datetime.now()

print("Runtime 2", end - start)

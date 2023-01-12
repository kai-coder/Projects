import numpy as np

testInput = np.array([2, 4])
testWeight = np.array([1, 3])
testBias = np.array([5])

ixw = np.dot(testInput, testWeight)

print(ixw)
print(np.add(ixw, testBias))

# (2 * 1 + 4 * 3) + 5
# f(x, w, b) = (x1 * w1 + x2 * w2) + b
# df_dx1 = dsum(x, w, b)_dx1_w1 * dx1_w1_x1
# dsum(x, w, b)_dx1_w1 = 1
# dx1_w1_x1 = w1
# df_dx1 = 1 * w1
# df_dx1 = w1

# df_dx2 = w2
# df_dw1 = x1
# df_dw2 = x2
# df_db = 1

df_dx = np.array([testWeight[0], testWeight[1]])
df_dw = np.array([testInput[0], testInput[1]])
df_db = 1

testInput = testInput - df_dx
testWeight = testWeight - df_dw
testBias = testBias - df_db
ixw = np.dot(testInput, testWeight)

print(ixw)
print(np.add(ixw, testBias))


import numpy as np

np.random.seed(3)


batchNum = 2
dataSize = 2
testInput = np.random.randint(1, 5, (batchNum, dataSize))

neuronNum = 3
testWeight = np.random.randint(1, 5, (neuronNum, dataSize))
testBias = np.random.randint(1, 5, (1, neuronNum))
output = np.dot(testInput, testWeight.T) + testBias
print("[", end="")
for i, index in zip(testInput, range(len(testInput))):
    if index != 0:
        print(end=" ")
    print("[", end="")
    for w, b, index2 in zip(testWeight, testBias[0], range(len(testWeight))):
        print("(", end="")
        for z, c, index3 in zip(i, w, range(len(i))):
            print(f"{z}*{c}", end="")
            if index3 == len(i) - 1:
                print(")", end="")
                print(f" + {b}", end="")
                if index2 != len(testWeight) - 1:
                    print(", ", end="")
                elif index != len(testInput) - 1:
                    print("], ")
                else:
                    print("]", end="")
            else:
                print(" + ", end="")
print("]")
print()
print("[", end="")
for i in range(len(testInput)):
    if i != 0:
        print(end=" ")
    print("[", end="")
    for w, b in zip(range(len(testWeight)), range(len(testBias[0]))):
        print("(", end="")
        for z, c in zip(range(len(testInput[i])), range(len(testWeight[w]))):
            print(f"x{i}{z}*w{w}{c}", end="")
            if z == len(testInput[i]) - 1:
                print(")", end="")
                print(f" + b{b}", end="")
                if w != len(testWeight) - 1:
                    print(", ", end="")
                elif i != len(testInput) - 1:
                    print("], ")
                else:
                    print("]", end="")
            else:
                print(" + ", end="")
print("]")
print()
print(f"f(x, w, b) = Σxw + b")
print("[", end="")
for i in range(len(testInput)):
    if i != 0:
        print(end=" ")
    print("[", end="")
    for w, b in zip(range(len(testWeight)), range(len(testBias[0]))):
        print(f"f(x{i}, w{w}, b{b})", end="")
        if w != len(testWeight) - 1:
            print(", ", end="")
        elif i != len(testInput) - 1:
            print("], ")
        else:
            print("]", end="")
print("]")
print()
print("dfx_dx:")
print("[", end="")
for i in range(len(testInput)):
    if i != 0:
        print(end=" ")
    print("[", end="")
    for z in range(len(testInput[i])):
        print(f"df_dx{i}{z}", end="")
        if z != len(testInput[i]) - 1:
            print(", ", end="")
        elif i != len(testInput) - 1:
            print("], ")
        else:
            print("]", end="")
print("]")
print()
print("[", end="")
for i in range(len(testInput)):
    if i != 0:
        print(end=" ")
    for c in range(len(testInput[i])):
        print("[", end="")
        for w in range(len(testWeight)):
            print(f"w{w}{c}", end="")
            if w == len(testWeight) - 1:
                if c != len(testInput[i]) - 1:
                    print("], ", end="")
                elif i != len(testInput) - 1:
                    print("], ")
                else:
                    print("]", end="")
            else:
                print(", ", end="")
print("]")
print()
print("[", end="")
for w, b in zip(range(len(testWeight)), range(len(testBias[0]))):
    if w != 0:
        print(end=" ")
    print("[", end="")
    for i in range(len(testInput)):
        print(f"df_dw{w}{i}", end="")
        if i != len(testInput) - 1:
            print(", ", end="")
        elif w != len(testWeight) - 1:
            print("], ")
        else:
            print("]", end="")
print("]")
print()
print("[", end="")
for i in range(len(testInput)):
    if i != 0:
        print(end=" ")
    print("[", end="")
    for w in range(len(testWeight)):
        for z, c in zip(range(len(testInput[i])), range(len(testWeight[w]))):
            print(f"w{c}{w}", end="")
            if z == len(testInput[i]) - 1:
                if w != len(testWeight) - 1:
                    print(", ", end="")
                elif i != len(testInput) - 1:
                    print("], ")
                else:
                    print("]", end="")
            else:
                print(" + ", end="")
print("]")

# print()
# print("Input:", testInput)
# print("Weight:", testWeight)
# print("Bias:", testBias)
#
# print("Test 1:")
# print(output)

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

# df_dx = np.array([testWeight[0], testWeight[1]])
# df_dw = np.array([testInput[0], testInput[1]])
# df_db = 1
#
# testInput = testInput - df_dx
# testWeight = testWeight - df_dw
# testBias = testBias - df_db
# ixw = np.dot(testInput, testWeight)
# print("Test 2:")
# print(ixw)
# print(np.add(ixw, testBias))


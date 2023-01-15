import numpy as np

np.random.seed(17)
def printLayerCalculations(inputs, weights, biases):
    print("[", end="")
    for index in range(len(weights.T)):
        print("(", end="")
        for index2 in range(len(inputs)):
            print(f"{inputs[index2]}*{weights[index2][index]}", end="")
            if index2 == len(inputs) - 1:
                print(")", end="")
                print(f" + {biases[index]}", end="")
                if index != len(weights.T) - 1:
                    print(", ", end="")
            else:
                print(" + ", end="")
    print("]")
    print("[", end="")
    for index in range(len(weights.T)):
        print("(", end="")
        for index2 in range(len(inputs)):
            print(f"{inputs[index2]*weights[index2][index]}", end="")
            if index2 == len(inputs) - 1:
                print(")", end="")
                print(f" + {biases[index]}", end="")
                if index != len(weights.T) - 1:
                    print(", ", end="")
            else:
                print(" + ", end="")
    print("]")
    print("[", end="")
    output = np.dot(inputs, weights) + biases
    for i, index in zip(output, range(len(output))):
        print(i, end="")
        if index != len(output) - 1:
            print(", ", end="")
    print("]")
    return output

def ReLUCalculations(input):
    print("[", end="")
    for i, index in zip(input, range(len(input))):
        print(f"max(0, {i})", end="")
        if index != len(input) - 1:
            print(", ", end="")
    print("]")
    return np.maximum(0, input)

def softMaxCalculations(input):
    print("[", end="")
    for i, index in zip(input, range(len(input))):
        print(f"(e^{i}) / ", end="")
        print(f"(", end="")
        for j, index2 in zip(input, range(len(input))):
            print(f"e^{j}", end="")
            if index2 != len(input) - 1:
                print(" + ", end="")
        print(f")", end="")
        if index != len(input) - 1:
            print(", ", end="")
    print("]")
    return np.maximum(0, input)

def printLayerOutputs(output):
    print("[", end="")
    for i, index in zip(output, range(len(output))):
        print(i, end="")
        if index != len(output) - 1:
            print(", ", end="")
    print("]")

dataSize = 2
testInput = np.random.randint(-5, 5, (dataSize))

neuronNum = 3
neuronNum2 = 3
neuronNum3 = 2
testWeight1 = np.random.randint(-5, 5, (neuronNum, dataSize)).T
testBias1 = np.random.randint(-5, 5, (neuronNum))
testWeight2 = np.random.randint(-5, 5, (neuronNum2, neuronNum)).T
testBias2 = np.random.randint(-5, 5, (neuronNum2))
testWeight3 = np.random.randint(-5, 5, (neuronNum3, neuronNum2)).T
testBias3 = np.random.randint(-5, 5, (neuronNum3))
output = np.dot(testInput, testWeight1) + testBias1
print(f"Input Layer ({dataSize} neurons):")
print(testInput)
print()
print(f"First Neural Layer ({neuronNum} neurons):")
print(testWeight1)
print(testBias1)
print()
print(f"Second Neural Layer ({neuronNum2} neurons):")
print(testWeight2)
print(testBias2)
print()
print(f"Output Layer ({neuronNum3} neurons):")
print(testWeight3)
print(testBias3)
print()
print("First Neural Layer Forward Propagation:")
output = printLayerCalculations(testInput, testWeight1, testBias1)
print()
print("First Neural Layer Activation Function:")
output = ReLUCalculations(output)
print("∀j(")
print("   max(")
print("       0,")
print("       Σi(")
print("          xi*w1ij")
print("   )")
print("   + b1k")
print(")")
print()
print("First Neural Layer Output:")
printLayerOutputs(output)
print()
print("Second Neural Layer Forward Propagation:")
output2 = printLayerCalculations(output, testWeight2, testBias2)
print()
print("Second Neural Layer Activation Function:")
output2 = ReLUCalculations(output2)
print("∀j(")
print("   max(")
print("       0,")
print("       Σi(")
print("          ∀j(")
print("             max(")
print("                 0,")
print("                 Σi(")
print("                    xi*w1ij")
print("                 )")
print("                 + b1j")
print("             )")
print("          )i*w2ij")
print("       )")
print("       + b2j")
print("   )")
print(")")
print()
print("Second Neural Layer Output:")
printLayerOutputs(output2)
print()
print("Output Layer Forward Propagation:")
output3 = printLayerCalculations(output2, testWeight3, testBias3)
print()
print("Output Layer Activation Function:")
output3 = softMaxCalculations(output3)
print("∀j(")
print("   e^(")
print("      Σi(")
print("         ∀j(")
print("            max(")
print("                0,")
print("                Σi(")
print("                   ∀j(")
print("                      max(")
print("                          0,")
print("                          Σi(")
print("                             xi*w1ij")
print("                          )")
print("                          + b1j")
print("                      )")
print("                   )i*w2ij")
print("                )")
print("                + b2j")
print("             )")
print("         )")
print("      )i*w3ij + b3j")
print("   )")
print("   /")
print("   Σk(")
print("      e^(")
print("         Σi(")
print("            ∀j(")
print("               max(")
print("                   0,")
print("                   Σi(")
print("                      ∀j(")
print("                         max(")
print("                             0,")
print("                             Σi(")
print("                                xi*w1ij")
print("                             )")
print("                             + b1j")
print("                         )")
print("                      )i*w2ij")
print("                   )")
print("                   + b2j")
print("                )")
print("            )")
print("         )i*w3ik + b3k")
print("      )")
print("   )")
print(")")
print()
print("Output Layer Output:")
printLayerOutputs(output3)
print()
print("[", end="")
for index in range(len(testWeight1.T)):
    print("(", end="")
    for index2 in range(len(testInput)):
        print(f"x{index2}*w{index2}{index}", end="")
        if index2 == len(testInput) - 1:
            print(")", end="")
            print(f" + b{index}", end="")
            if index != len(testWeight1.T) - 1:
                print(", ", end="")
        else:
            print(" + ", end="")
print("]")
print("[", end="")
for index in range(len(testWeight1.T)):
    print(f"Σi(xi*wi{index})", end="")
    if index2 == len(testInput) - 1:
        print(f" + b{index}", end="")
        if index != len(testWeight1.T) - 1:
            print(", ", end="")
    else:
        print(" + ", end="")
print("]")
print(f"∀j(Σi(xi*wij) + bj)")
print()
print(f"f(x, w, b) = Σxw + b")
print("[", end="")
for i in range(len(testInput)):
    if i != 0:
        print(end=" ")
    print("[", end="")
    for w, b in zip(range(len(testWeight1)), range(len(testBias1[0]))):
        print(f"f(x{i}, w:{w}, b{b})", end="")
        if w != len(testWeight1) - 1:
            print(", ", end="")
        elif i != len(testInput) - 1:
            print("], ")
        else:
            print("]", end="")
print("]")
print()
print("df_dxij = df_d(xij*wjk) * d(xij*wjk)_dxij")
print("df_d(xij*wjk) = 1")
print("d(xij*wjk)_dxij = wjk")
print("df_dxij = 1 * wjk")
print()
print("df_dxij = wjk")
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
    print("[", end="")
    for c in range(len(testInput[i])):
        print("[", end="")
        for w in range(len(testWeight1)):
            print(f"w{c}{w}", end="")
            if w == len(testWeight1) - 1:
                if c != len(testInput[i]) - 1:
                    print("], ", end="")
                elif i != len(testInput) - 1:
                    print("]], ")
                else:
                    print("]", end="")
            else:
                print(", ", end="")
print("]]")
print()
print("[", end="")
for w, b in zip(range(len(testWeight1)), range(len(testBias1[0]))):
    if w != 0:
        print(end=" ")
    print("[", end="")
    for i in range(len(testInput)):
        print(f"df_dw{w}{i}", end="")
        if i != len(testInput) - 1:
            print(", ", end="")
        elif w != len(testWeight1) - 1:
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
    for w in range(len(testWeight1)):
        for z, c in zip(range(len(testInput[i])), range(len(testWeight1[w]))):
            print(f"w{c}{w}", end="")
            if z == len(testInput[i]) - 1:
                if w != len(testWeight1) - 1:
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


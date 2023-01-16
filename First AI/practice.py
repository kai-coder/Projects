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

def printLayerCalculations2(inputs, weights):
    print("[", end="")
    for index in range(len(weights.T)):
        print("(", end="")
        for index2 in range(len(inputs)):
            print(f"x{index2}*w{index2}{index}", end="")
            if index2 == len(inputs) - 1:
                print(")", end="")
                print(f" + b{index}", end="")
                if index != len(weights.T) - 1:
                    print(", ", end="")
            else:
                print(" + ", end="")
    print("]")
    print("[", end="")
    for index in range(len(weights.T)):
        print(f"Σi(xi*wi{index})", end="")
        if index2 == len(inputs) - 1:
            print(f" + b{index}", end="")
            if index != len(weights.T) - 1:
                print(", ", end="")
        else:
            print(" + ", end="")
    print("]")
    print(f"∀j(Σi(xi*wij) + bj)")

def ReLUCalculations(input):
    print("[", end="")
    for i, index in zip(input, range(len(input))):
        print(f"max(0, {i})", end="")
        if index != len(input) - 1:
            print(", ", end="")
    print("]")
    return np.maximum(0, input)

def ReLUCalculations2():
    print(f"∀j(max(0, Σi(xi*wij) + bj))")

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
    expVal = np.exp(input)
    return expVal / np.sum(expVal)

def lossCalculation(input, expectedValue):
    for i, v, index in zip(input, expectedValue, range(len(input))):
        print(f"{v} * log({i})", end="")
        if index != len(input):
            print(f" + ", end="")
    return -np.dot(np.log(input), expectedValue)

def softMaxCalculations2():
    print("Dividend = exponentiate e to the neuron value:")
    print("∀j(e^(Σi(xi*wij) + bj))")
    print("Divisor = sum of all exponentiated neuron values(instead of ∀ it has Σ):")
    print("Σk(e^(Σi(xi*wik) + bk))")
    print()
    print("∀j(e^(Σi(xi*wij) + bj) / Σk(e^(Σi(xi*wik) + bk)))")

def printLayerOutputs(output):
    print("[", end="")
    for i, index in zip(output, range(len(output))):
        print(i, end="")
        if index != len(output) - 1:
            print(", ", end="")
    print("]")

dataSize = 3
testInput = np.random.randint(-5, 5, (dataSize))

neuronNum = 4
neuronNum2 = 4
neuronNum3 = 2
testWeight1 = np.random.randint(-5, 5, (neuronNum, dataSize)).T
testBias1 = np.random.randint(-5, 5, (neuronNum))
testWeight2 = np.random.randint(-5, 5, (neuronNum2, neuronNum)).T
testBias2 = np.random.randint(-5, 5, (neuronNum2))
testWeight3 = np.random.randint(-5, 5, (neuronNum3, neuronNum2)).T
testBias3 = np.random.randint(-5, 5, (neuronNum3))
output = np.dot(testInput, testWeight1) + testBias1
expectedOutput = np.eye(neuronNum3)[np.random.randint(0, neuronNum3, (1))[0]]
print("Expected Output:")
print(expectedOutput)
print()
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
print()
print("FIRST NEURAL LAYER:")
print("Forward Propagation:")
output = printLayerCalculations(testInput, testWeight1, testBias1)
print()
print("Activation Function:")
output = ReLUCalculations(output)
print()
print("Output:")
printLayerOutputs(output)
print()
print("Forward Propagation Equation:")
printLayerCalculations2(testInput, testWeight1)
print()
print("Forward Propagation + Activation Equation:")
ReLUCalculations2()
print()
print()
print("SECOND NEURAL LAYER:")
print("Forward Propagation:")
output2 = printLayerCalculations(output, testWeight2, testBias2)
print()
print("Activation Function:")
output2 = ReLUCalculations(output2)
print("Output:")
printLayerOutputs(output2)
print()
print("x = last layer output: ", end="")
ReLUCalculations2()
print("Forward Propagation Equation:")
printLayerCalculations2(testInput, testWeight1)
print()
print("Forward Propagation + Activation Equation:")
ReLUCalculations2()
print()
print("Full equation:")
print("∀j(max(0, Σi(∀j(max(0, Σi(xi*wij) + bj))i*wij) + bj))")
print()
print()
print("OUTPUT LAYER:")
print("Forward Propagation:")
output3 = printLayerCalculations(output2, testWeight3, testBias3)
print()
print("Activation Function:")
output3 = softMaxCalculations(output3)
print()
print("Output:")
printLayerOutputs(output3)
print()
print("x = last layer output: ", end="")
print("∀j(max(0, Σi(∀j(max(0, Σi(xi*wij) + bj))i*wij) + bj))")
print("Forward Propagation Equation:")
printLayerCalculations2(testInput, testWeight1)
print()
print("Forward Propagation + Activation Equation:")
softMaxCalculations2()
print()
print("Full equation:")
print("∀j(e^(Σi(∀j(max(0, Σi(∀j(max(0, Σi(xi*wij) + bj))i*wij) + bj))i*wij) + bj) / Σk((e^Σi(∀j(max(0, Σi(∀j(max(0, Σi(xi*wij) + bj))i*wij) + bj))i*wik) + bk))))")
print()
print("Equation with clearer variable names:")
print("∀j(e^(Σi(∀j(max(0, Σi(∀j(max(0, Σi(xi*w1ij) + b1j))i*w2ij) + b2j))i*w3ij) + b3j) / Σk((e^Σi(∀j(max(0, Σi(∀j(max(0, Σi(xi*w1ij) + b1j))i*w2ij) + b2j))i*w3ik) + b3k))))")
print()
print()
print("LOSS:")
print("Loss Calculation:")
output4 = lossCalculation(output3, expectedOutput)
print()
print("Output:")
print(output4)
print()
print("Loss Equation:")
print("x=∀j(e^(Σi(∀j(max(0, Σi(∀j(max(0, Σi(xi*w1ij) + b1j))i*w2ij) + b2j))i*w3ij) + b3j) / Σk((e^Σi(∀j(max(0, Σi(∀j(max(0, Σi(xi*w1ij) + b1j))i*w2ij) + b2j))i*w3ik) + b3k))))")
print("-Σl(yl*log(xl))")
print()
print("Full equation:")
print("-Σl(yl*log(∀j(e^(Σi(∀j(max(0, Σi(∀j(max(0, Σi(xi*w1ij) + b1j))i*w2ij) + b2j))i*w3ij) + b3j) / Σk((e^Σi(∀j(max(0, Σi(∀j(max(0, Σi(xi*w1ij) + b1j))i*w2ij) + b2j))i*w3ik) + b3k))))l))")
print()
print()
print("WEIGHT GRADIENTS:")
print("d(-Σl)_d(w1ij) = d(-Σl)_d(yl*log(xl)) * d(yl*log(xl))_d(log(xl)) * d(log(xl))_d(xl)")
print("d(-Σl)_d(w1ij) =          -1          *    {yl = 1: 1; yl=0: 0}  *     1 / xl")
print("d(-Σl)_d(w1ij) = -{yl = 1: 1; yl=0: 0} / xl")
print("d(-Σl)_d(w1ij) = d(e^(Σi)/Σk(e^(Σi)))_d(Σi)")
print("d(-Σl)_d(w1ij) = (d(e^(Σi))_d(Σi) * Σk(e^(Σi)) - e^(Σi) * d(Σk(e^(Σi)))_d(Σi)) / (Σk(e^(Σi)))^2")
print("d(-Σl)_d(w1ij) = (e^(Σi) * Σk(e^(Σi)) - e^(Σi) * d(Σk(e^(Σi)))_d(Σi)) / (Σk(e^(Σi)))^2")
print("d(Σk(e^(Σi)))_d(Σi) = d(Σk(e^(Σi)))_d(e^(Σi)) * d(e^(Σi))_d(Σi)")
print("d(Σk(e^(Σi)))_d(Σi) = 1 * e^(Σi)")
print("d(-Σl)_d(w1ij) = (e^(Σi) * Σk(e^(Σi)) - e^(Σi) * e^(Σi) / (Σk(e^(Σi)))^2")
print("d(-Σl)_d(w1ij) = -{yl = 1: 1; yl=0: 0} / xl")
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


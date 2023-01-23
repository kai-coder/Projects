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
        if index != len(input) - 1:
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
maxoutput = ReLUCalculations(output)
print()
print("Output:")
printLayerOutputs(maxoutput)
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
output2 = printLayerCalculations(maxoutput, testWeight2, testBias2)
print()
print("Activation Function:")
maxoutput2 = ReLUCalculations(output2)
print("Output:")
printLayerOutputs(maxoutput2)
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
output3 = printLayerCalculations(maxoutput2, testWeight3, testBias3)
print()
print("Activation Function:")
softoutput3 = softMaxCalculations(output3)
print()
print("Output:")
printLayerOutputs(softoutput3)
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
output4 = lossCalculation(softoutput3, expectedOutput)
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
print("d(-Σl)_d(w3ij) = d(-Σl)_d(xl) * d(∀j(e^(xj)/Σk))_d(xz) * d(Σi(xi*w3ij))_d(w3ij)")
print()
print()
print(softoutput3)
print()
print("d(-Σl)_d(xl) = d(-Σl)_d(yl*log(xl)) * d(yl*log(xl))_d(log(xl)) * d(log(xl))_d(xl)")
print("d(-Σl)_d(xl) =          -1          *  {yl = 1: 1; yl = 0: 0}  *     1 / xl")
print("d(-Σl)_d(xl) = -{yl = 1: 1; yl = 0: 0} / xl")
print("d(-Σl)_d(xl) = -yl / xl")
print()
print("∀j(-yj / xj)")
print("[", end="")
for i in range(len(output3)):
    print(f"d(-Σl)_d(x{i})", end="")
    if i != len(output3) - 1:
        print(", ", end="")
print("]")
print(f"GRADIENT OF SOFTMAX OUTPUTS: {np.divide(-expectedOutput, softoutput3)}")
print()
print()
print(output3)
print()
print("d(∀j(e^(xj)/Σk))_d(xz) = ∀j(d(e^(xj)/Σk)_d(xz))")
print("d(∀j(e^(xj)/Σk))_d(xz) = ∀j(       (d(e^(xj))_d(xz)         * Σk - e^(xj) * d(Σk)_d(xz)) / Σk^2)")
print("d(∀j(e^(xj)/Σk))_d(xz) = ∀j({xj = xz: e^(xj); xj != xz: 0}  * Σk - e^(xj) *   e^(xz)   ) / Σk^2)")
print("d(∀j(e^(xj)/Σk))_d(xz) = ∀j({xj = xz: (e^(xj) * Σk - e^(xj) * e^(xz)) / Σk^2     ; xj != xz: -e^(xj) * e^(xz) / Σk^2})")
print("d(∀j(e^(xj)/Σk))_d(xz) = ∀j({xj = xz:       e^(xj)(Σk - e^(xz)) / Σk^2           ; xj != xz: -e^(xj) * e^(xz) / Σk^2})")
print("d(∀j(e^(xj)/Σk))_d(xz) = ∀j({xj = xz:     e^(xj) / Σk * (Σk - e^(xz)) / Σk       ; xj != xz: -e^(xj) / Σk * e^(xz) / Σk})")
print("d(∀j(e^(xj)/Σk))_d(xz) = ∀j({xj = xz:   e^(xj) / Σk * (Σk / Σk - e^(xz) / Σk)    ; xj != xz: -e^(xj) / Σk * e^(xz) / Σk})")
print("d(∀j(e^(xj)/Σk))_d(xz) = ∀j({xj = xz:         soft(xj) * (1 - soft(xz))          ; xj != xz: -soft(xj) * soft(xz})")
print("d(∀j(e^(xj)/Σk))_d(xz) = ∀j({xj = xz:         soft(xj) * (1 - soft(xz))          ; xj != xz: soft(xj) * (0 - soft(xz)))")
print("d(∀j(e^(xj)/Σk))_d(xz) = ∀j({xj = xz: soft(xj); xj != xz: 0} - soft(xj) * soft(xz))")
print()
print("∀z(∀j({xj = xz: soft(xj); xj != xz: 0} - soft(xj) * soft(xz)))")
print("[", end="")
for i in range(len(softoutput3)):
    if i != 0:
        print(" ", end="")
    print("[", end="")
    for j in range(len(softoutput3)):
        print(f"d(e^(x{j})/Σk))_d(x{i})", end="")
        if j == len(output3) - 1:
            if i != len(softoutput3) - 1:
                print("], ")
            else:
                print("]", end="")
        else:
            print(", ", end="")
print("]")
print("Sum along 0 axis see total effect of xn on the array")
print("Multiply by previous gradient before summing")
print(f"GRADIENT OF THIRD LAYER OUTPUTS: {np.dot(np.diagflat(softoutput3) - np.dot(softoutput3.reshape(-1, 1), softoutput3.reshape(-1, 1).T), np.divide(-expectedOutput, softoutput3))}")
print()
print()
print("Σj(-yj / xj * ({xj = xz: soft(xj); xj != xz: 0} - soft(xj) * soft(xz)))")
print("Σj(-yj / xj * ({xj = xz: xj; xj != xz: 0} - xj * xz))")
print("(-yz / xz * (xz - xz * xz)) + Σj!=z(-yj / xj * -xj * xz)")
print("(-yz + yz * xz)) + Σj!=z(yj * xz)")
print("-yz + Σj(yj * xz)")
print("yj is 1 at only one spot so:")
print("-yz + xz")
print("xz - yz")
print(f"OPTIMIZED GRADIENT OF THIRD LAYER OUTPUTS: {np.subtract(softoutput3, expectedOutput)}")
print()
print()
print("d(Σi(xi*w3ij) + b3j)_d(w3ij) = d(Σi(xi*w3ij) + b3j)_d(Σi(xi*w3ij)) * d(Σi(xi*w3ij))_d(xi*w3ij) * d(xi*w3ij)_d(w3ij)")
print("d(Σi(xi*w3ij) + b3j)_d(w3ij) =                   1                 *               1           *        xi")
print("d(Σi(xi*w3ij) + b3j)_d(w3ij) = xi")
print("Technically ∀j is being used but w3ij has no appearance there so it just returns a 0 and has no effect in the sum")
print()
print("∀i(∀j(xi))")
print("[", end="")
for i in range(testWeight3.shape[0]):
    if i != 0:
        print(" ", end="")
    print("[", end="")
    for j in range(testWeight3.shape[1]):
        print(f"d(Σi(xi*w3i{j}) + b3{j})_d(w3{i}{j})", end="")
        if j == testWeight3.shape[1] - 1:
            if i != testWeight3.shape[0] - 1:
                print("], ")
            else:
                print("]", end="")
        else:
            print(", ", end="")
print("]")
thirdLayerGrad = np.dot(np.diagflat(softoutput3) - np.dot(softoutput3.reshape(-1, 1), softoutput3.reshape(-1, 1).T), np.divide(-expectedOutput, softoutput3))
print(f"GRADIENT OF THIRD LAYER WEIGHTS: {np.dot(output2.reshape(-1, 1), thirdLayerGrad.reshape(-1, 1).T)}")
print(f"GRADIENT OF THIRD LAYER BIASES: {thirdLayerGrad}")
print()
print()
print("d(Σi(xi*w3ij) + b3j)_d(xi) = d(Σi(xi*w3ij) + b3j)_d(Σi(xi*w3ij)) * d(Σi(xi*w3ij))_d(xi*w3ij) * d(xi*w3ij)_d(w3ij)")
print("d(Σi(xi*w3ij) + b3j)_d(xi) =                   1                 *               1           *        w3ij")
print("d(Σi(xi*w3ij) + b3j)_d(xi) = ∀j(w3ij)")
print()
print("∀i(∀j(w3ij))")
print("[", end="")
for i in range(len(output2)):
    if i != 0:
        print(" ", end="")
    print("[", end="")
    for j in range(testWeight3.shape[1]):
        print(f"d(Σi(xi*w3i{j}) + b3{j})_d(x{i})", end="")
        if j == testWeight3.shape[1] - 1:
            if i != testWeight3.shape[0] - 1:
                print("], ")
            else:
                print("]", end="")
        else:
            print(", ", end="")
print("]")
print(f"GRADIENT OF SECOND LAYER OUTPUTS: {np.dot(testWeight3, thirdLayerGrad)}")
secondLayerGrad = np.dot(testWeight3, thirdLayerGrad)
print()
print("Same two more times for second layer weights and biases and first layer weights and biases")
print()
print("Now cover batches")
print("Full equation for batches:")
print("Σz(-Σl(yzl*log(∀z(∀j(e^(Σi(∀z(∀j(max(0, Σi(∀z(∀j(max(0, Σi(xzi*w1ij) + b1j)))zi*w2ij) + b2j)))zi*w3ij) + b3j) / Σk(e^(Σi(∀z(∀j(max(0, Σi(∀z(∀j(max(0, Σi(xzi*w1ij) + b1j)))zi*w2ij) + b2j)))zi*w3ik) + b3k)))))zl)) / num")
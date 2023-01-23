import numpy as np

np.random.seed(17)
dataSize = 3
testInput = np.random.randint(-5, 5, (dataSize))

neuronNum = 4
neuronNum2 = 4
neuronNum3 = 2
testWeight1 =  np.random.randn(neuronNum, dataSize).T * 0.01
testBias1 = np.zeros((neuronNum))
testWeight2 = np.random.randn(neuronNum2, neuronNum).T * 0.01
testBias2 = np.zeros((neuronNum2))
testWeight3 = np.random.randn(neuronNum3, neuronNum2).T * 0.01
testBias3 = np.zeros((neuronNum3))
expectedOutput = np.eye(neuronNum3)[np.random.randint(0, neuronNum3, (1))[0]]

for i in range(1000):
    n1 = np.maximum(0, np.add(np.dot(testInput, testWeight1), testBias1))
    n2 = np.maximum(0, np.add(np.dot(n1, testWeight2), testBias2))
    n3temp = np.add(np.dot(n2, testWeight3), testBias3) - np.max(np.add(np.dot(n2, testWeight3), testBias3))
    n3 = np.exp(n3temp) / np.sum(np.exp(n3temp))
    n3 = np.clip(n3, 1e-7, 1 - 1e-7)
    out = -np.dot(expectedOutput, np.log(n3))

    n3grad = np.subtract(n3, expectedOutput)
    testWeight3 = testWeight3 - np.dot(n2.reshape(-1, 1), n3grad.reshape(-1, 1).T)
    testBias3 = testBias3 - n3grad

    n2grad = np.dot(testWeight3, n3grad)
    testWeight2 = testWeight2 - np.dot(n1.reshape(-1, 1), n2grad.reshape(-1, 1).T)
    testBias2 = testBias2 - n2grad

    n1grad = np.dot(testWeight2, n2grad)
    testWeight1 = testWeight1 - np.dot(testInput.reshape(-1, 1), n1grad.reshape(-1, 1).T)
    testBias1 = testBias1 - n1grad
    print(out)
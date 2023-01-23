import numpy as np
inputs = np.array([1, -2, 3])
weights = np.array([-3, -1, 2]).T
bias = 3
xw0 = inputs[0] * weights[0]
xw1 = inputs[1] * weights[1]
xw2 = inputs[2] * weights[2]
sum = xw0 + xw1 + xw2 + bias
max2 = max(0, sum)
print(max2)

# max(0, sum(inputs[0] * weights[0], inputs[1] * weights[1], inputs[2] * weights[2])
# dmax_dx0 = dmax_dsum * dsum_dxw0  * dxw0_dx0
# dmax_dw0 = dmax_dsum * dsum_dxw0  * dxw0_dw0
# max = {x    x > 0
#        0    x <= 0}
# dmax_dsum = {1    x > 0
#              0    x <= 0}
# dsum_dxw0 = 1
# dxw0_dx0 = dw0 * 1
# dxw0_dw0 = dx0 * 1
# dmax_dx0 = 1 || 0 * 1  * dw0
# dmax_dw0 = 1 || 0 * 1  * dx0
# y0 = i0 * n0w0 + i0 * n1w0 + i0 * n2w0
# dlayer_y0 = 2
# dy0_di0 = n0w0 + n1w0 + n2w0
# dlayer_di0 = 2 * n0w0 + 2 * n1w0 + 2 * n2w0
# dy0_dn0w0 = dlayer0 * i0 + dlayer1 * i0 + dlayer2 * i0
# dloss = [1 / l, 1 / l, 1 / l]


dvalues = np.array([[1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3]])
inputs = np.array([[1, 2, 3, 2.5],
                   [2, 5, -1, 2],
                   [-1.5, 2.7, 3.3, -0.8]])
weights = np.array([[0.2, 0.8, -0.5, 1],
                   [0.5, -0.91, 0.26, -0.5],
                   [-0.26, -0.27, 0.17, 0.87]])
print(np.dot(dvalues.T, inputs))


dmax_dsum = (1. if sum > 0 else 0)
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
dxw0_dx0 = weights[0]
dxw0_dw0 = inputs[0]
dxw0_dx1 = weights[1]
dxw0_dw1 = inputs[1]
dxw0_dx2 = weights[2]
dxw0_dw2 = inputs[2]
dmax_dx0 = dmax_dsum * dsum_dxw0 * dxw0_dx0
dmax_dw0 = dmax_dsum * dsum_dxw0 * dxw0_dw0
dmax_dx1 = dmax_dsum * dsum_dxw1 * dxw0_dx1
dmax_dw1 = dmax_dsum * dsum_dxw1 * dxw0_dw1
dmax_dx2 = dmax_dsum * dsum_dxw2 * dxw0_dx2
dmax_dw2 = dmax_dsum * dsum_dxw2 * dxw0_dw2
dmax_db = dmax_dsum * dsum_db
print(dmax_dx0, dmax_dw0)
print(dmax_dx1, dmax_dw1)
print(dmax_dx2, dmax_dw2)
print(dmax_db)
weights = weights - np.array([dmax_dw0, dmax_dw1, dmax_dw2]) * 0.01
print(weights)
xw0 = inputs[0] * weights[0]
xw1 = inputs[1] * weights[1]
xw2 = inputs[2] * weights[2]
sum = xw0 + xw1 + xw2 + bias
max2 = max(0, sum)
print(max2)

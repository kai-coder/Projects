class vec:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

def transpose(arr):
    return [[arr[j][i] for j in range(len(arr))] for i in range(len(arr[0]))]

def arrRange(arr, low=0, high=1):
    for i in arr: 
        if low >= i or i >= high: return False
    return True

def scrollArray(arr, amount):
    amount = amount%len(arr)
    return arr[amount:] + arr[:amount]

def fillArray(className, dim, inputVal=None):
    if inputVal:
        return [[className(inputVal) for i in range(dim.x)] for j in range(dim.y)] if dim.y > 0 else [className(inputVal) for i in range(dim.x)]
    else:
        return [[className for i in range(dim.x)] for j in range(dim.y)] if dim.y > 0 else [className for i in range(dim.x)]

def reverseArray(arr, dim):
    if dim == 0:
        return [arr[i][::-1] for i in range(len(arr))]
    elif dim == 1:
        return arr[::-1]
    return arr

def rotateArray(arr, amount):
    amount = amount % 4
    if amount != 0:
        newArr = transpose(arr)
        newArr = reverseArray(newArr, 0)
        for i in range(amount - 1):
            newArr = transpose(newArr)
            newArr = reverseArray(newArr, 0)
        return newArr
    else:
        return arr

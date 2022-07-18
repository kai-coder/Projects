class imageClass():
    def __init__(self, images, animationRate, reverse):
        self.images = images
        self.animationRate = animationRate
        self.reverse = reverse

def screenSetup(frame=60, type=IMAGE, trace=True):
    frameRate(frame)
    textureMode(type)
    if not trace: noStroke()
    
def readTileMap(img, size, animationSize=1, animationRate=60, reverse=False):
    img = loadImage(img)
    returnArr = []
    tempArr = []
    animationSize = [max(i, 1) for i in animationSize] if isinstance(animationSize, list) else max(animationSize, 1)
    tempIndex = 0
    for y in range(img.height / size.y):
        for x in range(img.width / size.x):
            tempArr.append(img.get(x * size.x, y * size.y, size.x, size.y))
            if isinstance(animationSize, list):
                if len(tempArr) >= animationSize[tempIndex]:
                    returnArr.append(imageClass(tempArr, animationRate))
                    tempArr = []
                    if tempIndex + 1 < len(animationSize):
                        tempIndex += 1
            elif len(tempArr) >= animationSize:
                returnArr.append(imageClass(tempArr, animationRate, reverse))
                tempArr = []
            
    else:
        if len(tempArr) > 0:
            returnArr.append(imageClass(tempArr, animationRate))
    return returnArr

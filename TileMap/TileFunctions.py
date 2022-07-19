from ArrayFunctions import *
from DrawFunctions import *

class Tile:
    def __init__(self, inputs):
        self.defaultInputs = inputs
        self.mapIndex, self.imgIndex = self.defaultInputs[0]
        self.rotation, self.scale = self.defaultInputs[1:]
        self.animation = 0
        self.opacity = 0
        
    def clear(self):
        self.mapIndex, self.imgIndex = self.defaultInputs[0]
        self.rotation, self.scale = self.defaultInputs[1:]
        self.animation = 0
        self.opacity = 0
    
    def set(self, inputs):
        self.mapIndex, self.imgIndex = inputs[0]
        self.rotation, self.scale = inputs[1:]
        self.animation = 0
    
    def draw(self, img, pos, size, rotation, scale, DELTA, animNum, animRate, reverse):
        self.opacity -= 0.05 * DELTA
        if reverse:
            if floor(frameCount/(60./animRate))%((animNum-1)*2) < animNum:
                self.animation = floor(frameCount/(60./animRate))%((animNum-1)*2)
            else:
                self.animation = -floor(frameCount/(60./animRate))%((animNum-1)*2)
        else:
            self.animation = floor(frameCount/(60./animRate))%(animNum)
        drawImg(img, pos, size, rotation, scale)
        fill(255, 255, 255, self.opacity * 255)
        rect(pos.x, pos.y, size.x, size.y)

def ruleCheck(rule, checkIndexX, checkIndexY, tileArr, tileMapIndex):
    for ruleY in range(3):
        for ruleX in range(3):
            if [ruleY, ruleX] == [1, 1]:
                continue
            currentRule = rule[ruleY][ruleX]
            offsetX, offsetY = ruleX - 1, ruleY - 1
            ruleCheckX, ruleCheckY = checkIndexX + offsetX, checkIndexY + offsetY
            if currentRule != None:
                if arrRange([ruleCheckX, ruleCheckY], -1, len(tileArr)):
                    ruleCheck = tileArr[ruleCheckY][ruleCheckX]
                    tileOccurance = 0 if ruleCheck.mapIndex == None or ruleCheck.mapIndex != tileMapIndex else 1
                else :
                    tileOccurance = 1
                if tileOccurance != currentRule:
                    return False
    return True

def ruleCheckLoop(rules, checkIndexX, checkIndexY, tileArr, tileMapIndex):
    for tileImgIndex in range(len(rules)):
        currentRule = rules[tileImgIndex]
        inputChar = currentRule[1][1]
        if "S" in inputChar:
            if ruleCheck(currentRule, checkIndexX, checkIndexY, tileArr, tileMapIndex):
                return [[tileMapIndex, tileImgIndex], 0, [1, 1]]
        else:
            scaleX = 0
            scaleY = 0
            rotation = 0
            for i in range(("R" in inputChar) * 3 + 1):
                for y in range(("Y" in inputChar) + 1):
                    ruleModified = rotateArray(currentRule, i)
                    rotation = i
                    if y == 1:
                        ruleModified = reverseArray(ruleModified, 1)
                    scaleY = y
                    for x in range(("X" in inputChar) + 1):
                        if x == 1:
                            ruleModified = reverseArray(ruleModified, 0)
                        scaleX = x
                        if ruleCheck(ruleModified, checkIndexX, checkIndexY, tileArr, tileMapIndex):
                            return [[tileMapIndex, tileImgIndex], rotation, [scaleX * -2 + 1, scaleY * -2 + 1]]
        
    return [[tileMapIndex, -1], 0, [1, 1]]

def tileCheckLoop(rules, tilePos, drawIndex, tileNum, selectLayer, centerTile):
    for tileCheckY in range(3):
        for tileCheckX in range(3):
            offsetX, offsetY = tileCheckX - 1, tileCheckY - 1
            checkIndexX, checkIndexY = tilePos.x + offsetX, tilePos.y + offsetY
    
            if arrRange([checkIndexX, checkIndexY], -1, tileNum):
                tileCheck = drawIndex[selectLayer][checkIndexY][checkIndexX]
                if (mouseButton != RIGHT or tileCheck != centerTile) and tileCheck.mapIndex != None and len(rules) > tileCheck.mapIndex:
                    tileCheck.set(ruleCheckLoop(rules[tileCheck.mapIndex], checkIndexX, checkIndexY, drawIndex[selectLayer], tileCheck.mapIndex))

from TileFunctions import *
from EnvSetup import *
from rules import rules
from MouseInputs import *
from ArrayFunctions import *

# ONLY THIS FILE AND RULE.PY IS IMPORTANT

# Tile number; Amount of tiles on screen; Should be a factor of 800
tileNum = 20
# Tile Maps; Each tile should be in order for animation

# Order Example: 0 1 2
#                3 4 5
#                6 7 8

# For tiles with Animation, put them in an array like so: [file, animationsize, animationSpeed, Loop Backwards(T/F)]
# When having varying animationsize, you can have an array there like so: [file, [animationsize1, animationsize2, animationsize3], animationSpeed, Loop Backwards(T/F)]

tileMaps = [["GrassTiles.png", 5, 3, True], "WaterTiles.png", ["CharTiles.png", 4, 6]]

# Set the tile size for each tile map
imgTileSizes = [[50, 50], [50, 50], [14, 16]]



DELTA = 1
tileMapIndex = 0
selectLayer = 0
    
def setup():
    global tileSize, drawIndex, layerData, imgArray, buttonclick
    buttonclick = True
    
    # Set screen size and make it less pixelated
    size(1000, 800, P2D)
    pixelDensity(displayDensity())
    screenSetup(60, NORMAL)
    
    tileSize = (width - 200)/tileNum
    # Create tiles and store in array
    drawIndex = [fillArray(Tile, vec(tileNum, tileNum), [[None, None], 0, [1, 1]])]
    layerData = [[1, tileNum]]
    
    # Create imgs and store in array
    imgArray = []
    for imgIndex in range(len(tileMaps)):
        img = tileMaps[imgIndex]
        imgSizes = vec(imgTileSizes[imgIndex][0], imgTileSizes[imgIndex][1])
        if isinstance(img, list):
            if len(img) == 1:
                imgArray.append(readTileMap(img[0], imgSizes))
            elif len(img) == 2:
                imgArray.append(readTileMap(img[0], imgSizes, img[1]))
            elif len(img) == 3:
                imgArray.append(readTileMap(img[0], imgSizes, img[1], img[2]))
            else:
                imgArray.append(readTileMap(img[0], imgSizes, img[1], img[2], img[3]))
        else:
            imgArray.append(readTileMap(img, imgSizes))
                            
def draw():
    global tileSize, drawIndex, imgArray, tileMapIndex, selectLayer, buttonclick, layerData, DELTA
    DELTA = frameRate/60
    background(0, 0, 255)
    if mousePressed:
        if mouseX <= 800:
            for tileY in range(tileNum):
                for tileX in range(tileNum):
                    if rectClick(tileX * tileSize, tileY * tileSize, tileSize, tileSize):
                        tilePos = vec(tileX, tileY);
                        centerTile = drawIndex[selectLayer][tilePos.y][tilePos.x]
                        if (mouseButton == RIGHT and centerTile.mapIndex != None) or (mouseButton == LEFT and (centerTile.mapIndex == None or centerTile.mapIndex != tileMapIndex)):
                            if mouseButton == RIGHT and centerTile.mapIndex != None:
                                centerTile.clear()
                            elif mouseButton == LEFT and (centerTile.mapIndex == None or centerTile.mapIndex != tileMapIndex):
                                centerTile.set([[tileMapIndex, 0], 0, [1, 1]])
                            if len(rules) > tileMapIndex:
                                tileCheckLoop(rules, tilePos, drawIndex, tileNum, selectLayer, centerTile)
                            elif mouseButton == LEFT:
                                tileCheckLoop(rules, tilePos, drawIndex, tileNum, selectLayer, centerTile)
                                centerTile.set([[tileMapIndex, int(random(len(tileMaps[tileMapIndex]) + 1))], 0, [1, 1]])
                            
        else:
            for img in range(len(tileMaps)):
                if rectClick(img % 3 * 71 + 804, floor(img/3) * 71 + 4, 50, 50):
                    tileMapIndex = img
            for i in range(len(drawIndex)):
                if rectClick(800, 700 - i * 50, 50, 50) and buttonclick == True:
                    buttonclick = False
                    if mouseButton == RIGHT:
                        drawIndex.pop(i)
                        layerData.pop(i)
                        if len(drawIndex) > 0:
                            selectLayer = (len(drawIndex) - 1) if selectLayer >= len(drawIndex) else selectLayer
                        else:
                            drawIndex.append(fillArray(Tile, vec(tileNum, tileNum), [[None, None], 0, [1, 1]]))
                            layerData.append([1, tileNum])
                    else:
                        for z in drawIndex[selectLayer]:
                            for y in z:
                                y.opacity = 0
                        selectLayer = i
                        for z in drawIndex[selectLayer]:
                            for y in z:
                                y.opacity = 1 if y.mapIndex != None else y.opacity
            if rectClick(804, 746, 50, 50) and buttonclick == True:
                drawIndex.append(fillArray(Tile, vec(tileNum, tileNum), [[None, None], 0, [1, 1]]))
                layerData.append([max([i[0] for i in layerData]) + 1, tileNum])
                selectLayer = len(drawIndex) - 1
                buttonclick = False
    if not mousePressed:
        buttonclick = True
    # Draw Tiles
    stroke(100, 70, 30)
    strokeWeight(4)
    fill(150, 100, 30)
    rect(802, 2, 196, 796)
    stroke(0)
    strokeWeight(1)
    for i in range(len(drawIndex)):
        textSize(15)
        fill(*[0, 255, 255] if selectLayer == i else [255])
        rect(804, 696 - i * 50, 50, 50)
        fill(0)
        textAlign(CENTER);
        text("Layer\n" + str(layerData[i][0]), 829, 716 - i * 50); 
    textSize(15)
    fill(255, 0, 0)
    rect(804, 746, 50, 50)
    fill(0)
    textAlign(CENTER);
    text("New\nLayer", 829, 766); 
    noStroke()
    for img in range(len(imgArray)):
        drawImg(imgArray[img][0].images[0], vec(img % 3 * 71 + 804, floor(img/3) * 71 + 4), vec(50, 50))
    for layer in drawIndex:
        for tileY in range(tileNum):
            for tileX in range(tileNum):
                drawTile = layer[tileY][tileX]
                if drawTile.imgIndex != None:
                    if drawTile.imgIndex == -1:
                        fill(255)
                        rect(tileX * tileSize, tileY * tileSize, tileSize, tileSize)
                    else:
                        drawTile.draw(imgArray[drawTile.mapIndex][drawTile.imgIndex].images[drawTile.animation], vec(tileX * tileSize, tileY * tileSize), vec(tileSize, tileSize), drawTile.rotation, drawTile.scale, DELTA, len(imgArray[drawTile.mapIndex][drawTile.imgIndex].images), imgArray[drawTile.mapIndex][drawTile.imgIndex].animationRate, imgArray[drawTile.mapIndex][drawTile.imgIndex].reverse)
    for tileY in range(tileNum):
        for tileX in range(tileNum):
            if rectClick(tileX * tileSize, tileY * tileSize, tileSize, tileSize):
                
                drawImg(imgArray[tileMapIndex][0].images[0], vec(tileX * tileSize, tileY * tileSize), vec(tileSize, tileSize), 0, [1,1], 200)

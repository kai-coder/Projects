from ArrayFunctions import *

def drawImg(img, pos, size, rotation = 0, scale = [0,0], trans = 255):
    tint(255, trans);
    beginShape(QUAD)
    texture(img)
    points = rotateArray([[[0, 0], [1, 0]],
                         [[0, 1], [1, 1]]], rotation)
    if scale[0] == -1: points = reverseArray(points, 0)
    if scale[1] == -1: points = reverseArray(points, 1)
    vertex(pos.x, pos.y, *points[0][0])
    vertex(pos.x, pos.y + size.y, *points[1][0])
    vertex(pos.x + size.x, pos.y + size.y, *points[1][1])
    vertex(pos.x + size.x, pos.y, *points[0][1])
    endShape()

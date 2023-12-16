import taichi as ti 
from bilinear_interpolation import BilinearInterpolation

@ti.func
def SampleColor(srcColor, img: ti.types.ndarray(element_dim = 1), cellSize:ti.i32, cellIndex : ti.i32, cellCol:ti.i32):
    colIndex = cellIndex % cellCol
    rowIndex = ti.floor(cellIndex / cellCol)

    col = cellSize * (colIndex + srcColor[2]) - srcColor[2]
    row = cellSize * (rowIndex + srcColor[1]) - srcColor[1]

    color = BilinearInterpolation(img, row, col)

    # print(srcColor, color, cellIndex, cellCol,img[0, 15], rowIndex, colIndex, row, col)
    
    return color / 255.0

@ti.func
def LookUpTable(srcColor, lutImg: ti.types.ndarray(element_dim = 1), cellSize:ti.f32, cellRow:ti.f32, cellCol:ti.f32):
    cellCount = cellRow * cellCol
    b = srcColor[0]
    cellIndex = b * (cellCount - 1)
    cellIndex1 = ti.floor(cellIndex) # low
    cellIndex2 = ti.ceil(cellIndex) # high
    
    lowColor = SampleColor(srcColor, lutImg, cellSize, cellIndex1, cellCol)
    highColor = SampleColor(srcColor, lutImg, cellSize, cellIndex2, cellCol)

    return ti.math.mix(lowColor, highColor, cellIndex - cellIndex1)
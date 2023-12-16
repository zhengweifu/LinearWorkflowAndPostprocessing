import cv2, os
import numpy as np
import taichi as ti 
from lut import LookUpTable

ti.init()

Img2d = ti.types.ndarray(element_dim = 1)
Vector3 = ti.types.vector(n=3, dtype=ti.f32)

CELL_SIZE: ti.f32 = 16.0

@ti.kernel
def Kernel(srcImg: Img2d, lutImg: Img2d):
    h, w = srcImg.shape[0], srcImg.shape[1]
    lutH, lutW = lutImg.shape[0], lutImg.shape[1]
    cellRow = lutH / CELL_SIZE
    cellCol = lutW / CELL_SIZE
    for i, j in ti.ndrange(h, w):
        c = Vector3(srcImg[i, j] / 255.0)
        srcImg[i, j] = (LookUpTable(c, lutImg, CELL_SIZE, cellRow, cellCol) * 255.0).cast(ti.u8)

def Main():
    imgDir = os.path.join(os.getcwd(), "images")
    srcImgPath = os.path.join(imgDir, "sh_smaller.png")
    # srcImgPath = os.path.join(imgDir, "test.png")
    srcImg = cv2.imread(srcImgPath, cv2.IMREAD_UNCHANGED)

    lutImgPath = os.path.join(imgDir, "LUT_Sepia.webp")
    # lutImgPath = os.path.join(imgDir, "RGBTable16x1.webp")
    lutImg = cv2.imread(lutImgPath, cv2.IMREAD_UNCHANGED)

    Kernel(srcImg, lutImg)

    titleName = "Color Grading"
    cv2.imshow(titleName, srcImg)
    cv2.waitKey()

if __name__ == "__main__":
    Main()
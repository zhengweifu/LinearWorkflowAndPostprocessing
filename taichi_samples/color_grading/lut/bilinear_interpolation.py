import cv2
import numpy as np
import taichi as ti
import taichi.math as tm

@ti.func
def BilinearInterpolation(img: ti.types.ndarray(element_dim = 1), row: ti.f32, col: ti.f32):
    r1, c1 = ti.i32(row), ti.i32(col) #top-left corner
    r2, c2 = ti.i32(ti.ceil(row)), ti.i32(ti.ceil(col)) #bottom-right corner

    color11 = img[r1, c1]
    color21 = img[r2, c1]
    color12 = img[r1, c2]
    color22 = img[r2, c2]

    color1 = tm.mix(color11, color21, row - r1)
    color2 = tm.mix(color12, color22, row - r1)

    return tm.mix(color1, color2, col - c1)
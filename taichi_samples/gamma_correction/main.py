import cv2, os
import numpy as np
import taichi as ti 

ti.init()

Img2d = ti.types.ndarray(element_dim = 1)
Vector3 = ti.types.vector(n=3, dtype=ti.f32)

@ti.func
def GammaCorrection(color):
    return ti.math.pow(color, 0.454545)


@ti.kernel
def GammaCorrectionKernel(img: Img2d):
    h, w = img.shape[0], img.shape[1]
    for i, j in ti.ndrange(h, w):
        c = Vector3(img[i, j] / 255.0)
        img[i, j] = (GammaCorrection(c) * 255.0).cast(ti.u8)

def Main():
    imgPath = os.path.join(os.getcwd(), "images/sh_smaller.png")
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    
    GammaCorrectionKernel(img)

    titleName = "Gamma Correction"
    cv2.imshow(titleName, img)
    cv2.waitKey()

if __name__ == "__main__":
    Main()
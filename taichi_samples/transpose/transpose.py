import cv2, os
import numpy as np
import taichi as ti 

img2d = ti.types.ndarray(element_dim = 1)

@ti.kernel
def transpose(src: img2d, dst: img2d, w: int, h: int):
    for i, j in ti.ndrange(h, w):
        dst[j, i] = src[i, j]

def main():
    ti.init()
    imgPath = os.path.join(os.getcwd(), "images/cat.jpg")
    src = cv2.imread(imgPath)
    h, w, c = src.shape
    dst = np.zeros((w, h, c), dtype = src.dtype)

    transpose(src, dst, w, h)

    cv2.imshow("transpose", dst)
    cv2.waitKey()

if __name__ == "__main__":
    main()
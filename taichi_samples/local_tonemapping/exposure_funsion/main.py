#参考论文：
#https://www.tommertens.com/papers/exposure_fusion_reduced.pdf
#https://bartwronski.com/2022/02/28/exposure-fusion-local-tonemapping-for-real-time-rendering/
#参考实现：
#https://github.com/bartwronski/bartwronski.github.io/blob/main/local_tonemapping_js_demo/main.js
#https://bartwronski.github.io/local_tonemapping_js_demo/main.js

import cv2, os
import numpy as np
import taichi as ti 

ti.init()

Img2d = ti.types.ndarray(element_dim = 1)
Vector3 = ti.types.vector(n=3, dtype=ti.f32)
Vector2 = ti.types.vector(n=2, dtype=ti.f32)


@ti.kernel
def Kernel(img: Img2d, intensity: ti.f32, smooth: ti.f32):
    h, w = img.shape[0], img.shape[1]
    for i, j in ti.ndrange(h, w):
        c = Vector3(img[i, j] / 255.0)
        uv = Vector2(j / w, i / h)
        img[i, j] = (Vector3([0, uv.y, uv.x]) * 255.0).cast(ti.u8)

def Main():
    imgPath = os.path.join(os.getcwd(), "images/veranda_1k.hdr")
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    
    Kernel(img, 1.0, 5.0)

    titleName = "Exposure Funsion tonemapping"
    cv2.imshow(titleName, img)
    cv2.waitKey()

if __name__ == "__main__":
    Main()
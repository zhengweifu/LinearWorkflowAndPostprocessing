
import cv2, os
import numpy as np
import taichi as ti 

ti.init()

Img2d = ti.types.ndarray(element_dim = 1)
Vector3 = ti.types.vector(n=3, dtype=ti.f32)
Vector2 = ti.types.vector(n=2, dtype=ti.f32)

@ti.func
def Vigentte(color, uv, vigentteColor, intensity, smooth):
    center = Vector2([0.5, 0.5])
    dist = ti.abs(uv - center) * intensity
    factor = ti.math.pow(ti.math.clamp(1.0 - ti.math.dot(dist, dist), 0.0, 1.0), smooth)
    return color * factor +  vigentteColor * (1.0 - factor)


@ti.kernel
def VigentteKernel(img: Img2d, intensity: ti.f32, smooth: ti.f32):
    h, w = img.shape[0], img.shape[1]
    vigentteColor = Vector3([0.1, 0.3, 0.2]) #BGR
    for i, j in ti.ndrange(h, w):
        c = Vector3(img[i, j] / 255.0)
        uv = Vector2(j / w, i / h)
        # img[i, j] = (Vector3([0, uv.y, uv.x]) * 255.0).cast(ti.u8)
        img[i, j] = (Vigentte(c, uv, vigentteColor, intensity, smooth) * 255.0).cast(ti.u8)

def Main():
    imgPath = os.path.join(os.getcwd(), "images/sh_smaller.png")
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    
    VigentteKernel(img, 1.0, 5.0)

    titleName = "Vigentte"
    cv2.imshow(titleName, img)
    cv2.waitKey()

if __name__ == "__main__":
    Main()
import cv2, os
import numpy as np
import taichi as ti 

from reinhard_tonemapping import ReinhardToneMapping
from cryengine_tonemapping import CryEngineToneMapping
from filmic_tonemapping import FilmicToneMapping
from aces_tonemapping import ACESToneMapping
from advance_aces_tonemapping import AdvanceACESToneMapping

ti.init()

Img2d = ti.types.ndarray(element_dim = 1)
Vector3 = ti.types.vector(n=3, dtype=ti.f32)

@ti.func
def GammaCorrection(color):
    return ti.math.pow(color, 0.454545)

# 0->ReinhardToneMapping, 
# 1->CryEngineToneMapping, 
# 2->FilmicToneMapping, 
# 3->ACESToneMapping,
# 4->AdvanceACESToneMapping
tonemapingType = 4

@ti.kernel
def ToneMapping(img: Img2d):
    h, w = img.shape[0], img.shape[1]
    halfW = w * 0.5
    halfH = h * 0.5
    for i, j in ti.ndrange(h, w):
        c = Vector3(img[i, j])
        if tonemapingType == 0:
            img[i, j] = ReinhardToneMapping(c, 1.0)
        elif tonemapingType == 1:
            img[i, j] = CryEngineToneMapping(c, 1.0)
        elif tonemapingType == 2:
            img[i, j] = FilmicToneMapping(c, 1.0)
        elif tonemapingType == 3:
            img[i, j] = ACESToneMapping(c, 1.0)
        elif tonemapingType == 4:
            img[i, j] = AdvanceACESToneMapping(c, 1.0)
        img[i, j] = GammaCorrection(img[i, j])

def Main():
    imgPath = os.path.join(os.getcwd(), "images/veranda_1k.hdr")
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    
    ToneMapping(img)

    titleName = "No tonemapping"
    if tonemapingType == 0:
        titleName = "Reinhard tonemapping"
    elif tonemapingType == 1:
        titleName = "CryEngine tonemapping"
    elif tonemapingType == 2:
        titleName = "Filmic tonemapping"
    elif tonemapingType == 3:
        titleName = "ACES tonemapping"
    elif tonemapingType == 4:
        titleName = "ACES tonemapping advance"
    cv2.imshow(titleName, img)
    cv2.waitKey()

if __name__ == "__main__":
    Main()
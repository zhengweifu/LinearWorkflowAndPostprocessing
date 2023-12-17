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

@ti.kernel
def ToneMapping(img: Img2d, tonemapingType: ti.u8):
    h, w = img.shape[0], img.shape[1]
    halfW = w * 0.5
    halfH = h * 0.5
    for i, j in ti.ndrange(h, w):
        c = Vector3(img[i, j])
        if tonemapingType == 1:
            img[i, j] = ReinhardToneMapping(c, 1.0)
        elif tonemapingType == 2:
            img[i, j] = CryEngineToneMapping(c, 1.0)
        elif tonemapingType == 3:
            img[i, j] = FilmicToneMapping(c, 1.0)
        elif tonemapingType == 4:
            img[i, j] = ACESToneMapping(c, 1.0)
        elif tonemapingType == 5:
            img[i, j] = AdvanceACESToneMapping(c, 1.0)
        img[i, j] = GammaCorrection(img[i, j])

def Main():
    imgPath = os.path.join(os.getcwd(), "images/veranda_1k.hdr")
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    srcImg = img.swapaxes(0, 1)[:, ::-1, ::-1].copy()

    titleName = "Tone Mapping"
    window = ti.ui.Window(titleName, res=(srcImg.shape[0], srcImg.shape[1]), fps_limit=200, pos = (150, 150))

    tonemapingType = 0
    tonemapingTypeName = "No tonemapping"
    while window.running:
        window.GUI.begin("Tone Mapping", 0.0, 0.0, 0.2, 0.2)
        if tonemapingType == 0:
            tonemapingTypeName = "No tonemapping"
        elif tonemapingType == 1:
            tonemapingTypeName = "Reinhard tonemapping"
        elif tonemapingType == 2:
            tonemapingTypeName = "CryEngine tonemapping"
        elif tonemapingType == 3:
            tonemapingTypeName = "Filmic tonemapping"
        elif tonemapingType == 4:
            groutonemapingTypeNamepName = "ACES tonemapping"
        elif tonemapingType == 5:
            tonemapingTypeName = "ACES tonemapping advance"
        window.GUI.text(tonemapingTypeName)
        tonemapingType = window.GUI.slider_int('', tonemapingType, minimum=0, maximum=5)
        window.GUI.end()

        img = srcImg.copy()

        ToneMapping(img, tonemapingType)

        uiShape = window.get_window_shape()
        img_padded = np.zeros(dtype=np.float32, shape=(uiShape[0], uiShape[1], 3))
        img_padded[:img.shape[0], :img.shape[1]] = img
        window.get_canvas().set_image(img_padded)

        window.show()

if __name__ == "__main__":
    Main()
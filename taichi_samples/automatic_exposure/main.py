import cv2, os, datetime 
import numpy as np
import taichi as ti 

ti.init()

Img2d = ti.types.ndarray(element_dim = 1)
Vector3 = ti.types.vector(n=3, dtype=ti.f32)
# Step 0
# -------------- Just Input Image Luminance Begin -------------- 
@ti.kernel
def JustImageLuminanceKernel(srcImg: Img2d, scaleFactor: ti.f32):
    h, w = srcImg.shape[0], srcImg.shape[1]
    for i, j in ti.ndrange(h, w):
        srcImg[i, j] *= scaleFactor
# -------------- Just Input Image Luminance End   -------------- 

# Step 1
# -------------- Automatic Exposure Luminance Histogram Begin -------------- 
EPSILON = 0.005
NUM_HISTOGRAM_BINS = 256

gMinLogLuminance = ti.field(ti.f32, ())
gMaxLogLuminance = ti.field(ti.f32, ())
gLogLuminanceRange = ti.field(ti.f32, ())
gOneOverLogLuminanceRange = ti.field(ti.f32, ())

gLuminanceHistograms = ti.field(ti.i32, NUM_HISTOGRAM_BINS)

@ti.func
def GetLuminance(color: Vector3):
    return ti.math.dot(color, Vector3(0.2127, 0.7152, 0.0722))

@ti.func
def HDRToHistogramBin(color: Vector3) -> ti.i32:
    luminance = GetLuminance(color)
    result = 0
    if luminance > EPSILON:
        logLuminance = ti.math.clamp((ti.math.log2(luminance) - gMinLogLuminance[None]) * gOneOverLogLuminanceRange[None], 0.0, 1.0) 

        # from [0, 1] to [1, 255]
        result = ti.i32(logLuminance * 254.0 + 1.0) 
    return result
    

@ti.kernel
def ClearHistogramKernel():
    for i in ti.ndrange(NUM_HISTOGRAM_BINS):
        gLuminanceHistograms[i] = 0

@ti.kernel
def HistogramKernel(srcImg: Img2d):
    h, w = srcImg.shape[0], srcImg.shape[1]
    for i, j in ti.ndrange(h, w):
        c = Vector3(srcImg[i, j])
        binIndex = HDRToHistogramBin(c)
        ti.atomic_add(gLuminanceHistograms[binIndex], 1)

def PrintgLuminanceHistograms():
    for i in range(NUM_HISTOGRAM_BINS):
        print(gLuminanceHistograms[i])
# -------------- Automatic Exposure Luminance Histogram End -------------- 

# Step 2
# -------------- Automatic Exposure Luminance Average Begin --------------
gLastFrameLuminance = ti.field(ti.f32, ())
gTimeCoff = ti.field(ti.f32, ())
@ti.kernel
def LuminanceAverageKernel(srcImg: Img2d):
    pixelCount = srcImg.shape[0] * srcImg.shape[1]
    adaptedLuminance = 0.0
    for i in range(NUM_HISTOGRAM_BINS):
        perLuminances = gLuminanceHistograms[i] * i
        ti.atomic_add(adaptedLuminance, perLuminances)
    weightedLogAverage = adaptedLuminance / (ti.math.max(pixelCount - gLuminanceHistograms[0], 1.0)) - 1.0
    weightedAvgerageLuminance = ti.math.pow(2.0, weightedLogAverage / 254.0 * gLogLuminanceRange[None] + gMinLogLuminance[None])
    adaptedLuminance = gLastFrameLuminance[None] + (weightedAvgerageLuminance - gLastFrameLuminance[None]) * gTimeCoff[None]
    gLastFrameLuminance[None] = adaptedLuminance
    #print(weightedLogAverage, weightedAvgerageLuminance, gLastFrameLuminance[None])
# -------------- Automatic Exposure Luminance Average End --------------

# Step 3
# -------------- Tone Mapping and Gamma Correction Begin  --------------
@ti.func
def GammaCorrection(color):
    return ti.math.pow(color, 0.454545)

# sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
ACESInputMat = ti.Matrix([
    [0.59719, 0.35458, 0.04823],
    [0.07600, 0.90834, 0.01566],
    [0.02840, 0.13383, 0.83777]
])

# ODT_SAT => XYZ => D60_2_D65 => sRGB
ACESOutputMat = ti.Matrix([
    [ 1.60475, -0.53108, -0.07367],
    [-0.10208,  1.10813, -0.00605],
    [-0.00327, -0.07276,  1.07602]
])

@ti.func
def RRTAndODTFit(v):
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    return a / b

@ti.func
def AdvanceACESToneMapping(color, adaptedLum):
    color = ACESInputMat @ color

    color *= adaptedLum

    # Apply RRT and ODT
    color = RRTAndODTFit(color)

    color = ACESOutputMat @ color

    # Clamp to [0, 1]
    color = ti.math.clamp(color, 0.0, 1.0)

    return color

gUseToneMapping = ti.field(ti.i32, ())

@ti.kernel
def ToneMappingKernel(srcImg: Img2d, bUseExposure: ti.u8):
    h, w = srcImg.shape[0], srcImg.shape[1]
    for i, j in ti.ndrange(h, w):
        c = Vector3(srcImg[i, j])
        adaptedLum = 1.0 / (9.6 * gLastFrameLuminance[None] + 0.0001)
        adaptedLum = adaptedLum if bUseExposure else 1.0
        if gUseToneMapping[None] > 0:
            srcImg[i, j] = AdvanceACESToneMapping(c, adaptedLum)
        else:
            srcImg[i, j] *= adaptedLum
        srcImg[i, j] = GammaCorrection(srcImg[i, j])

# -------------- Tone Mapping and Gamma Correction End  --------------
UI_TOP = 10.0
UI_RIGHT = 10.0
UI_WIDTH = 360.0
UI_HEIGHT = 200.0
def Main():
    imgDir = os.path.join(os.getcwd(), "images")
    srcImgPath = os.path.join(imgDir, "veranda_1k.hdr")
    srcImg = cv2.imread(srcImgPath, cv2.IMREAD_UNCHANGED)
    srcImg = srcImg.swapaxes(0, 1)[:, ::-1, ::-1].copy()

    guiRes = (srcImg.shape[0], srcImg.shape[1])
    titleName = "Automic Exposure"
    window = ti.ui.Window(titleName, guiRes, fps_limit=200, pos = (150, 150))
    uiCanvas = window.get_canvas()
    gui = window.get_gui()

    # ui default
    uiMinLogLuminance = -10
    uiMaxLogLuminance = 20.0
    uiSpeed = 1.0
    uiUseToneMapping = True
    uiUseExposure = True
    uiSceneLuminance = 1.0

    lastTime = datetime.datetime.now()
    while window.running:
        # copy image
        img = srcImg.copy()

        # ui
        uiShape = window.get_window_shape()
        uiTop = UI_TOP / uiShape[1]
        uiLeft = (uiShape[0] - UI_RIGHT - UI_WIDTH) / uiShape[0]
        uiWidth =  UI_WIDTH / uiShape[0]
        uiHeigh = UI_HEIGHT / uiShape[1]
        gui.begin("Exposure", uiLeft, uiTop, uiWidth, uiHeigh)
        uiMinLogLuminance = gui.slider_float('Min EV100', uiMinLogLuminance, minimum=-10, maximum=20)
        uiMaxLogLuminance = gui.slider_float('Max EV100', uiMaxLogLuminance, minimum=-10, maximum=20)
        uiSpeed = gui.slider_float('Speed', uiSpeed, minimum=0.2, maximum=20)
        uiUseExposure = gui.checkbox('Use Exposure', uiUseExposure)
        uiUseToneMapping = gui.checkbox('Use Tone Mapping', uiUseToneMapping)
        uiSceneLuminance = gui.slider_float('Scene Luminance', uiSceneLuminance, minimum=0.1, maximum=10.0)
        gui.end()

        # time delta
        nowTime = datetime.datetime.now()
        timeDelta = (nowTime - lastTime).total_seconds()
        lastTime = nowTime

        # step 0
        gMinLogLuminance[None] = uiMinLogLuminance
        gMaxLogLuminance[None] = uiMaxLogLuminance
        gLogLuminanceRange[None] = ti.max(gMaxLogLuminance[None] - gMinLogLuminance[None], 0.01)
        gOneOverLogLuminanceRange[None] = 1.0 / gLogLuminanceRange[None]
        gTimeCoff[None] = 1.0 - ti.exp(-timeDelta * uiSpeed)
        gUseToneMapping[None] = 1 if uiUseToneMapping else 0
        JustImageLuminanceKernel(img, uiSceneLuminance)
        
        # step 1
        ClearHistogramKernel()
        HistogramKernel(img)
        
        # step 2
        LuminanceAverageKernel(img)
        
        # step 3
        ToneMappingKernel(img, uiUseExposure)

        img_padded = np.zeros(dtype=np.float32, shape=(guiRes[0], guiRes[1], 3))
        img_padded[:img.shape[0], :img.shape[1]] = img
        uiCanvas.set_image(img_padded)
        window.show()
    

if __name__ == "__main__":
    Main()
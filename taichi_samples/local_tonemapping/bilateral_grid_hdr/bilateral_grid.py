import cv2, os
import taichi as ti
import numpy as np

Img2d = ti.types.ndarray(element_dim = 1)
Vector3 = ti.types.vector(n=3, dtype=ti.f32)
Vector2 = ti.types.vector(n=2, dtype=ti.f32)

ti.init(arch=ti.gpu, debug=True)

#global parameters
gGrid = ti.Vector.field(2, dtype = ti.f32, shape = (512, 512, 128))
gGridBlurred = ti.Vector.field(2, dtype = ti.f32, shape = (512, 512, 128))
gWeights = ti.field(dtype=ti.f32, shape = (2, 512), offset = (0, -256))

@ti.func
def ComputeWeights(i, radius, sigma):
    total = 0.0
    ti.loop_config(serialize = True) #禁用taichi并行
    for j in range(-radius, radius + 1):
        val = ti.exp(-0.5 * (j / sigma)**2)
        gWeights[i, j] = val
        total += val

    ti.loop_config(serialize = True) #禁用taichi并行
    for j in range(-radius, radius + 1):
        gWeights[i, j] /= total

@ti.func
def SampleGridSpatial(i, j, k):
    g = ti.static(gGridBlurred)  # Create an alias
    mixIndex0 = ti.math.mix(g[int(i), int(j), k], g[int(i) + 1, int(j), k],
                     ti.math.fract(i))
    mixIndex1 = ti.math.mix(g[int(i), int(j) + 1, k], g[int(i) + 1,
                                                 int(j) + 1, k], ti.math.fract(i))
    return ti.math.mix(mixIndex0, mixIndex1, ti.math.fract(j))


@ti.func
def SampleGrid(i, j, k):
    return ti.math.mix(SampleGridSpatial(i, j, int(k)),
                  SampleGridSpatial(i, j,
                                      int(k) + 1), ti.math.fract(k))

LOG_LUMINANCE_SCALE = 16

@ti.func
def LogLuminance(c):
    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    return ti.max(ti.min((ti.log(lum) / ti.log(2) * LOG_LUMINANCE_SCALE) + 256, 256),
               0)

@ti.kernel
def BilateralFilterKernel(img: Img2d, 
                        scaleFactorWH: ti.f32, 
                        scaleFactorD: ti.f32,
                        sigmaFactorWH: ti.f32,
                        sigmaFactorD: ti.f32):
    # Step 1: Reset the grid
    gGrid.fill(0)
    gGridBlurred.fill(0)
    for i, j in ti.ndrange(img.shape[0], img.shape[1]):
        luminance = LogLuminance(img[i, j])
        row = ti.round(i / scaleFactorWH, ti.i32)
        col = ti.round(j / scaleFactorWH, ti.i32)
        dim = ti.round(luminance / scaleFactorD, ti.i32)
        gGrid[row, col, dim] += ti.math.vec2(luminance, 1)
    ComputeWeights(0, ti.ceil(sigmaFactorWH * 3, ti.i32), sigmaFactorWH)
    ComputeWeights(1, ti.ceil(sigmaFactorD * 3, ti.i32), sigmaFactorD)

    # Step 2: Grid processing (blur)
    gridRow: ti.i32 = (img.shape[0] + scaleFactorWH - 1) // scaleFactorWH
    gridCol: ti.i32 = (img.shape[1] + scaleFactorWH - 1) // scaleFactorWH
    gridDim: ti.i32 = (255 + scaleFactorD - 1) // scaleFactorD
    # print(img.shape[0], scaleFactorWH, gridRow)
    # print(img.shape[1], scaleFactorWH, gridCol)
    # print(255, scaleFactorD, gridDim)

    blurRadius = ti.ceil(sigmaFactorWH * 3, ti.i32)

    for i, j, k in ti.ndrange(gridRow, gridCol, gridDim):
        begin = ti.max(0, i - blurRadius)
        end = ti.min(gridRow, i + blurRadius + 1)

        total = ti.math.vec2(0, 0)
        for n in range(begin, end):
            total += gGrid[n, j, k] * gWeights[0, i - 1]
        gGridBlurred[i, j, k] = total

    for i, j, k in ti.ndrange(gridRow, gridCol, gridDim):
        begin = ti.max(0, j - blurRadius)
        end = ti.min(gridRow, j + blurRadius + 1)

        total = ti.math.vec2(0, 0)
        for n in range(begin, end):
            total += gGridBlurred[i, n, k] * gWeights[0, j - 1]
        gGrid[i, j, k] = total

    blurRadius = ti.ceil(sigmaFactorWH * 3, ti.i32)
    for i, j, k in ti.ndrange(gridRow, gridCol, gridDim):
        begin = ti.max(0, k - blurRadius)
        end = ti.min(gridRow, k + blurRadius + 1)

        total = ti.math.vec2(0, 0)
        for n in range(begin, end):
            total += gGrid[i, j, n] * gWeights[0, k - 1]
        gGridBlurred[i, j, k] = total

    # Step 3: Slicing
    for i, j in ti.ndrange(img.shape[0], img.shape[1]):
        luminance = LogLuminance(img[i, j])
        row = i / scaleFactorWH
        col = j / scaleFactorWH
        dim = luminance / scaleFactorD
        sample = SampleGrid(row, col, dim)
        print(sample)
        img[i, j] = ti.u8(sample[0] / sample[1])


def Main():
    imgPath = os.path.join(os.getcwd(), "images/mountain.jpg")
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    
    BilateralFilterKernel(img, 16, 16, 1, 1)

    titleName = "Bilateral grid tonemapping"
    cv2.imshow(titleName, img)
    cv2.waitKey()

if __name__ == "__main__":
    Main()
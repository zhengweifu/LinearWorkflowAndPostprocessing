import taichi as ti 

#参考：https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl


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
    color *= adaptedLum

    color = ACESInputMat @ color

    # Apply RRT and ODT
    color = RRTAndODTFit(color)

    color = ACESOutputMat @ color

    # Clamp to [0, 1]
    color = ti.math.clamp(color, 0.0, 1.0)

    return color
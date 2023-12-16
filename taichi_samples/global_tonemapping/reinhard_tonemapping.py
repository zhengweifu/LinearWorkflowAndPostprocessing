import taichi as ti 

@ti.func
def ReinhardToneMapping(color, adaptedLum):
    middleGrey = 1.0 # 中灰
    color *= middleGrey / adaptedLum
    return ti.math.clamp(color / (1.0 + color), 0.0, 1.0)
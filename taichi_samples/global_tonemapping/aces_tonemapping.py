import taichi as ti 

@ti.func
def ACESToneMapping(color, adaptedLum):
    a = 2.51; b = 0.03; c = 2.43; d = 0.59; e = 0.14
    color *= adaptedLum
    return ti.math.clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0)
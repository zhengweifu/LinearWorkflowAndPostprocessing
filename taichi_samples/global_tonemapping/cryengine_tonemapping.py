import taichi as ti 

@ti.func
def CryEngineToneMapping(color, adaptedLum):
    return 1.0 - ti.math.exp(-adaptedLum * color)
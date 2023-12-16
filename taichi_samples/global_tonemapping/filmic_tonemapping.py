import taichi as ti 

# a、b、c、d、e、f都是多项式的系数，而width是个magic number，表示白色的位置。
# 这个方法开启了tone mapping的新路径，让人们知道了曲线拟合的好处。
# 并且，其他颜色空间的变换，比如gamma矫正，也可以一起合并到这个曲线里来，一次搞定，不会增加额外开销。
# 缺点就是运算量有点大，两个多项式的计算，并且相除。
# a = Shoulder Strength
# b = Linear Strength
# b = Linear Angle
# d = Toe Strength
# e = Toe Numerator
# f = Toe Denominator
# Note: e/f = Toe Angle
# LinearWhite = Linear White Point Value
# F(x) = ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f
# FinalColor = F(LinearColor)/F(LinearWhite)
@ti.func
def F_(x):
    a = 0.22; b = 0.30; c = 0.10; d = 0.20; e = 0.01; f = 0.30
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f

@ti.func
def FilmicToneMapping(color, adaptedLum):
    width = 11.2
    return F_(1.6 * adaptedLum * color) / F_(width)
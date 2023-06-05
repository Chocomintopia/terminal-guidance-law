import math


def airdensityFun(ym):
    # 函数用于计算给定高度处空气密度和声速
    cT = -6.5*math.pow(10, -3)
    Tb = 288.25
    Pb = 10332.3
    Zb = 0.0
    r0 = 6356766
    g0 = 9.80665
    cP = 287.05287
    Z = r0*ym/(r0+ym)
    T = Tb+cT*(Z-Zb)
    P = Pb*g0*math.exp(g0/(cP*cT)*math.log(Tb/T))
    rou = P/(cP*T)
    v_sound = 20.046769*math.sqrt(T)
    return rou, v_sound

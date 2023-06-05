def table_aerocoeff_Ma_lookup(t, n):
    Table_aerocoeff = [
        [2.0000000e-001, 2.4100000e-001, 1.1000000e-001, 9.1500000e+000],
        [7.8000000e-001, 2.1300000e-001, 1.3600000e-001, 7.5100000e+000],
        [9.4000000e-001, 2.5800000e-001, 1.3500000e-001, 7.6100000e+000],
        [1.0700000e+000, 4.0700000e-001, 1.0900000e-001, 9.6300000e+000],
        [1.3200000e+000, 4.4500000e-001, 1.0800000e-001, 9.8900000e+000],
        [1.6100000e+000, 3.7200000e-001, 1.1500000e-001, 9.1800000e+000],
        [2.4300000e+000, 2.5500000e-001, 1.2100000e-001, 8.9100000e+000],
        [3.5000000e+000, 1.9000000e-001, 1.3400000e-001, 7.9400000e+000],
        [5.0000000e+000, 1.5000000e-001, 1.5400000e-001, 7.0300000e+000],
        [6.1000000e+000, 1.4500000e-001, 1.6000000e-001, 6.9300000e+000]
                        ]
    row_Table_aerocoeff = len(Table_aerocoeff)
    n0 = 1    # 矩阵下标从1开始，二维数组下标从0开始
    n1 = row_Table_aerocoeff
    y = 0
    if t >= Table_aerocoeff[n1-1][0]:
        y = Table_aerocoeff[n1-1][n-1]
    elif t <= Table_aerocoeff[n0-1][0]:
        y = Table_aerocoeff[n0-1][n-1]
    else:
        while n1-n0 > 1:
            # 此处需要注意python的round函数并不是精确四舍五入的 而是ROUND_HALF_EVEN策略
            # 所以在进行取整数四舍五入时需要进行一些处理
            # 将运算结果加上0.5再取整 就是四舍五入的结果
            i_interpolation = int((n1-n0)/2+n0+0.5)
            if t < Table_aerocoeff[i_interpolation-1][0]:
                n1 = i_interpolation
            else:
                n0 = i_interpolation
        if n1-n0 == 1:
            x1 = Table_aerocoeff[n0-1][0]
            x2 = Table_aerocoeff[n1-1][0]
            y1 = Table_aerocoeff[n0-1][n-1]
            y2 = Table_aerocoeff[n1-1][n-1]
            x = t
            y = y1+(y2-y1)/(x2-x1)*(x-x1)
    return y



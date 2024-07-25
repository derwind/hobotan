import numpy as np
from symengine import symbols as symengine_symbols

"""
SympyのSymbol関数にそのまま投げる関数
importをTYTANだけにするための申し訳ない方策
"""
def symbols(passed_txt):
    return symengine_symbols(passed_txt)

class TytanException(Exception):
    pass

"""
リストでまとめて定義する関数
"""
def symbols_list(shape, format_txt):
    #単一intの場合
    if type(shape) == int:
        shape = [shape]
    #print(shape)

    #次元チェック
    dim = len(shape)
    if dim != format_txt.count('{}'):
        raise TytanException("specify format option like format_txt=\'q{}_{}\' as dimension")
    
    #{}のセパレートチェック
    if '}{' in format_txt:
        raise TytanException("separate {} in format_txt like format_txt=\'q{}_{}\'")

    #次元が1～5でなければエラー
    if dim not in [1, 2, 3, 4, 5]:
        raise TytanException("Currently only dim<=5 is available. Ask tytan community.")

    #次元で分岐、面倒なのでとりあえずこれで5次元まで対応したこととする
    if dim == 1:
        q = []
        for i in range(shape[0]):
            q.append(symbols(format_txt.format(i)))
    elif dim == 2:
        q = []
        for i in range(shape[0]):
            tmp1 = []
            for j in range(shape[1]):
                tmp1.append(symbols(format_txt.format(i, j)))
            q.append(tmp1)
    elif dim == 3:
        q = []
        for i in range(shape[0]):
            tmp1 = []
            for j in range(shape[1]):
                tmp2 = []
                for k in range(shape[2]):
                    tmp2.append(symbols(format_txt.format(i, j, k)))
                tmp1.append(tmp2)
            q.append(tmp1)
    elif dim == 4:
        q = []
        for i in range(shape[0]):
            tmp1 = []
            for j in range(shape[1]):
                tmp2 = []
                for k in range(shape[2]):
                    tmp3 = []
                    for l in range(shape[3]):
                        tmp3.append(symbols(format_txt.format(i, j, k, l)))
                    tmp2.append(tmp3)
                tmp1.append(tmp2)
            q.append(tmp1)
    elif dim == 5:
        q = []
        for i in range(shape[0]):
            tmp1 = []
            for j in range(shape[1]):
                tmp2 = []
                for k in range(shape[2]):
                    tmp3 = []
                    for l in range(shape[3]):
                        tmp4 = []
                        for m in range(shape[4]):
                            tmp4.append(symbols(format_txt.format(i, j, k, l, m)))
                        tmp3.append(tmp4)
                    tmp2.append(tmp3)
                tmp1.append(tmp2)
            q.append(tmp1)

    return np.array(q)




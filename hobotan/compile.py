import re
import numpy as np
import sympy


#最高次数チェック
def highest_order(expr):
    # 数式から変数を取得
    variables = list(expr.free_symbols)
    # 数式を多項式としてオブジェクト化
    poly_obj = sympy.Poly(expr, *variables)
    # 多項式の各項について総合的な次数を計算し、その中で最大のものを返す
    return max(sum(mon) for mon in poly_obj.monoms())


class Compile:
    def __init__(self, expr):
        self.expr = expr

    #hoboテンソル作成
    def get_hobo(self):
        #式を展開して同類項をまとめる
        expr = sympy.expand(self.expr)
        
        #二乗項を一乗項に変換
        expr = expr.replace(lambda e: isinstance(e, sympy.Pow) and e.exp == 2, lambda e: e.base)
        
        #最高次数
        ho = highest_order(expr)
        # print(ho)
        
        #もう一度同類項をまとめる
        expr = sympy.expand(expr)
        # print(expr)
        
        #定数項をoffsetとして抽出
        offset = 0
        for ex in expr.as_ordered_terms()[::-1]:
            if 'numbers.' in str(type(ex)):
                offset = ex
                break
        # print(offset)
        
        #offsetを引いて消す
        expr2 = expr - offset
        # print(expr2)
        
        #文字と係数の辞書
        coeff_dict = expr2.as_coefficients_dict()
        
        #量子ビット数
        num = 0
        for key, _ in coeff_dict.items():
            # print(key)
            tmp = re.split('[q*]', str(key))
            tmp = [int(s) for s in tmp if len(s) > 0]
            if max(tmp) > num:
                num = max(tmp)
        num += 1
        # print(num)
            
        #HOBO行列生成コマンド
        command = 'global hobo\r\n'
        command += f'if ho == {ho}:\r\n'
        command += f'    hobo = np.zeros([{num}' + f', {num}' * (ho - 1) + '])\r\n'
        command += f'    for key, value in coeff_dict.items():\r\n'
        command += f'        tmp = str(key).split(\'*\')\r\n'
        command += f'        tmp = [int(re.sub(r\'\D\', \'\', s)) for s in tmp]\r\n'
        for i in range(1, ho + 1):
            command += f'        if len(tmp) == {i}:\r\n'
            command += f'            hobo[tmp[0]'
            j = i - (ho - 1)
            for _ in range(ho - 1):
                command += f', tmp[{max(0, j)}]'
                j += 1
            command += f'] = float(value)\r\n'
        # print(command)
        
        #HOBO生成
        exec(command)
        
        return hobo, offset
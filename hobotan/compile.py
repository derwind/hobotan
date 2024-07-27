import re
import numpy as np
import symengine
from sympy import Rational


def replace_function(expression, function, new_function):
    if expression.is_Atom:
        return expression
    else:
        replaced_args = (
                replace_function(arg, function,new_function)
                for arg in expression.args
            )
        if ( expression.__class__ == symengine.Pow):
            return new_function(*replaced_args)
        else:
            return expression.func(*replaced_args)

class Compile:
    def __init__(self, expr):
        self.expr = expr

    #hoboテンソル作成
    def get_hobo(self):
        
        #式を展開して同類項をまとめる
        expr = symengine.expand(self.expr)
        
        #二乗項を一乗項に変換
        expr = replace_function(expr, lambda e: isinstance(e, symengine.Pow) and e.exp == 2, lambda e, *args: e)
        
        #最高字数を調べながらオフセットを記録
        #項に分解
        members = str(expr).split(' ')
        
        #各項をチェック
        offset = 0
        ho = 0
        for member in members:
            #数字単体ならオフセット
            try:
                offset += float(member) #エラーなければ数字
            except:
                pass
            #'*'で分解
            texts = member.split('*')
            #係数を取り除く
            try:
                texts[0] = re.sub(r'[()]', '', texts[0]) #'(5/2)'みたいなのも来る
                float(Rational(texts[0])) #分数も対応 #エラーなければ係数あり
                texts = texts[1:]
            except:
                pass
            
            # 以下はセーフ
            # q0   ['q0']
            # q0*q1   ['q0', 'q1']
            # q0**2   ['q0', '', '2']
            
            # 以下はダメ
            # q0*q1**2   ['q0', 'q1', '', '2']
            # q0*q1*q2   [q0', 'q1', 'q2']
            # q0**2*q1**2    ['q0', '', '2', 'q1', '', '2']
            # if len(texts) >= 4:
            #     raise Exception(f'Error! The highest order of the constraint must be within 2.')
            # if len(texts) == 3 and texts[1] != '':
            #     raise Exception(f'Error! The highest order of the constraint must be within 2.')
            
            #最高次数の計算
            # ['-']
            # ['q2']
            # ['q3', 'q4', 'q1', 'q2']
            if len(texts) > ho:
                ho = len(texts)
        # print(ho)
        
        #もう一度同類項をまとめる
        expr = symengine.expand(expr)

        #文字と係数の辞書
        coeff_dict = expr.as_coefficients_dict()
        # print(coeff_dict)
        
        #定数項を消す　{1: 25} 必ずある
        del coeff_dict[1]
        # print(coeff_dict)
        
        #シンボル対応表
        # 重複なしにシンボルを抽出
        keys = list(set(sum([str(key).split('*') for key in coeff_dict.keys()], [])))
        # print(keys)
        
        # 要素のソート（ただしアルファベットソート）
        keys.sort()
        # print(keys)
        
        # シンボルにindexを対応させる
        index_map = {key:i for i, key in enumerate(keys)}
        # print(index_map)
        
        #量子ビット数
        num = len(index_map)
        # print(num)
        
        """
        #旧実装
        #HOBO行列生成コマンド
        command = 'global hobo\r\n'
        command += f'if ho == {ho}:\r\n'
        command += f'    hobo = np.zeros([{num}' + f', {num}' * (ho - 1) + '])\r\n'
        command += f'    for key, value in coeff_dict.items():\r\n'
        command += f'        tmp = str(key).split(\'*\')\r\n'
        for i in range(1, ho + 1):
            command += f'        if len(tmp) == {i}:\r\n'
            command += f'            hobo[index_map[tmp[0]]'
            j = i - (ho - 1)
            for _ in range(ho - 1):
                command += f', index_map[tmp[{max(0, j)}]]'
                j += 1
            command += f'] = float(value)\r\n'
        print(command)
        
        #HOBO生成
        exec(command)
        """
        
        #HOBO行列生成
        hobo = np.zeros(num ** ho, dtype=float).reshape([num] * ho)
        for key, value in coeff_dict.items():
            qnames = str(key).split('*')
            indices = sorted([index_map[qname] for qname in qnames])
            indices = [indices[0]] * (ho - len(indices)) + indices
            hobo[tuple(indices)] = float(value)
        
        return [hobo, index_map], offset
## インストール
```
pip install -U git+https://github.com/ShoyaYasuda/hobotan
```

## サンプルコード

メモ：GPU対応しましたが、CPUの方が圧倒的に速いです。

```python
import numpy as np
from hobotan import *

"""
多次元定義OK　[4, 4]とか
記号もq以外でもOK
"""
q = symbols_list(50, 'q{}')
print(q)

"""
何次でもOK
"""
H = (q[0] + q[10] + q[20] + q[30] + q[40] - 5)**2 - 20*q[0]*q[10] - 10*q[0]*q[10]*q[20] + 5*q[10]*q[20]*q[30]*q[40]

#HOBOテンソルにコンパイル
hobo, offset = Compile(H).get_hobo()
print(offset)
# print(hobo)

#サンプラー選択（乱数シード固定）
solver = sampler.SASampler(seed=0)
            
#サンプリング（100回）
result = solver.run(hobo, shots=100)

#結果
for r in result:
    print(r)
    arr, subs = Auto_array(r[0]).get_ndarray('q{}')
    print(arr)
```
```
[q0 q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 q11 q12 q13 q14 q15 q16 q17 q18 q19 q20
 q21 q22 q23 q24 q25 q26 q27 q28 q29 q30 q31 q32 q33 q34 q35 q36 q37 q38
 q39 q40 q41 q42 q43 q44 q45 q46 q47 q48 q49]
25.0
[{'q0': 1, 'q10': 1, 'q20': 1, 'q30': 0, 'q40': 1}, -54.0, 23]
[1 1 1 0 1]
[{'q0': 1, 'q10': 1, 'q20': 1, 'q30': 1, 'q40': 0}, -54.0, 77]
[1 1 1 1 0]
```

## 更新履歴
|日付|ver|内容|
|:---|:---|:---|
|2024/07/26|0.0.2|いろいろ修正|
|2024/07/26|0.0.1|初期版|


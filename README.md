## インストール
```
pip install -U git+https://github.com/ShoyaYasuda/hobotan
```

## サンプルコード

メモ：CPU計算だしeinsumを個体数だけforしてるから遅いはず。

```python
import numpy as np
from hobotan import *

"""
1次元配列のみ対応　[4, 4]とかはダメ
記号はqのみ対応
"""
q = symbols_list(5, 'q{}')
print(q)

"""
式はおそらく何次でもOK
"""
H = (q[0] + q[1] + q[2] + q[3] + q[4] - 5)**2 - 20*q[1]*q[2] - 10*q[1]*q[2]*q[3] + 5*q[1]*q[2]*q[3]*q[4]


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
[q0 q1 q2 q3 q4]
25
[{'q0': 1, 'q1': 1, 'q2': 1, 'q3': 1, 'q4': 0}, -54.0, 100]
[1 1 1 1 0]
```

## 更新履歴
|日付|ver|内容|
|:---|:---|:---|
|2024/07/26|0.0.1|初期版|


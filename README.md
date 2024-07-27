## インストール
```
pip install -U git+https://github.com/ShoyaYasuda/hobotan
```

## サンプルコード（CPU利用）

以下は基本的なコード。

```python
import numpy as np
from hobotan import *

#量子ビットを用意
q = symbols_list(50, 'q{}')

#式
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
25.0
[{'q0': 1, 'q10': 1, 'q20': 1, 'q30': 0, 'q40': 1}, -54.0, 23]
[1 1 1 0 1]
[{'q0': 1, 'q10': 1, 'q20': 1, 'q30': 1, 'q40': 0}, -54.0, 77]
[1 1 1 1 0]
```

## サンプルコード（GPU利用）

量子ビットは2次元配列で定義でき、定式化でq[0, 0]のように使用できる。（3次元もできるよ）

MIKASAmpler()はGPUを使用（別途pytorchをインストールしておくこと）。shots=10000に増やしても遅くなりにくいのが特徴。

以下のコードは、5✕5の席にできるだけ多くの生徒を座らせる（ただし3席連続で座ってはいけない）を解いたもの。

```python
import numpy as np
from hobotan import *
import matplotlib.pyplot as plt

#量子ビットを用意
q = symbols_list([5, 5], 'q{}_{}')

#すべての席に座りたい（できれば）
H1 = 0
for i in range(5):
    for j in range(5):
        H1 += - q[i, j]

#どの直線に並ぶ3席も連続で座ってはいけない（絶対）
H2 = 0
for i in range(5):
    for j in range(5 - 3 + 1):
        H2 += np.prod(q[i, j:j+3])
for j in range(5):
    for i in range(5 - 3 + 1):
        H2 += np.prod(q[i:i+3, j])

#式の合体
H = H1 + 10*H2

#HOBOテンソルにコンパイル
hobo, offset = Compile(H).get_hobo()
print(f'offset\n{offset}')

#サンプラー選択
solver = sampler.MIKASAmpler()

#サンプリング
result = solver.run(hobo, shots=10000)

#上位3件
for r in result[:3]:
    print(f'Energy {r[1]}, Occurrence {r[2]}')

    #さくっと配列に
    arr, subs = Auto_array(r[0]).get_ndarray('q{}_{}')
    print(arr)

    #さくっと画像に
    img, subs = Auto_array(r[0]).get_image('q{}_{}')
    plt.figure(figsize=(2, 2))
    plt.imshow(img)
    plt.show()
```
```
offset
0
MODE: GPU
DEVICE: cuda:0
Energy -17.0, Occurrence 686
[[1 1 0 1 1]
 [1 1 0 1 1]
 [0 0 1 0 0]
 [1 1 0 1 1]
 [1 1 0 1 1]]
Energy -17.0, Occurrence 622
[[1 1 0 1 1]
 [1 0 1 1 0]
 [0 1 1 0 1]
 [1 1 0 1 1]
 [1 0 1 1 0]]
Energy -17.0, Occurrence 496
[[0 1 1 0 1]
 [1 1 0 1 1]
 [1 0 1 1 0]
 [0 1 1 0 1]
 [1 1 0 1 1]]
```
<img src="https://github.com/ShoyaYasuda/hobotan/blob/main/img/img1.png" width="%">
<img src="https://github.com/ShoyaYasuda/hobotan/blob/main/img/img2.png" width="%">
<img src="https://github.com/ShoyaYasuda/hobotan/blob/main/img/img3.png" width="%">


## その他の使い方

[TYTANパッケージ](https://github.com/tytansdk/tytan) の派生形なのでそちらを参照してみてください。

## 開発㌠

derwindさん（理論）、yuminさん（マネージャー）、Shoya Yasudaさん（実装）

## 更新履歴
|日付|ver|内容|
|:---|:---|:---|
|2024/07/27|0.0.5|symbols_nbitを追加|
|2024/07/26|0.0.4|MIKASAmplerに改名|
|2024/07/26|0.0.3|テンソル計算を高速化|
|2024/07/26|0.0.2|いろいろ修正|
|2024/07/26|0.0.1|初期版|


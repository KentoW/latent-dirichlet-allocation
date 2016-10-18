# content-model
##概要
トピックモデル(latent dirichlet allocation)をPythonで実装
##lda.pyの使い方(コンテンツモデル)
```python
# Sample code.
from lda import LDA

alpha = 0.01    # 初期ハイパーパラメータalpha
beta = 0.01     # 初期ハイパーパラメータbeta
K = 10          # 隠れ変数の数
N = 1000        # 最大イテレーション回数
converge = 0.01 # イテレーション10回ごとに対数尤度を計算し，その差分(converge)が小さければ学習を終了する

lda = LDA("data.txt")
lda.set_param(alpha, beta, K, N, converge)
lda.learn()
lda.output_model()
```
##入力フォーマット
1単語をスペースで分割した1行1文書形式  
先頭に#(シャープ)記号を入れてコメントを記述可能
```
# 文書1
単語1 単語2 単語3 ...
# 文書2
単語10 単語11 単語11 ...
...
```
例として[Wiki.py](https://github.com/KentoW/wiki)を使用して収集した アニメのあらすじ文章をdata.txtに保存

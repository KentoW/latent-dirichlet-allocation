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
##出力フォーマット
必要な情報は各自で抜き取って使用してください．
```
model	lda                             # 学習の種類
@parameter
corpus_file	data.txt                    # トレーニングデータのPATH
hyper_parameter_alpha	1.834245        # ハイパーパラメータalpha
hyper_parameter_beta	0.089558        # ハイパーパラメータbeta
number_of_topic	10          # トピック数
number_of_iteration	121     # 収束した時のイテレーション回数
@likelihood         # 尤度
initial likelihood	-1389.55970144
last likelihood	-1382.11395248
@vocaburary         # 学習で使用した単語v
target_word	出産
target_word	拓き
target_word	土
target_word	吉日
target_word	遂げる
...
@count
m_z	0	15800   
m_z_v	0	の	1064
m_z_v	0	に	688
m_z_v	0	て	578
m_z_v	0	が	565
m_z_v	0	は	564
m_z_v	0	を	544
m_z_v	0	で	520
m_z_v	0	と	457
...
```

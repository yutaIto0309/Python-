import numpy as np 
import sys

class NeuralNetMLP(object):
    """
    多層パーセプトロン分類機

    パラメータ
    ----------
    n_hidden : int (デフォルト : 30)
        隠れユニットの個数
    l2 : float (デフォルト : 0.)
        L2正則化のλパラメータ
        l2=0の場合は正則化なし｛default｝
    epochs : int (デフォルト : 100)
        訓練の回数
    eta : float (デフォルト : 0.001)
        学習率
    shuffle : bool (デフォルト : True)
        Trueの場合、循環を避けるためにエポックごとに訓練データをシャッフル
    minibatch_size : int (デフォルト : 1)
        ミニバッチあたりの訓練データの個数
    seed : int  (デフォルト: none)
        重みとシャッフルを初期化するための乱数シード

    属性
    ----------
    eval_ : dict
        属性のエポックごとに、コスト、訓練の正解率、検証の正解率を収集するディクショナリ
    """
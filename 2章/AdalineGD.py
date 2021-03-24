import numpy as np

class AdalineGD(object):
    """
    ADAptive LInear NEuron 分類器
    
    パラメータ
    ----------
    eta : float
        学習率
    n_iter : int
        訓練データの訓練回数
    random_state : int
        重みを初期化するための乱数シード

    属性
    ----------
    w_ : 1次元配列
        適合後の重み
    cost_ : リスト
        各エポックでの誤差平方和のコスト関数

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        訓練データを適合させる

        パラメータ
        ----------
        X : {配列のようなデータ構造}, shape = [n_examples, n_features]
            訓練データ
            n_examplesは訓練データの個数, n_featuresは特徴量の個数
        Y : 配列のようなデータ構造, shape = [n_examples]
            目的変数

        戻り値
        ---------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            """
            activationメソッドは単なる恒等関数であるため、
            このコードでは何の意味もないことを注意。代わりに
            直接'output = self.net_input(X)'と記述することもできた
            activationメソッドの目的は、より概念的なものである。
            つまり、ロジスティック回帰の場合は、
            ロジスティック回帰の分類器を実装するためにシグモイド関数に変更することもできる
            """
            output = self.activation(net_input)
            # 誤差の計算
            errors = (y - output)
            # 重みの更新
            # Δwj = ηΣi(yi - φ(zi))xj(j = 1,...,m)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # コスト関数の計算
            cost = (errors**2).sum / 2.0
            self.cost_.append(cost)

        return self


    def net_input(self, X):
        """
        総入力を計算
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """
        線形活性化関数の出力を計算
        """
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


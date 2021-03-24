class Perceptron(object):
    """
    パーセプトロン分類機

    パラメータ
    ------------
    eta : float
        学習率（0.0 より大きく1.0以下の値）
    n_iter : int
        訓練データの訓練回数
    random_state : int
        重みを初期化するための乱数シード

    属性
    ------------
    w_ : 1次元配列
        適合後の重み
    errors_ : リスト
        各エボックでの後分類（更新）の数
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        訓練データに適合させる

        パラメータ
        ------------
        X : {配列のようなデータ構造}, shape = [n_examples, n_features]
            訓練データ
            n_examplesは訓練データの個数、n_featuresは特徴量の個数
        y : 配列のようなデータ構造、 shape = [n_examples]
            目的変数

        戻り値
        ------------
        self : object 
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        # 訓練回数まで訓練データを反復
        for _ in range(self.n_iter):
            errors = 0
            for xi, target, in zip(X,y):
                # 重みの更新
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                # 重みの更新が0出ない場合は誤分類としてカウント
                errors += int(update != 0.0)
            self.errors_.append(errors)

    def predict(self, X):
        """
        1ステップ前のクラスラベルを返す
        """
        net_input = np.dot(X, self.w_[1:]) + self.w_[0]
        return np.where(net_input >= 0.0, 1, -1)


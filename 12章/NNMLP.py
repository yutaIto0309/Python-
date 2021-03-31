import numpy
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

    def __init__(self, n_hidden=30, l2=0., epochs=100, eta=0.001, shuffle=True, minibatch_size=1, seed=None):
        """
        初期化
        """
        self.random = numpy.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """
        ラベルをone-hot表現にエンコード
        
        パラメータ
        ----------
        y : array, shape = [n_examples]
            目的変数の値

        戻り値
        ----------
        onehot : array, shape = {n_examples, n_labels}
        """

        onehot = numpy.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        
        return onehot.T

    def _sigmoid(self, z):
        """
        ロジスティック関数を計算
        """
        return 1. / (1. + numpy.exp(-numpy.clip(z, -250, 250)))

    def _forward(self, X):
        """
        フォワードプロバゲーションのステップ計算
        """

        # ステップ1 : 隠れ層の総入力
        # [n_examplesm, n_features] dot [n_features, n_hidden]
        # -> [n_examples, n_hidden]
        z_h = numpy.dot(X, self.w_h) + self.b_h

        # ステップ2 : 隠れ層の活性化関数
        a_h = self._sigmoid(z_h)

        # ステップ3 : 出力層の総入力
        # [n_examples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_examples, n_classlabels]
        z_out = numpy.dot(a_h, self.w_out) + self.b_out

        # ステップ4 : 出力層の活性化関数
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """
        コスト関数を計算

        パラメータ
        ----------
        y_enc : array, shape = (n_examples, n_labels)
            one-hot表現にエンコーとされたクラスラベル
        output : array, shape = [n_examples, n_output_units]
            出力層の活性化関数(フォーワードプロバゲーション)

        戻り値
        ----------
        cost : float
            正則化されたコスト
        """

        L2_term = (self.l2 * (numpy.sum(self.w_h ** 2) + numpy.sum(self.w_out ** 2.)))
        term1 = -y_enc * (numpy.log(output))
        term2 = (1. - y_enc) * numpy.log(1. - output)
        cost = numpy.sum(term1 - term2) + L2_term

        return cost

    def predict(self, X):
        """
        クラスラベルを予測

        パラメータ
        ----------
        X : array, shape = [n_examples, n_features]
            元の特徴量が設定された入力層

        戻り値
        ----------
        y_pred : array, shape = [n_examples]
            予測されたクラスラベル
        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = numpy.argmax(z_out, axis=1)

        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        訓練データから重みを学習

        パラメータ
        ----------
        X_train : array, shape = [n_examples, n_features]
            元の特徴量が設定された入力層
        y_train : array, shpae = [n_examples]
            目的変数(クラスラベル)
        X_valid : array, shape = [n_examples, n_features]
            訓練時の検証に使うサンプル特徴量
        y_valid : array, shape = [n_examples]
            訓練時の検証に使うサンプルラベル

        戻り値
        ----------
        self
        """

        # クラスラベルの個数
        n_output = numpy.unique(y_train).shape[0]
        n_features = X_train.shape[1]

        # 重みの初期化
        # 入力層 -> 隠れ層の重み
        self.b_h = numpy.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))

        # 隠れ層 -> 出力層の重み
        self.b_out = numpy.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))

        # 書式設定
        epoch_stlen = len(str(self.epochs))
        self.eval_ = {
            'cost' : [], 
            'train_acc': [],
            'valid_acc':[]
            }

        y_train_enc= self._onehot(y_train, n_output)

        # エポック数だけ訓練を繰り返す
        for i in range(self.epochs):
            # ミニバッチの反復処理
            indices = numpy.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # フォーワードプロバゲーション
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                # バックプロバゲーション
                # [n_examples, n_classlabels]
                delta_out = a_out - y_train_enc[batch_idx]

                # [n_examples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_examples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_examples, n_hidden]
                delta_h = (numpy.dot(delta_out, self.w_out.T) * sigmoid_derivative_h)
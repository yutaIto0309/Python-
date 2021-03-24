import numpy as np
from numpy.random import seed

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
            cost = (errors**2).sum() / 2.0
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

class AdalineSGD(object):
    """
    ADAptive LInear NEuron 分類器

    パラメータ
    ----------
    eta : float
        学習率 (0.0より大きく1.0以下の値)
    n_iter : int
        訓練データの訓練回数
    shuffle : bool(デフォルト : True)
        Trueの場合は、循環を回避するためにエポックごとに訓練データをシャッフル
    random_state : int
        重みを初期化するための乱数シード

    属性
    ----------
    w_ : 1次元配列
        適合後の重み
    cost_ : リスト
        各エポックで全ての訓練データの平均を求める誤差平方和コスト関数
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """
        訓練データに適合させる
        
        パラメータ
        ----------
        X : {配列のようなデータ構造], shape = {n_examples, n_features}
            n_examplesは訓練データの個数、n_featuresは特徴量の個数
        y : 配列のようなデータ構造、shape = [n_examples]
            n_examplesは目的変数の個数

        戻り値
        ----------
        self : object        
        """
        # 重みベクトルの生成
        self._initialize_weights(X.shape[1])
        # コストを格納するリストの生成
        self.cost_ = []
        # 訓練回数分まで訓練データ反復
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self.shuffle(X, y)
            # 各訓練データのコストを格納するリストの生成
            cost = []
            # 各訓練データに対する計算
            for xi, target in zip(X, y):
                # 特徴量xiと目的変数yを用いた重みの更新とコストの計算
                cost.append(self._update_weights(xi, target))
            # 訓練データの平均コストの計算
            avg_cost = sum(cost) / len(y)
            # 平均コストを格納
            self.cost_.append(avg_cost)

    def partial_fit(self, X, y):
        """
        重みを再初期化することなく訓練データに適合させる
        """
        # 初期化されていない場合は初期化を実行
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        # 目的変数yの要素数が2以上の場合は、各訓練データの特徴量xiと目的変数tartetで重みを更新
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        return self

    def _shuffle(self, X, y):
        """
        訓練データをシャッフル
        """
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """
        重みを小さな乱数に初期化
        """
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.1,size= 1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """
        ADALINEの学習規則を用いて重みを更新
        """
        # 活性化関数の出力の計算
        output = self.activation(self.net_input(xi))
        # 誤差の計算
        error = (target- output)
        # 重みを更新
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        # コストの計算
        cost = 0.5 * error**2
        return cost

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
        """
        1ステップ後のクラスラベルを返す
        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
#%% アイリスデータセットの読み込み
import os
import pandas as pd 
s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data')
df = pd.read_csv(s, header=None, encoding='utf-8')
df.tail()
#%% 
import matplotlib.pyplot as plt

# 目的変数の抽出
y = df.loc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# 2つの特徴量を抜き出す
X = df.loc[0:100, [0,2]].values

# アイリスデータをプロットする
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100,1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
#plt.show()
# %% パーセプトロンの後分類をプロット
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.close()
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of update')
# %%
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # マーカーとカラーマップの準備
    markers = ('s', 'x', 'o', '^', 'v')
    colors  = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:,1].max() + 1
    # グリッドポイントの生成
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min,x2_max, resolution))

    # 各特徴量を1次元配列に変換して予測を実行
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 予測結果を元のグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    # グリッドポイントの等高線プロット
    plt.contourf(xx1,xx2,Z, alpha=0.3, cmap=cmap)
    # 軸の範囲設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとに訓練データをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],y=X[y == cl, 1],alpha=0.8,
                    c = colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

# %%
# 決定領域のプロット
plot_decision_regions(X, y, classifier=ppn)
# 軸ラベルの設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# 凡例の設定
plt.legend(loc='upper left')
# 図の表示
#plt.show()
# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ada1 = AdalineGD(eta=0.01, n_iter=10).fit(X,y)
ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')
ada2 = AdalineGD(eta=0.0001, n_iter=10).fit(X,y)
ax[1].plot(range(1, len(ada2.cost_)+1), np.log10(ada2.cost_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
# %%
# データのコピー
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
# 勾配効果法によるADALINEの学習(標準化後、学習率eta=0.01)
ada_gd = AdalineGD(eta=0.01, n_iter=15)
ada_gd.fit(X_std, y)
# 境界線のプロット
plot_decision_regions(X_std, y, classifier=ada_gd)
# タイトルの設定
plt.title('Adaline - Gradient Descent')
# 軸のラベルの設定
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
# 判例の設定 
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
# エポック数とコストの関係を表す折線グラフのプロット
plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
# 軸のラベルの設定
plt.xlabel('Epoches')
plt.ylabel('Sum-squared-error')
plt.tight_layout()
plt.show()
# %%
# 確率勾配降下法によるADALINEの学習
ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
# モデルへの適合
ada_sgd.fit(X_std, y)
# 境界領域のプロット
plot_decision_regions(X_std, y, classifier=ada_sgd)
# タイトルの設定
plt.title('Adaline - Stochastic Gradient Descent')
# 軸ラベルの設定
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
# 判例の設定(左上に配置)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
# エポックとコストの折線グラフのプロット
plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
# 軸ラベルの設定
plt.xlabel('Epochs')
plt.ylabel('Avarage Cost')
# プロットの表示
plt.tight_layout()
plt.show()
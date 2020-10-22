# 機器學習
* 是經過資料科學的運算和演算技巧加上統計學的思考方式所導出的一個研究集合，用來推論和進行資料探索些抽象的事物

## 基本定義

### 監督式學習(supervised learning)
* 基於已經標上標籤的資料建立模型來預測標籤data

### 分類(classification)
* 建立可以把2個或是更多的獨立的類別標上標籤的模型

### 迴歸(Regression)
* 建立可以用來預測連續標籤的模型

### 非監督式學習(Unsupervised learning)
* 建立在未標上標籤的資料中識別出結構的模型

### 集群(clustering)
* 建立在資料中偵測並識別出不同的群組的模型

### 維度降低(Dimensionality reduction)
* 建立在比較高維度的資料中可以偵測及識別出較低維度結構的模型

### holdout set
* 把一些資料的子集合從此模型的訓練資料中先暫時留下來，然後使用holdout set去檢查模型的效能(train_test_split)

### cross-validation


## scikit-learn一些package用法

### 降維演算法
* 在降維演算法中，一種方式是提供點的座標進行降維(PCA)
* 另一種方式是提供點之間的距離矩陣(Isomap -> MDS(Multidimensional Scaling))
* MDS在計算兩點間的距離是透過 `歐式距離`

#### Isomap(流行學習 manifold learning algorithm)
* 是一種非監督學習的降維度演算法
* 非線性的結構
* 基於MDS演算法延伸出來的
* 兩個點的距離為圖中兩點的最短路徑，然後採用內積進行推導 --> 能更好的擬和流行體數據
<details>
 <summary>演算法流程</summary>
    <p> 1. 設定每個點最近鄰點數k，建構連通圖和鄰接矩陣
    <p> 2. 通過圖的最短路徑建構原始空間中的距離矩陣
    <p> 3. 計算內積矩陣(B)
    <p> 4. 對距離B進行特徵值解析，獲得特徵值矩陣和特徵向量矩陣
    <p> 5. 取特徵值距離最大的前項目及其對應的特徵向量
</details>

```python
from sklearn.manifold import Isomap
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data

iso = Isomap(n_components=2)
data_projected = iso.fit_transform(X)
print(data_projected.shape)
```

#### PCA
* 是典型的線性降維

### 梯度下降分類器

#### 批量梯度下降法
* 每次反覆運算使用所有樣本，這樣的好處是每次反覆運算都顧及了全部的樣本，考慮的是全域最優化
* 缺點: 每次反覆運算都要計算訓練集中所有樣本的訓練誤差，`當資料很大時，效率很差`

#### 隨機梯度下降分類器
* 每次反覆運算都隨機從訓練集中抽取`1個樣本`，在樣本量極大的情況下，可能不用取出所有樣本，就可以獲得一個損失值在可接受範圍內的模型了
* 缺點: 由於`單個樣本`可能會帶來雜訊，導致不是每次反覆運算都項著整體最佳的方向前進

##### SGD(SGDClassifier)
* 是一個簡單有效的方法，去判斷使用凸loss函數(convex loss function)的分類器(SVM或logistic迴歸)
* 應用在大規模稀疏機器學習問題上，用於文本分類及自然語言處理上
* 優點:
    * 高效
    * 容易實現(有許多機會進行調參)
* 缺點
    * 需要許多超參數:像是正則項參數、反覆運算數
    * 對特徵歸一化(feature scaling)是敏感的
* 注意: 再進行模型fit前必須重排(permute/shuffle)訓練集，或者在每次反覆運算後使用shuffle=True進行shuffle

```python
from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, y)
clf.predict([[2., 2.]])  # > array([1])
clf.coef_  # > array([[9.85221675, 9.85221675]])

```


#### 小批量梯度下降法
* 介於上面兩者之間，每次隨機從訓練集`抽取一定數量的數據`進行訓練

### 基本步驟
1. 選擇一個模型的類別
2. 選用模型的超參數
3. 擬和模型到訓練資料
4. 使用模型對新資料進行標籤預測

## 流行學習(Manifold Learning)
* 


## 參考文件
* [梯度下降(SGDClassifier)](https://zhuanlan.zhihu.com/p/60983320)
* [梯度下降2(SGDClassifier)](https://www.jianshu.com/p/28a68bb4a45a)
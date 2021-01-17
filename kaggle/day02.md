### day02 筆記

1. Numeric Features
    - 使用Feature Scaling 特徵縮放避免在數字差異很大時被牽著跑，正規化很重要

###  Feature Scaling
- MinMaxScaler --> To[0,1]
    > sklearn.preprocessing.MinMaxScaler
    - X=(X-X.min())/(X.max()-X.min())
- StandardScaler --> To mean=0, std=1
    > sklearn.preprocessing.StandardScaler
    - X=(X-X.mean())/X.std()
- rank <-- 適用在有 outlier 時
    > scipy.stats.rankdata
    - rank=([-100, 0, 1e5]) == [0,1,2]
    - rank([1000,1,10]) = [2,0,1]


1. 特徵縮放的 Scaling, Rank 用在數值型特徵 numeric features 有兩件事要注意.
    a. Tree-based models 不適用
    b. Non-tree-based models 適用
2. 常用到的預處理
    a. MinMaxScaler - to [0,1]
    b. StandardScaler - to mean==0, std==1
    c. Rank
   d. np.log(1+x) and np.sqrt(1+x)-->今天沒講到
3. 厲害的特徵生成能力來自於
   a. Prior knowledge(先驗知識) -> 房屋仲介的例子，資料集給價格跟面積 -> 價格/面積
   b. EDA / Exploratory data analysis
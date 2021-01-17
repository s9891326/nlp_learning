### day04 特徵預處理及生成 Feature Preprocessing and Generation

2. Ordinal Feature
    - 建議用tree-based model樹型，比較好找依存關係
    - Alphabetical (sorted/依照字母排序)
        - [S、C、Q] -> [2, 1, 3]
        - sklearn.preprocess.LabelEncoder
    - Order of appearance (依出現順序)
        - [S、C、Q] -> [1, 2, 3]
        - Pandas.Factorize
    -  Frequency encoding (依頻率)
        - [S、C、Q] -> [0.5, 0.3, 0.2]
        - from SciPy.stats import t=rankdata
3. Catergoical Feature
    - One-hot encoding
        - 適用 non-tree-based-models(KNN, NN)
        - One-hot encoding 已scaling/縮放到[0,1]
[one-hot.png](one-hot.png) 
    - Feature generation
        - 針對特徵萃取比較好用的方法是直接將特徵間互動建立新特徵, 通常用在 non-tree-based-models, linear model, KNN 等
[feature-generation.png](feature-generation.png) 
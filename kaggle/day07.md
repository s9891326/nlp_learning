### day07 Word2vec, CNN

- Word2Vec: 把文字變向量
    - 例如: king + woman - man = queen
    - [w2v_bag.png](w2v_bag.png)

- image -> Vector
- CNN(convolutional neural network)
- Finetuning example, 擴增(image augmentation)
    - 加上90度照片，樣本數增加四倍

#### Text
1. 預處理
     - Lowercase、stemming、lemmarization、stopwords
2. 詞袋/Bag of words
    - 大向量
    - Ngram可以用在本地文檔
    - TFIDF-可用在預處理階段
3. Word2Vec
    - 相對小向量
    - pretrained models(預訓練模型)

#### Images
1. 特徵提取可依照不同 layers
2. 仔細挑選 pretrained models
3. pretrained models 也可 finetuning
4. 資料擴增可以改善 model

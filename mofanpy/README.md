# 莫凡NLP教學筆記

## 章節
* [TF-IDF](#tf-idf)
* [詞向量](#詞向量)
* [CBOW](#continuous bag-of-word)
* [TF-IDF](#TF-IDF)
* [ELMO](#ELMO)

### TF-IDF(Term Frequency - Inverse Document Frequency)
* 主要的功能 : 使用詞語的重要程度與獨特性來`代表`每篇文章，然後透過對比`搜索詞`與`代表`的相似性，給你最相似的文章列表
* TF(詞頻) : 
* IDF(逆文本頻率指數) :  
* 舉例來說: 想像你是新手房仲，你面前有100篇文章資料，當有客戶來詢問時，要怎麼快速找到某一篇文章呢? 
我們當然會想要使用某種方式將其歸類，甚麼方式適合呢?你會不會去找每篇文章中的關鍵字?那些在文章中出現頻率很高的詞，比如說"租屋"、"二手房"
等等，這些高頻的字詞其實就代表著這篇文章的屬性
* 我們可以從很多文章中歸納出各篇文章所代表的意思，方便我們快速尋找要找的文章，但是光看各篇的局部訊息(某篇文章中的詞頻TF)，會有統計偏差，
所以需要引入一個全局參數(IDF)，來判斷這個詞在所有文章中是不是垃圾訊息。藉此把局部(TF)和全局(IDF)的訊息整合再一起看，我們就可以快速找到文章

## 詞向量
* 是詞語的向量表示，我們的電腦最擅長使用這種數字化向量表示來計算和理解詞與。所以詞向量對於理解詞語甚至是句子都有很強的適用性。

- 詞向量
    - 把這些對於詞與理解的向量通過特定方法組合起來，就可以有對某句話的理解了
    - 可以在向量空間中找尋同義詞，因為同義詞表達的意思相近，往往在空間中距離也非常近
    - 詞語的距離換算

## CBOW(Continuous Bag-of-Word)
* 一句話概述 : 挑一個要預測的詞，來學習這個詞`前後文中詞語`和`預測詞`的關係
* 使用上下文來預測下文之間的結果
* 句子是由詞語組成的，就是將這個句子中所有詞語的詞向量都加起來，然後就變成句子的解釋了?
    * 問題點 : 這種空間上的向量相加，從直觀上理解，就不是特別成立，因為他加出來以後，還是在這個詞彙空間中的某個點，你說他是句向量嗎?好像也不是，說他是一個詞的理解嗎?好像也不對
    * 解決辦法 : 將這些訓練好的詞向量當作預訓練模型，然後放入另一個神經網路(像是RNN)當成輸入，使用另一個神經網路加工後，訓練句向量
<details>
<summary>模型輸入輸出</summary>

    # 1
    # 输入：[我,爱] + [烦,Python]
    # 输出：莫
    
    # 2
    # 输入：[爱,莫] + [Python, ，]
    # 输出：烦
    
    # 3
    # 输入：[莫,烦] + [，,莫]
    # 输出：Python
    
    # 4
    # 输入：[烦,Python] + [莫,烦]
    # 输出：，
</details>


## Skip-Gram(skip-gram)
* 與CBOW相反，使用文中的某個詞，然後預測這個詞周邊的詞
* skip-gram 和 CBOW最大的不同，就是剔除掉了中間的那個SUM求和的過程
* 是一種提取詞向量的過程，有了這些詞向量，我們可以用在對鋸子的理解、對詞語的匹配等後續NLP流程中。
- 能不能更好?
    - 我們已經訓練出詞向量了，不過在生活中，一定會遇到一詞多義的情況。用skip-gram和cbow無法解決這類的問題，因為他們會針對每一個詞生成唯一的詞向量，也就是說這個詞只有一個解釋(向量表達)
    - 範例 : `2`
        - 我住在`2`號房
        - 高鐵還有`2`站就到了
    - 真實情況
        - 我住在`二`號房
        - 高鐵還有`兩`站就到了
    - 同樣是2，但兩個2意思不一樣，一個是房間號碼，一個是數字
    - 有什麼辦法讓模型表達出詞語的不同含意呢?
        - 當然我們還是站在向量的角度，只是這個詞向量的表達如果能考慮到句子上下文的訊息，那麼這個詞向量就能表達詞語在不同句子中的涵義了。 `ELmo`模型中，能探討這做法

## Seq2Seq Attention

## Transformer

## ELMO (Embeddings from Language Models)
- 目標：找出詞語放在句子中的意思
- 運用兩個RNN(LSTM)的概念，來計算出詞在句子中代表甚麼意思，透過句向量(一個由前往後，一個由後往前) + 詞向量

```
step:  0 | time: 1.52 | loss: 9.463
| tgt:  <GO> hovan , a resident of trumbull , conn . , had worked as a bus driver since <NUM> and had no prior criminal record . <SEP>
| f_prd:  atsb knew knew competition competition competition competition markup floors festivals merit merit merit korkuc korkuc korkuc fingerprinting grade grade car car nicky roush thoughts roush gain gain
| b_prd:  stockwell stockwell stockwell mta mta mta mta mta mta mta tornadoes tornadoes tornadoes router router halliburton halliburton talked engaged ona db2 life rashid rashid ursel ursel

step:  9920 | time: 8.36 | loss: 0.543
| tgt:  <GO> of personal vehicles , <NUM> percent are cars or station wagons , <NUM> percent vans or suvs , <NUM> percent light trucks . <SEP>
| f_prd:  the the vehicles , <NUM> percent are cars or station wagons , <NUM> percent vans or suvs , <NUM> percent light trucks . <SEP>
| b_prd:  <GO> of personal vehicles , <NUM> percent are cars or station wagons , <NUM> percent vans or suvs , <NUM> percent light <NUM> .
```

## GPT

## BERT

##


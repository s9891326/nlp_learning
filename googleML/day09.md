### day09

#### Generalization bounds
- 模型的複雜度
- training data的效率
- 好的generalization應該符合下面三個假設:
    1. independently and identically (i.i.d) at random: 隨便選資料，資料不會互相影響，而且有同個distribution
    2. 分布是靜止的: 同個資料set內的分布不會改變
    3. 每個partition的example都有相同的distribution

#### Generalize
- 避免模型「背下資料集」導致overfitting，拿模型沒看過的新資料看性能如何
- 泛化(generalize)：基本上就是指ML模型對未知資料集的預測能力。
- 如果沒有良好的泛化(generalize)，表示我們的模型只對我們的資料集預測好而已。
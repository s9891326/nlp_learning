### day 21 Multi-class Neural Networks

- softmax -> Full Softmax (class 數量少時適用)
- Candidate sampling (class 數量多時比較有效率)，僅針對positive label算所有的機率，並對隨機選幾個negative labels算機率。在考慮品種的時候可以使用，考慮'狗的品種'不需要把所有'貓'的機率算下去
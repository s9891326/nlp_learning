### day06

- y' = w1x + b
一開始b、w1都喜歡設定成0
- y' = 0 * x + 0
- loss function = squared loss = (y - y') ^ 2 = (y - 0) ^ 2 = y ^ 2

#### Gradient Descent(梯度下降)
- 定義w1 = 0的初始值 => loss = 0
- 把原本的預測函數微分(斜率)，越大就越往前 or 後一點，越小就往前 or 後小一點，也就是說它包含了方向及大小的資訊在裡面
- [gradient_descent.png](gradient_descent.png) 

#### Learning Rate(step size)，又稱Hyperparameters，決定箭頭的長短，小一點就是紅色箭頭，大一點就是綠色箭頭
- w = w - LearningRate * gradient(w)
- [learn_rate.png](learn_rate.png) 
- [動手玩一下](https://developers.google.com/machine-learning/crash-course/fitter/graph)

#### Gradient Descent Algorithm
- Stochastic Gradient Descent (SDG)
    - 一次挑一個example，如果資料量大的話會跑很久才能train完一個model
- Batch Gradient Descent
    - 一次把所有example餵進去，這跑一次要跑很久
- Mini-batch stochastic gradient descent(mini-batch SGD)
    - 一次餵少部分的example去train，平均每個loss算出一個W

- 每個都是Machine Learning的大課題，要找初始值、Learning rate、batch-size，大家都盡可能的優化每個步驟，以至於優化整個Learning process 
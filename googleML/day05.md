### day05

- 預測溫度與蟋蟀叫聲之間"是否"有關係
**y = mx + b**
y: 溫度(預測項目)
m: 最後線的斜率
x: 蟋蟀叫聲次數
b: y截距(偏移量)

- 替換變數 -> y = w1x + w0

#### loss 函數
- loss - Squared loss == L2 loss
= (observation - prediction(x)) ^ 2
= (y - y') ^ 2

- Mean square error(MSE): 把所有dataset裡的squared loss 相加後平均
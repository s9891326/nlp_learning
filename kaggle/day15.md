### day15 Metrics / 評估指標 - Regression metrics

- Regression metrics
    - MSE，RMSE，R-squared
        - MSE : Mean Square Error / 均分誤差，又稱L2損失，計算方式是求預測值與真實值之間距離的平方和，由於有平方, 懲罰偏離真實比較重, __因此適合梯度計算__
[MSE](MSE.png) 
        - RMSE : Root Mean Square Erro / 均方根誤差
[RMSE](RMSE.png) 
        - R-squared
[R-squared](R-squared.png) 
    - MAE : Mean Absolute Error / 平均絕對誤差，常用在財務方面，又稱L1損失，將每次測量的絕對誤差取絕對值後球的平均值
[MAE](MAE.png) 
    - (R)MSPE，MAPE
[MSE->MSPE，MAE->MAPE](MSE-MSPE%20MAE-MAPE.png) 
    - (R)MSLE
[RMSLE](RMSLE.png)
    - tips:
        1. 若想要知道資料中有沒有 outliers --> MAE
        2. 想要確定某些是否 outliers --> MAE
        3. 若只是沒想到的數值但其實我們要留意時 --> MSE
- Classification
    - Accuracy，LogLoss，AUC
        - Accuracy ex:
            - Dataset : 10 cakes, 90 eggs
            - Predict always eggs : accuracy = 0.9!
        - Logarithmic loss(logloss) ex:
            - Dataset : 10 cakes, 90 eggs
            - α = [0.1, 0.9]
        - Area Under Curve(AUC ROC)
        [AUC](AUC.png) 
    - Cohen's (Quadratic weighted) Kappa 
        - Cohen's (Quadratic weighted) Kappa ex:
            - Dataset : 10 cakes, 90 eggs
            - Baseline accuracy=0.9
            - predict 20 cakes and 80 eggs at random : accuracy ~0.74
            - 0.2 * 0.1 + 0.8 * 0.9 = 0.74
            - error ~ 0.26
           [Cohen's Kappa motivation](Cohen's%20Kappa%20motivation.png) 
- Regression
    - RMSE, MSE, R-squared
    - MAE
    - MSPE as weighted MSE; MAPE as weighted MAE
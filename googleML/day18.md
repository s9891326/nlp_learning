### day18 Regularization - Sparsity

L1 vs L2 Regularization
- 處罰項目的差異：
    - L2 處罰 weight^2
    - L1 處罰 |weight|
- 微分後的意義：
    - L2 微分是 2 * weight，永遠不會讓weight為0，只會很接近0
    - L1 微分後是個常數k(跟weight沒關係)，因為絕對值得特性，它會在0的時候中斷下來，就是在此時把這個weight zeroed out。
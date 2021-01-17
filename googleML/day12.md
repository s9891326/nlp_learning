### day12 Representation

- 機器學習分成三個主要部分
    - 表現(Representation)
    - 最佳化(Optimization)
    - 評估(Evaluation)
- Qualities of Good Features
    - Avoid rarely used discrete feature values:
        - 同個Feature value出現五次以上最好，可以看到同個feature value而其他參數不一樣時，對model的影響。可以想像unique_id 只會出現一次，就是個不好的feature.
    - Prefer clear and obvious meanings: 
        - 一些打錯字的、單位跟一般不一樣的data濾掉，兩百歲正常嗎？年齡用毫秒計算正常嗎？
    - Don't mix "magic" values with actual data
        - 過濾一些奇怪的設定值，可以多開一個feature代表。金額-1代表未上架？多開一個是否上架is_launched吧。
    - Account for upstream instability
        - 不要用一些不穩定的值，city: 'taipei'很穩定， city: 0每個程式對city 0的定義很難一致。
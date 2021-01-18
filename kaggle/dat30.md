# day30

- Before you enter a competition
    - Importance : 整理並排序參數，從最重要到最無用的， 會跟data structure、target、metric有關
    - Feasibility :  評估最容易調的參數到那種要調到天荒地老的
    - Understanding : 評估從自己最了解到完全不瞭解的參數，因為不管接下來要增加特徵或改變占比比率，或CNN的階層數量，理解並掌握參數的狀況是很有用的 (changing one parameter can affect the whole pipeline)
- Data loading
    - 基本格式處理labeling、codingcategory recovery、，然後將csv/txt files轉成hdf5/npy以加速上傳/下載時間 (hdf5 for pandas, npy for non-bit arrays)
    - 因為 Pandas 預設 64-bit 陣列儲存, 其實不必要, 可以轉成 32 位元以節省記憶體資源
    - Large datasets can be processed in chunks - 數據集大的可以切塊, Pandas 可以支援將資料即時連接起來, 不然跑一次大筆完整資料就又要等到天荒地老
- Performance evaluation
    - 不一定每次都要用到cross validation，有時基本作法切成train/test就足夠
    - Start with fastest models - 這邊是推薦用 LightGBM 去找到好特徵跟快速評估特徵的表現，early stoping是在這時 (初期)可以採用的, 只有在特徵工程後才會調整模, 用 stacking, NN..等
    - Fast and dirty always better - 意思是專注在最重要的，就是探索資料本身，用EDA去挖掘不同的特徵，找出產業或該項專業的domain-specific knowledge，語法在此刻是次重要事項
    - EDA 很重要 !!! 如果一直處於很擔心電腦資源跑不動, 就去租伺服器吧
- Initial pipeline
    - 從簡單甚至原始的解決方案開始
    - 建構自己的pipeline，重點不是建模而是解決方案的整體架構設計，包括初期資料處理，預測結果的檔案格式寄送，在競賽資料的kernel或organizer提供的，都可以找到baseline解決方案
    - "From simple to complex" : 從簡單到複雜, 寧願從Random Forest開始, 而不是Gradient Boosted Decision Trees，再一步步到比較複雜的方法
- Best Practices from Software Development
    - Use good variable names : 使用好的變數名字，不然到最後自己也搞混
    - Keep your research reproducible / 保持所有東西都可重製. 包括 : Fix random seed ; 寫下所有特徵生成的過程 ; 使用版控 (VCS, for example, git)
    - Reuse code / 使用同樣的程式碼，特別在 train/test 要使用相同程式碼，前後才能一致，因此建議另外存起來
    - Read papers : 
        1. 可以技術方面的專業知識，例如 how to optimize AUC; 
        2. 可以了解問題本質，尤其在幫助找到特徵，例如微軟的手機競賽，就是讀了手機相關papaer才可以增加跟找到好特徵
- Code organization
    - keeping it clean
    - test/val
    - macros
    - custom library
- pipeline
    - Read forums and examine kernels first : There are always discussions happening!
    - Start with EDA and a baseline : 
        1. To make sure the data is loaded correctly
        2. To check if validation is stable
    - I add features in bulks
        1. At start I create all the features I can make up
        2. I evaluate many features at once (not "add one and evaluate")
    - Hyperparameters optimization
        1. First find the parameters to overfit train dataset
        2. And then try to trim model

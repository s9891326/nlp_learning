### day05 Datetime Feature 與 Coordinate Feature

1. Datetime Feature
    1. 週期性
    2. 自特定(事件/活動)時間點起
    3. 某段時間差

2. Coordinate Feature (座標)
    1. 經計算而來的距離 - 通常可以從地圖上計算到重要點距離, 若有基礎設施建築物的額外數據, 就可以加距離最近的商店, 到醫院, 到附近最好的學校等
    2. cluster 中心點
    3. 整合/匯總/聚合統計

3. Handle missing values
    1. -999, -1, etc - missing values 直接替換成不在feature值域範圍內的數，-999，-1等等
    2. missing values替換成mean或者median value
    3. Reconstruction value(重建值) - 可加一個 Binary feature 的 isnull feature，標記每個feature在每一行是否missing value。
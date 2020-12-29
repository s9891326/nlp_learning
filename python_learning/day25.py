import bisect

# 二元搜尋法
# 時間複雜度為O(logN)。(log這邊是以2為底的)

lt = [1, 3, 3, 3, 5, 8, 9]
print(bisect.bisect_left(lt, 3))  # 同值的最左邊

print(bisect.bisect_right(lt, 3))  # 同值的最右邊

bisect.insort(lt, 3)   # 實際插入

print(lt)

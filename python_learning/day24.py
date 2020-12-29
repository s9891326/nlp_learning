import heapq
from typing import List

lt = [2, 7, 4, 1, 8, 1]

# heapq.heapify(lt)
# print(lt)

# heapify (將一個list轉為heap)
# heappush/heappop/heappushpop (放入/取出/先放入後取出)
# nlargest/nsmallest (取前n大/前n小的元素)
# heap類型也適用於限縮個數的狀況。

class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        h = [-x for x in stones]
        heapq.heapify(h)
        while len(h) > 1:
            y = heapq.heappop(h)
            x = heapq.heappop(h)

            if y != x:
                heapq.heappush(h, y - x)
        if len(h) == 0:
            return 0
        else:
            return h[0] * -1

if __name__ == '__main__':
    # solution = Solution()
    # print(solution.lastStoneWeight(lt))
    heapq.heapify(lt)
    print(lt)  # [1, 1, 2, 7, 8, 4]
    heapq.heappushpop(lt, 5)
    print(lt)  # [1, 5, 2, 7, 8, 4]
    heapq.heappushpop(lt, 5)
    print(lt)  # [2, 5, 4, 7, 8, 5]

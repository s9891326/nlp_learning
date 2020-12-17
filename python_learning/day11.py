from collections import defaultdict, OrderedDict
from collections import Counter
from collections import deque
# defaultdict
dic = {}
# print(dic["XD"])  # KeyError: 'XD'

""" -> 1. if key in dic """
""" -> 2. get() """
# print(dic.get("c", 0))

""" -> 3. defaultdict() """
cnt = defaultdict(int)
# print(cnt[3])
# print(cnt["xd"])


"""Counter"""
scores = [35, 70, 10, 20, 35, 70, 70]
score_counter = Counter(scores)
# print(score_counter)

names = ["James", "Michael", "Ted", "James", "Leo"]
# print(Counter(names))

# print(score_counter.most_common())  # most_common()會按由大到小的出現次數來排序
# print(score_counter.most_common(2))  # 代入的值代表要取幾個數

"""OrderedDict"""
scores = OrderedDict([('James', 80), ('Andy', 70), ('Curry', 100)])  # 每組要用tuple處理
# print(scores)

# for s in scores:
#     print(s)

""" deque """
s = 'abcde'
d = deque(s)
print(d)

print(d.popleft())  # 從左邊取出
print(d.popleft())
print(d.pop())  # 從右邊取出

d.append("XDFES")
print(d)

d.appendleft("YES")
print(d)


"""HW"""
# 給定兩個字串s跟t，已經知道t的組成，
# 是將s的字母打亂以後進行重組，再隨機加上一個字母。
# 請用前面所學，找出被加上的那個字母。
s = "abc"
t = "abcd"



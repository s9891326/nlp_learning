ans = 37
l, r = 0, 100

def try_except_func():
    while True:
        try:
            guess_ans = int(input("pls input num from 1 ~ 100: \n"))
        except:
            print("please input correct type, don't input other type symbol")
            continue
        if guess_ans < l or guess_ans > r:
            print("please input correct range number")
            continue

        if guess_ans == ans:
            print("correct answer!!!")
            break
        elif guess_ans > ans:
            print("too high in guess")
        else:
            print("too small in guess")

# try_except_func()

""" recursion """

# not recursion
def cal(end=100):
    res = 0
    for i in range(1, end + 1):
        res += 1
    return res
# cal()

# recursion
def cal(end=100):
    return end + cal(end - 1) if end != 1 else 1
# print(cal(1000))

""" call stack """
import sys
# print(sys.getrecursionlimit())  # output 1000 -> 最大上限是1000個stack，所以把上面的cal(1000)，程式就會壞掉了


""" HW """
# 假定有一個樓梯，你從第0階要爬到第n階，
# 每次你只能選擇爬1階或者爬2階，這樣稱做一步。
# 請寫出一個函式名為cs，給定n的値以後(n > 0)，
# 計算出從第0階爬到第n階的方法共有幾種不同的變化？
# 例：
# cs(1) = 1 (1)
# cs(2) = 2 (1+1, 2)
# cs(3) = 3 (1+2, 2+1, 1+1+1)
# cs(4) = 5 (1+1+2, 2+2, 1+2+1, 2+1+1, 1+1+1+1)
# 請分別給出遞迴解和迭代解。


def cs(n):
    if n == 1 or n == 2: return n
    s1, s2 = 1, 2
    for i in range(n - 2):
        s1, s2 = s2, s1 + s2
    return s2

for i in range(1, 101):
    print(cs(i))


# def cs(n):
#     if n == 2 or n == 1:
#         return n
#     else:
#         return cs(n - 1) + cs(n - 2)
# print(cs(0))

# 如果數量越來越大會花越多時間 cs(9) -> cs(8) + cs(7) ...
# for i in range(1, 101):
#     print(cs(i))

""" 修改1，將用過的答案記下來(list or dict) """
def cs(n, dic):
    if n in dic:
        return dic[n]
    dic[n] = cs(n - 1, dic) + cs(n - 2, dic)
    return dic[n]

# dic = {1: 1, 2: 2}
# for i in range(1, 101):
#     print(cs(i, dic))

""" 修改2，lru_cache，是一個工具，他可以記住函式計算過的內容，並存放起來 """
import functools

@functools.lru_cache(maxsize=None)
def cs(n):
    if n == 2 or n == 1:
        return n
    else:
        return cs(n - 1) + cs(n - 2)

# for i in range(1, 101):
#     print(cs(i))





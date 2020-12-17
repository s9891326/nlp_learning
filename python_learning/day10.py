# 上次我們的猜數字遊戲本來是固定的數字，
# 已知現在可以使用從random模組中的函式取得亂數法，
# (詳見Python Document https://docs.python.org/3/library/random.html)
# 請利用random.randint(a,b)或random.random()，
# 將前面的題目中要猜的數字改成隨機的1~100(含)之間的整數。
#
# 承上題，1~100當中有一些數假設有我們想避開，不想被成為要猜的數字的話，
# 若給定該串列avoid_lt = [4, 14, 44, 94]，
# 請參照上面的說明，使用random.choice(seq)來處理。
# (random.choice()方法可以從一個序列型態的東西seq中隨機取出一個值)
# (序列是有順序的元素的集合統稱，比如list, tuple, range)
# 提示：可以先新增一個數列並去處掉不要的元素再做random.choice()

import random

# 1
ans = random.randint(1, 100)
def hw_one():
    while True:
        guess_ans = int(input("pls input num from 1 ~ 100: \n"))
        if guess_ans == ans:
            print("correct answer!!!")
            break
        elif guess_ans > ans:
            print("too high in guess")
        else:
            print("too small in guess")
# hw_one()

# 2
avoid_lt = [4, 14, 44, 94]
ans = [i for i in range(1, 101) if i not in avoid_lt]
print(ans)
# print(random.choice(random.randint()))

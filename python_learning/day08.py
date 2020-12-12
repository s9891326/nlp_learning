def print_all(r, pi=3.14):
    def area():
        return pi * r ** 2

    def perimeter():
        return 2 * pi * r

    # 下面{}的用法是所謂的format，可以將多個變數按照順序放置到{}中
    print('半徑 = {}的圓，其周長 = {}，面積 = {}'.format(r, area(), perimeter()))


# print_all(3, 3.14159)


"""global"""
fak = 'global'  # First-Aid Kit


def home1():
    print(fak)  # 直接取得全域的變數，不做修改


def home2():
    fak = 'h2'  # 定義一個local的變數，所以修改到的變數跟全域的fak無關
    print(fak)


def home3():
    global fak  # 告訴Python現在要用的就是全域的那個fak
    fak = 'h3'
    print(fak)

# print('Before:')
# print(fak)
# print('\nhome1:')
# home1()
# print('After home1:')
# print(fak)
#
# print('\nhome2:')
# home2()
# print('After home2:')
# print(fak)
#
# print('\nhome3:')
# home3()
# print('After home3:')
# print(fak)

"""home work"""
# 1.請使用兩個迴圏，將1~10之間的偶數兩兩相乘並放到一個空的list中。
# (所以這個list應該會有2 * 2, 2 * 4, 2 * 6, 2 * 8, 2 * 10, 4 * 2, 4 * 4, ..., 10 * 10)

# 2. 請改用列表生成式來完成1的問題。

# 3.請用while, if else等，寫出一個猜數字的遊戲，遊戲的答案為37， 請在開始時提示使用者猜1~100範圍中的數字，
# 並依據使用者的答案，逐步將範圍縮小，直到猜中答案，則印出恭喜訊息並離開迴圏。

# 1
result = list()
for i in range(2, 11, 2):
    for j in range(2, 11, 2):
        result.append(i * j)

print(result)

# 2
result = [i * j for i in range(2, 11, 2) for j in range(2, 11, 2)]
print(result)

# 3
ans = 37
while True:
    guess_ans = int(input("pls input num from 1 ~ 100: \n"))
    if guess_ans == ans:
        print("correct answer!!!")
        break
    elif guess_ans > ans:
        print("too high in guess")
    else:
        print("too small in guess")

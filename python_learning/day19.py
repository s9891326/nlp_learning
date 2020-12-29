from datetime import date
double_ten = date(2020, 10, 10)
# print(double_ten.year, double_ten.month, double_ten.day)

# print(double_ten.isoformat())

# print(str(double_ten))

today = date.today()
# print(str(today))

# print(today.weekday())  # 0 ~ 6
# print(today.isoweekday())  # 1 ~ 7

# print(double_ten.strftime('%Y--%m--%d, WeekDay: %a'))

"""time"""
from datetime import time
now = time()
# print(now)
# print(now.isoformat())

now = time(0, 0, 0, 28)
# print(now.microsecond)

now = time(hour=19, second=50)
# print(str(now))

"""datetime"""
from datetime import datetime
# print(datetime.today())
# print(datetime.utcnow())
# print(str(datetime.now()))
# print(datetime.now().isoformat())

"""caculate datetime"""
from datetime import date, datetime, time, timedelta
valentine = date(2020, 2, 14)
today = date.today()
romanticlen = today - valentine
# print(romanticlen)

remember = [100, 200, 520, 1000, 2000]
memorialday = [valentine + timedelta(days=i) for i in remember]
# print(memorialday)

diff = [i - today for i in memorialday]
# print(diff)

"""time"""
import time
# print(time.time())
# print(time.localtime())
# print(time.ctime())
# print(time.strftime('現在是%A, %Y年%m月%d日的%H時%M分%S秒', time.localtime()))

"""os , time"""
import os
start = time.time()
# for path, dirs, files in os.walk("."):
#     print(path)
#     for f in files:
#         print(os.path.join(path, f))
#
#     for d in dirs:
#         print(os.path.join(path, d))
# time.sleep(10)
# end = time.time()
#
# print("總耗時：%f" % (end - start))

"""hw"""
# 承前面小亦和阿啾的狀況需要補救，
# 請幫其重新列出接下來到明年(2021)幾個重要的節日的時間，
# 計算距離2020-10-01的天數，並排序將其從近排到遠，
# 這樣小亦才不會漏掉。(1000天或2000天什麼的先不用算)
# [七夕(2021-08-14)、情人節(2021-02-14)、白色情人節(2021-03-14)、阿啾生日(2020-10-03)、阿啾生日(2021-10-03)]
# (什麼？你說為什麼有兩個生日？生日當然要每個生日都要過阿，
# 會問這個問題的讀者請檢討一下自己是不是憑實力單身阿XD?)

now_date = date(2020, 10, 1)
remember = [date(2021, 8, 14), date(2021, 2, 14), date(2021, 3, 14),
            date(2020, 10, 3), date(2021, 10, 3)]

diff_days = sorted([(i - now_date).days for i in remember])

print(diff_days)

import os
import shutil

print(os.path.exists("test.py"))

print(os.path.isfile("day18.py"))

print(os.path.isdir("day18.py"))  # 是檔案 不是資料夾

print(os.path.isdir("utils"))

# 複製檔案
# shutil.copy('poem.txt', 'poem2.txt')  # 前面是來源，後面是目的地

print(os.listdir())

"""os.walk()"""
for root, dirs, files in os.walk("."):
    print(root)
    for f in files:
        print(os.path.join(root, f))



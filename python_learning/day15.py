# 方法一
# file = open(name, mode)
# ... (使用file來處理檔案)
# file.close() # 用完要關閉檔案

# 方法二
# with open(name, mode) as file:
#     ...(使用file來處理檔案)
# # 離開這個with的區塊以後，file自動關閉。


# 'r' -> 讀取(read)
# 'w' -> 寫入(write)(但不給r預設還是會可讀)
# 'x' -> 新增檔案(exclusive creation)，如果檔案已存在則回傳錯誤
# 'a' -> 在結尾處寫入(append)
#
# 'b' -> 用二進位的方式來處理
# (預設則是當成文字來處理)
# '+'號： -> 更新(updating) (可讀可寫)
# 通常會用'r+'，代表可讀可寫。

file_name = "poem.txt"

def write():
    f = open(file_name, "w")
    f.write("院子落葉\n跟我的思念厚厚一疊")
    f.close()

def read():
    f = open(file_name, "r")
    poem = f.read()
    print(poem)
    f.close()

def read_line():
    f = open(file_name, "r")
    cnt = 0
    poem = ""
    while True:
        cnt += 1
        line = f.readline()
        if not line: break
        print(f"Line {cnt}: {line}", end="")
    f.close()

def read_lines():
    f = open(file_name, "r")
    lines = f.readlines()
    print(lines)
    f.close()


if __name__ == '__main__':
    # write()
    # read()
    # read_line()
    read_lines()

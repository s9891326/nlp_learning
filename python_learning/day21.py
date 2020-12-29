import timeit

# 因為在這裡預設的次數是1000000次(一百萬次)
# print(timeit.timeit("'-'.join(str(n) for n in range(100))"))

# print(timeit.timeit("'-'.join(str(n) for n in range(100))", number=10000))

# print(timeit.timeit(stmt='x=3;y=5;res=x*y'))

# print(timeit.repeat(stmt="x=3;y=5;res=x*y", number=10000, repeat=7))


def f():
    import os
    for path, dirs, files in os.walk('.'):
        print(path)
        for f in files:
            print(os.path.join(path, f))

        for d in dirs:
            print(os.path.join(path, d))

# print(timeit.timeit(stmt=f, number=5))
# print(timeit.timeit('f()', setup='from __main__ import f', number=5))
# print(timeit.timeit('f()', globals=globals(), number=5))


"""hw"""


def cs1(n):
    if n == 1 or n == 2:
        return n
    return cs1(n - 1) + cs1(n - 2)


def cs2(n, dic):
    if n in dic:
        return dic[n]
    dic[n] = cs2(n - 1, dic) + cs2(n - 2, dic)
    return dic[n]


# dic = {1 : 1, 2 : 2} # 這個應該要放到setup裡

import functools


@functools.lru_cache(maxsize=None)
def cs3(n):
    if n == 1 or n == 2:
        return n
    return cs3(n - 1) + cs3(n - 2)

# print(timeit.timeit(stmt="print(cs1(35))", globals=globals(), number=10))
# print(timeit.timeit(stmt="print(cs2(35, dict))", setup="dict = {1: 1, 2: 2}", globals=globals(), number=10))
# print(timeit.timeit(stmt="print(cs3(35))", setup="import functools", globals=globals(), number=10))

from PIL import Image, ImageDraw, ImageFont
img = Image.open("flower.jpg")
print(f"image format: {img.format}")
print(f"image size: {img.size}")
print(f"image mode: {img.mode}")
# img.show()

img_rotate = img.rotate(180)  # 將圖片順時針轉180度
# img_rotate.show()

# 要裁切請用crop, 並且必須傳入Tuple，4個數分別代表左上(x1, y1)/右下(x2, y2)
# 在一般程式的圖形處理中，原點(0, 0)在左上角
img_crop = img_rotate.crop((960, 50, 1920, 1080))
# img_crop.show()

img_resize = img_crop.resize((240, 250))
# img_resize.show()

img_res = img_resize.rotate(180)
# img_res.save("res.bmp")

img_rotate.paste(img_res, (0, 0))
# img_rotate.show()
# img_rotate.save("combination.png")


"""長輩圖"""
img = Image.open("flower.jpg")
backup_img = img.copy()
# backup_img.show()

# 建立一個Draw物件，接下來draw所有的操作都會影響到img上面。
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("./msjhbd.ttc", 100)  # 給定字型及大小
text = "霹靂卡霹靂拉拉 波波力那貝貝魯多"
draw.text((960, 320), text, font=font)
# img.show()

font = ImageFont.truetype("./msjhbd.ttc", 50)  # 給定字型及大小
draw.text((960, 320), text, font=font)
# img.show()

img = backup_img.copy()  # 洗掉吧！
draw = ImageDraw.Draw(img)
draw.ink = 0xff0000  # 0x代表16進位
draw.text((960, 320), text, font=font)
# img.show()
draw.text((1060, 960), "認同分享", font=font, fill=(255, 0, 0, 128))
# img.show()

# 正式來囉！關於顏色的選擇可以使用如htmlcolorcodes等網站來取得色碼
img = backup_img.copy()  # 洗掉吧！
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('./msjhbd.ttc', 60)
draw.ink = 0xF39C12
text = '請常唸\n\n　　霹靂卡霹靂拉拉\n　　波波力那貝貝魯多\n\n唸時\n須心無雜念 專注 便可心想事成'
draw.text((1000, 300), text, font=font)
font = ImageFont.truetype('./msjhbd.ttc', 70)
draw.text((1520, 960), '認同請分享', font=font, fill=(165, 105, 189, 0))
img.show()
img.save("elder.jpg")

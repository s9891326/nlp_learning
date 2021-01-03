import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0, 0.1
s = np.random.normal(mu, sigma, 1000)
# plt.hist(s)  # 繪製直方圖，預設分成10組

# plt.hist(s, 30)  # 第二個參數為30, 代表將值的範圍切分成30等份(預設則為10)

x = np.arange(1, 9, 2)
y = 2 * x
# plt.plot(x, y, linewidth=1.5)  # 這樣應該是y=2x的樣子

# plt.plot([1, 2, 3, 4])
# plt.plot([1, 2, 3, 4], [1, 2, 3, 4])  # 留意X座標的變化!

"""
一次畫多張圖片(1, 1, 1)
第一個1 -> x
第二個1 -> y
第三個1 -> 第幾張圖片
"""
# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
# ax.set_title('Title')  # 表題
# ax.set_xlabel("label: x")  # X軸文字
# ax.set_ylabel("label: y")  # Y軸文字
# fig.suptitle('Sup Title', fontsize=20, fontweight='bold')
#
# ax = fig.add_subplot(1, 2, 2)
# ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
# ax.set_title('Title')  # 表題
# ax.set_xlabel("label: x")  # X軸文字
# ax.set_ylabel("label: y")  # Y軸文字
# fig.suptitle('Sup Title', fontsize=20, fontweight='bold')


"""各種畫線方式"""
# plt.plot([1, 2, 3, 4], [3, 5, 15, 18], 'rx')  # r代表紅色，x代表用'x'來表示點，且不畫線
# plt.plot([1, 2, 3, 4], [3, 9, 1, 6], 'b.--')  # b代表藍色，'.'代表用單個小點表示一個點，'--'表示用虛線(dashed line)來畫線
# plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'go-', linewidth=3)  # g代表綠色，o代表實心圓，linewidth表示粗度修為3
# plt.legend(('red', 'blue', 'green'), loc='upper left')  # 畫圖例及決定位置
# plt.grid(True)  # 畫出網格
#
# print(np.random.random(5))
x = np.random.random(500)
y = np.random.random(500)
# plt.scatter(x, y)  # 其實用plt.plot(x, y, 'o')也可以啦！


z = np.random.normal(0, 0.2, 500)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z, color='blue')


if __name__ == '__main__':
    plt.show()

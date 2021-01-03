import numpy as np

a = np.array([1, 5, 3, -1])
# print(a)
# print(a.ndim)
# print(a.size)
# print(a.shape)

# b = np.array([[1, 2, 3, 4], [1, 2, 3]])  # 長度要符合

b = np.array([[1, 3, 2, 4], [5, 4, 3, 3]])
# print(b.ndim, b.size, b.shape)

# print(np.arange(10))
# print(np.arange(5, 10))
# print(np.arange(5, 11, 2))

# print(np.zeros(3))
# print(np.zeros(3,))  # 因為傳入是Tuple，所以多寫一個逗號主要是提醒不要把外面的括號省略
# print(np.zeros((1, 5)))
# print(np.zeros((3, 4)))

# print(np.random.random((2, 6)))

mu, sigma = 0, 0.2
c = np.random.normal(mu, sigma, (2, 8))
# print(c)

s = np.random.normal(mu, sigma, 9)
# print(s)

s.reshape(3, 3)  # reshape完後若沒有回頭存起來，並不會修改到s呦!
# print(s)

s.reshape(1, 9)
# print(s)

s.reshape(9, 1)
# print(s)

s.shape = (3, 3)
# print(s)

d = np.arange(24)
# print(d)
# print(d[7])
# print(d[-1])
# print(d[10:18])
# print(d[10:18:2])

d.shape = 3, 2, 4
# print(d)
# print(d[1:, 1, 3:])
# print(d[1:, 1, 3:])
# print(d[1, 1])
# print(d[1, 1, 2:])
d[1, 1, 2:] = 99999  # 將指定範圍的值全數換成99999
# print(d)
# print(d + 1)
# print(d / 3 + 2)

x = np.arange(1, 5)
x.shape = 2, 2
y = np.arange(5, 9)
y.shape = 2, 2

print(x)
print(y)
print(x.dot(y))
print(np.dot(x, y))
print(np.multiply(x, y))


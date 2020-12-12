# 型態
print(type(10.0))
print(type(10))
print(type('25'))
print(type(True))
print(type(False))
print(type(None))

# 將標籤名稱貼到同一個箱子上。
a = 10
b = a
print(f"\na: {a}, id: {id(a)}")
print(f"b: {b}, id: {id(b)}")
print(f"a == b : {id(a) == id(b)}")
a = 123
print("------change a value------")
print(f"a: {a}, id: {id(a)}")
print(f"b: {b}, id: {id(b)}")
print(f"a == b : {id(a) == id(b)}")

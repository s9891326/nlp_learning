# def my_range(n):
#     i = 0
#     while i < n:
#         print("我先在\"yield " + str(i) + "\"前睡了，等__next__來再叫我")
#         yield i
#         i += 1
#         print("i = ", i)
#
#
# for i in my_range(5):
#     print(i)


# TEST default dict

# from collections import defaultdict
#
# better_dict = defaultdict(list)
# check_default = better_dict["a"]
# print(check_default)
#
# better_dict["b"] = [1]
# better_dict["b"].append(2)
# better_dict["b"].append(3)
# print(better_dict["b"])

# TEST Counter

# from collections import Counter
#
# a_str = "abcaaabccabaddeae"
# counter = Counter(a_str)  # 可直接由初始化的方式統計個數
# print(counter)
# print(counter.most_common(3))  # 輸出最常出現的3個元素
# print(counter["a"])
# print(counter["z"])  # 對於不存在的key值給出default值0
#
# counter.update('aaeebbc')  # 可用update的方式繼續統計個數
# print(counter)

# test namedtuple

# from collections import namedtuple
#
# Identity = namedtuple('Identity', ["first_name", "last_name", "birthday"])
# identity = Identity("Wang", "eddy", "8/8")
# print(identity)
# print(identity.first_name)
# print(identity.last_name)
# print(identity.birthday)
#
# print(identity[0])
# print(identity[1])
# print(identity[2])
#
# # 更改數據
# identity = identity._replace(birthday="11/25")
# print(identity)
#
# default_identity = Identity("", "", "")
#
# def dict_to_identity(s):
#     return default_identity._replace(**s)
#
# print(dict_to_identity({"first_name": "eddy"}))
# print(dict_to_identity({"first_name": "wang", "last_name": "eddy", "birthday": "8/8"}))

# deque
#
# from collections import deque
#
# d = deque("123456789")
# print(d)
#
# print(d.pop())  # 提取右邊元素
# print(d)
#
# print(d.popleft())  # 提取右邊元素
# print(d)
#
# d.extend(["a", "b"])  # 從tail端新增元素
# print(d)
#
# d.extendleft(["x", "y"])  # 從head端新增元素
# print(d)
#
# d.rotate(1)  # 看起來稍微巧妙一點的一個方法，其功能和d.appendleft(d.pop())一樣，像轉動輪盤一樣把所有元素的位置像右移了一位，參數可接受負值向左移位
# print(d)
#
# d.reverse()
# print(d)
#
# d = deque("123456", maxlen=6)
# print(d)
#
# d.extend("789")
# print(d)
#
# d.extendleft("xyz")
# print(d)

# orderedDict

# from collections import OrderedDict
#
# d = OrderedDict()
# d['first'] = 5
# d['second'] = 4
# d['third'] = 8
# d['fourth'] = 7
# print(d)
#
# print(d.pop("third"))
# print(d)
#
# d["third"] = 8
# print(d)
#
# for k, v in d.items():
#     print(k, v)
#
# order = OrderedDict([('1', 4), ('2', 3), ('4', 1), ('3', 2)])  # OrderedDict會依賴著tuple構成的list的順序來排序，我們可以藉此特性來排序
# print(order)
#
# d = {'4': 2, '1': 4, '2': 3, '3': 5}
# print(d)
#
# print(sorted(d.items(), key=lambda x: x[0]))  # 以key值排序
# print(sorted(d.items(), key=lambda x: x[1]))  # 以value值排序
# print(OrderedDict(sorted(d.items(), key=lambda x: x[0])))  # 以key值排序
# print(OrderedDict(sorted(d.items(), key=lambda x: x[1])))  # 以key值排序


# chainMap
#
# from collections import ChainMap
#
# a_dict = {'a1': 1, 'a2': 2, 'c1': 3}  # a_dict裏面的key:'c1',會和c_dict裡重複
# b_dict = {'b1': 4, 'b2': 5, 'b3': 6}
# c_dict = {'c1': 7, 'c2': 8, 'c3': 9}
#
# dicts = ChainMap(a_dict, b_dict, c_dict)
#
# print(dicts['a2'])  # 2
# print(dicts['c1'])  # 3 而不是 7
# print(dicts['c2'])  # 8
# print(dicts['b2'])  # 5
# # print(dicts['b4'])  # KeyError


# # str() vs repr()
# class Test:
#     def __init__(self):
#         pass
#
#     def __repr__(self):
#         return "the repr"
#
#     def __str__(self):
#         return "the str"
#
# a = Test()
# a  # the repr
# print(a)  # the str

# # set 是一個沒有順序的不重複元素序列
# thisset = set(("Google", "Runoob", "Taobao"))
# print(thisset)
#
# thisset.add("Facebook")
# print(thisset)
#
# thisset.update({1, 3}, [2, 4])  # set update 可以同時多個，也可以用dict or list的方式
# print(thisset)
#
# thisset.remove("Google")
# print(thisset)
#
# thisset.discard("Google")
# print(thisset)
#
# print(thisset.pop())
# print(thisset)
#
# thisset.clear()
# print(thisset)

"""
利用 __iter__ 和 __next__ 實作一個 Iterator：
"""


# class MyIterator:
#     def __init__(self, max_num):
#         self.max_num = max_num
#         self.index = 0
#
#     # first
#     # def __iter__(self):
#     #     return self
#
#     # def __next__(self):
#     #     self.index += 1
#     #     if self.index <= self.max_num:
#     #         yield self.index
#     #     else:
#     #         raise StopIteration
#
#     # second
#     # yield自動return一個Generator的object，而這個generator自帶`__next__` -> Iterator
#     def __iter__(self):
#         print("__iter__")
#         num = 1
#         while num <= self.max_num:
#             yield num
#             num += 1
#
# my_iterator = MyIterator(3)
# for item in my_iterator:
#     print(item)
#
# for item in my_iterator:
#     print(item)

# """
# 利用 __getitem__ 實作一個 Iterator：
# """
# class MyIterator:
#     def __init__(self, max_num):
#         self.max_num = max_num
#
#     def __getitem__(self, key):
#         if key <= self.max_num:
#             return key
#         else:
#             raise IndexError
#
# my_iterator = MyIterator(3)
#
# for item in my_iterator:
#     print(item)

"""深入了解 yield"""

# def generator_func(value=0):
#     while value < 10:
#         value = yield value
#         value += 1
#
# generator = generator_func()
#
# print('step 1')
# print(next(generator))
# print('step 2')
# print(generator.send(1))
# print('step 3')
# print(generator.send(7))
# print('step 4')
# # print(generator.send(10))
#
#
# x = (i for i in range(2))
#
# print(type(x))
# # <class 'generator'>
#
# print(next(x))
# print(next(x))
# # print(next(x))

"""
test import
"""
# print(f"__file__{__file__}")
# print(f"__package__{__package__}")

"""
test hnswlib
"""

import hnswlib
import numpy as np

dim = 128
num_elements = 10000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))
data_labels = np.arange(num_elements)

# Declaring index
p = hnswlib.Index(space = 'l2', dim = dim) # possible options are l2, cosine or ip

# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)

# Element insertion (can be called several times):
p.add_items(data, data_labels)

# Controlling the recall by setting ef:
p.set_ef(50) # ef should always be > k

# Query dataset, k - number of closest elements (returns 2 numpy arrays)
labels, distances = p.knn_query(data, k=1)

print(p)

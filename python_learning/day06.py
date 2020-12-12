dict0 = dict()
print(dict0)

# key必須要是hashable的資料型態
# dic = {'xd': 1, '放棄': False, [1,5,9]: 3}

dic = {'xd': 1, '放棄': False, 3.33: 3}  # 這樣就沒問題啦！
print(dic)

lol = [('易大師', '我的劍，就是你的劍。'), ('犽宿', '死亡如風，常伴吾身。'), ('阿祈爾', '蘇瑞瑪！你的王已經歸來了！')]
diclol = dict(lol)
print(diclol)

diclol['國動'] = '社 社 社社社 社社 社會搖'
print(diclol)

diclolfame = {'國動': '還敢下來阿冰鳥!', '統神': '他的手怎麼可以穿過我的叭叭啊！'}
diclol.update(diclolfame)  # 合併過去，重覆的會被覆蓋
print(diclol)

dicchs = {'a': 123, 'c': 428, '1': '3', 'eee': 11}
print(dicchs)
dicchs.clear()  # 清空字典
print(dicchs)

print('統神' in diclolfame)
print('統神' in diclol)  # 我們剛剛刪除了，所以'統神'不會在diclol裡

dicn = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
print(dicn)

print(dicn.keys()) # 取得key

print(list(dicn.keys()))

print(dicn.values())

print(dicn.items())

dico = dicn
dico['a'] = 'XD'
print(dico)
print(dicn)  # 結果原先的也一起被改到了！

dico = dicn.copy()  # 使用copy()來處理
dico['a'] = 1
print(f"dico : {dico}")

print(f"dicn : {dicn}")

print("============================================")
st = set()
print(st)

st.add(3)
print(st)

st.add("XD")
print(st)

st.add("xddd?")
print(st)

print({'XDDDD', 'XD', 'XD', 'XDDD'})  # list則會以元素為單位，此外，留意它並沒有順序。)

st1 = {"A", "C", "E"}
st2 = {"B", "C", "A", "D"}
print(f"'A' in st1 : {'A' in st1}")
print(f"'e' in st1 : {'e' in st1}")

print(f"st1 & st2 = {st1 & st2}")
print(f"st1.intersection(st2) : {st1.intersection(st2)}")
print(f"st1 | st2 = { st1 | st2}")
print(f"st1.union(st2) = {st1.union(st2)}")
print(f"st1 - st2 = {st1 - st2}")
print(f"st2 - st1 = {st2 - st1}")
print(f"st1.difference(st2) = {st1.difference(st2)}")
print(f"st1 ^ st2 = {st1 ^ st2}")
print(f"st1.symmetric_difference(st2)= {st1.symmetric_difference(st2)}")
print(f"st1 <= st2 = {st1 <= st2}")  # 檢查前者是不是後者的子集
print(f"st2 <= st1 = {st2 <= st1}")
print(f"st1.issubset(st2) = {st1.issubset(st2)}")


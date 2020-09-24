## Python雜談

### python collections雜談之一
1. defaultdict
     * default_factory：就是defaultdict在初始化的過程中，第一個參數所接受的函數對象(也就是上述的list()或是zero())，而第二個之後的參數都比照一般dict傳入參數的格式。
     * __missing__(key)：在我們調用不存在的key值時defaultdict會調用__missing__(key)方法，這個方法會利用default_factory創造一個default值給我們使用。

```python
from collections import defaultdict

def zero():
    return 0

counter_dict = defaultdict(zero) # default值以一個zero()方法產生
a_list = ['a','b','x','a','a','b','z']

for element in a_list:
        counter_dict[element] += 1

print(counter_dict) # 會輸出defaultdict(<function zero at 0x7fe488cb7bf8>, {'x': 1, 'z': 1, 'a': 3, 'b': 2})
```

2. Counter

```python
from collections import Counter

a_str = 'abcaaabccabaddeae'
counter = Counter(a_str) # 可直接由初始化的方式統計個數
print(counter)
print(counter.most_common(3)) # 輸出最常出現的3個元素
print(counter['a'])
print(counter['z']) # 對於不存在的key值給出default值0

counter.update('aaeebbc') # 可用update的方式繼續統計個數
print(counter)
```

3. Namedtuple
* 因為不可變，當然就用在不會變動的資料上阿！
* 因為不可變，當然就用在不能給人變動的資料上阿！

```python
from collections import namedtuple

Identity = namedtuple('Identity', ["first_name", "last_name", "birthday"])
identity = Identity("Wang", "eddy", "8/8")
print(identity)
print(identity.first_name)
print(identity[0])
```

4. deque
* 能用在音訊分析或其他時間序列分析上所用的一個被稱為'幀'(time frame)的概念，也就是從比較長的時間間隔中取一段小的時間間隔，並在之後對其做一些數值分析。 

```python
from collections import deque

d = deque("123456789")
print(d)

print(d.pop())  # 提取右邊元素
print(d)

d.extend(["a", "b"])  # 從tail端新增元素
print(d)
```

5. OrderedDict
* 能夠排序的字典

```python
from collections import OrderedDict

d = OrderedDict()
d['first'] = 5
d['second'] = 4
d['third'] = 8
d['fourth'] = 7
print(d)

print(d.pop("third"))
print(d)

d["third"] = 8
print(d)

for k, v in d.items():
    print(k, v)

print(OrderedDict(sorted(d.items(), key=lambda x: x[0])))  # 以key值排序
```

6. ChaninMap
* ChainMap就是把多個map，也就是dict鏈在一起，在功能上可以理解成，把所有的dict合併成一個大的dict，但在比較底層的實現中，這些dict還是維持著自己原本的樣子，ChainMap的作法就像是把他們存在一個list裡。

```python
from collections import ChainMap

a_dict = {'a1': 1, 'a2': 2, 'c1': 3}  # a_dict裏面的key:'c1',會和c_dict裡重複
b_dict = {'b1': 4, 'b2': 5, 'b3': 6}
c_dict = {'c1': 7, 'c2': 8, 'c3': 9}

dicts = ChainMap(a_dict, b_dict, c_dict)

print(dicts['a2'])  # 2
print(dicts['c1'])  # 3 而不是 7
print(dicts['c2'])  # 8
print(dicts['b2'])  # 5
print(dicts['b4'])  # KeyError
```

### str() vs repr()

> str()
>> 可讀性較高
>> '有用資訊'的字串


> repr() -> representation
>> 給python的直譯器看的
>> '明確且教詳盡的資訊'

```python
class Test:
    def __init__(self):
        pass
    
    def __repr__(self):
        return "the repr"
    
    def __str__(self):
        return "the str"

a = Test()
a  # the repr
print(a)  # the str
```

* [註一]： python的eval和exec是一個可以動態執行程式碼的函數，凡傳進這兩個函數裡的字串參數，都會被當作一段代碼來執行，不同的是eval只能執行一行expression(表達式)，exec可以執行多行程式碼，這兩個函數讓python可以動態產生新的程式碼並執行，這實在是一個強大的功能，但是這樣的自由度也帶來一定的危險性，因為光是一行字串就可能給予整個程式或是整個系統莫大的影響，比如說"os.system('rm -rf /')"，這會讓你整個作業系統的檔案全部被刪除！


### 定義 Python 的 Iteration / Iterable / Iterator

* Iteration : 走訪/迭代/遍歷一個 object 裡面被要求的所有元素之「過程」或「機制」。是一個概念性的詞。
* Iterable : 可執行 Iteration 的 objects 都稱為 Iterable(可當專有名詞)。是指可以被 for loop 遍歷的 objects。以程式碼來說，只要具有 `__iter__` 或 `__getitem__` 的 objects 就是 Iterable。
* Iterator : 遵照 Python Iterator Protocol 的 objects。以 Python3 而言，只要具有 `__iter__` 和 `__next__` 的 objects 皆為 Iterator。Iterator 是 Iterable 的 subset。

* [註一]: `__getitem__` : 凡是在class中定義了`__getitem__`方法，那他的實例對象(假定為p)，當像這樣p[key]取值時，會觸發到`__getitem__`方法
```python
class DataBase:
    '''Python 3 中的类'''

    def __init__(self, id, address):
        '''初始化方法'''
        self.id = id
        self.address = address
        self.d = {self.id: 1,
                  self.address: "192.168.1.1",
                  }

    def __getitem__(self, key):
        # return self.__dict__.get(key, "100")
        # return self.d.get(key, "default")
        return "eddy"
       

data = DataBase(1, "192.168.2.11")
print(data["hi"])  # return 'eddy'
print(data[data.id])  # return 'eddy'
```
* dir() : 不帶參數時，返回當前範圍內的變量、方法和定義的類型;帶有參數時，返回參數的屬性、方法列表。
```python
dir([1,2,3]) # ['__add__', '__class__', '__contains__', '__delattr__', '__delitem__',...]
```
* hasattr(object, name) : 判斷object是否包含了name屬性
```python
hasattr([1, 2, 3], '__iter__')  # return True
```


### yield
1. `yield` 只能出現在 function 裡，而 call 帶有 `yield` 的 function 會回傳一個 `Generator` object。
2. `Generator` object 是一個 Iterator，帶有 `__iter__` 和 `__next__` attributes。
3. 第一次 call `next(generator)` 執行內容等價於將原 function 執行到第一次出現 `yield` 之處，「暫停」執行，並返回 `yield` 後的值。
4. 第二次之後每次 call `next(generator)` 都會不斷從上次「暫停」處繼續執行，直到 function 全部執行完畢並 raise `StopIteration` 。因為 `yield` 沒有將原 function 從 call stack 中移除，故能暫停並重新回到上次暫停處繼續執行。這邏輯也是 `yield` 和 `return` 最核心不同之處，`return` 會直接將原 function 從 call stack 中移除，終止 function，不論 `return` 後面是是否還有其他程式碼。
5. `yeild` 除了可傳出值外，也可以接受由外部輸入的值，利用 `generator.send()` 即可同時傳入值也傳出值。此設計，讓 `Generator` object 可和外部進行雙向溝通，可以傳出也可以傳入值。
6. 關於 `Generator` object 的創建有兩種語法：一是在 function 內加入 `yield`」，二是形如 `x = (i for i in y)` 的方式。其實大家常用的產生 list 的其中一種寫法 `x = [i for i in range(10)]` 就是創建一個 `Generator` object 的變形。

* 概念性總結一下，原先和 `return` 搭配的 function，就是吃進 input 然後輸出 output，life cycle 就會消失，`yield` 的寫法可以視為擴充 function 的特性，使其具有「記憶性」、「延續性」的特質，可以不斷傳入 input 和輸出 output，且不殺死 function。未來要撰寫具有該特質的 function 時就可以考慮使用 `yield` 來取代「在外部存一堆 buffer 變數」的做法。

### 如何 import module
* 限制使用者的機制
1. 可以使用"_"(相對寬鬆的方式)
2. 定義一個list對象__all__，當使用者利用"from model_name import *" 這種語法時__all__可以決定對使用者提供某些對象
```python
# In module.py:
pub_var = 'I\'m public variance.'
_pri_var = 'I\'m private variance.'

def pub_func():
    return 'I\'m public func.'
def _pri_func():
    return 'I\'m private func.'


#class pub_obj():
#    def __init__():
#        self.str = 'I\'m public obj.'
#class _pri_obj():
#    def __init__():
#        self.str = 'I\'m private obj.'

__all__ = [pub_var, pub_func, _pri_var]

# other.py
#from module import *
dir() # pub_obj不見了，但是多了_pri_var
['__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', '_pri_var', 'pub_func', 'pub_var']
```
3. 如果要import別人寫好的module，但她不存在當下的工作目錄底下，那該如何?
    * 把欲加入的module的路徑手動加到sys.path這個list內
    ```python
   import sys
   sys.path # ['', '/usr/lib/python3.4', '/usr/lib/python3.4/plat-x86_64-linux-gnu', '/usr/lib/python3.4/lib-dynload', '/usr/local/lib/python3.4/dist-packages', '/usr/lib/python3/dist-packages']
   sys.path.insert(0, 'some path')
   sys.path.append('some path')
   sys.path.extend(['some path','some path'])
    ```

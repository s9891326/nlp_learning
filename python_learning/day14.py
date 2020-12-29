# class -> object

class Student:
    cnt = 0  # 這個變數是屬於整個Student類別的
    cp_cnt = 0

    def __init__(self, name, math_score=0, eng_score=0, physical_score=0):
        Student.cnt += 1  # 每次新開一個Student的物件，計數器就會+1
        self.name = name
        self.math_score = math_score
        self.eng_score = eng_score
        self.physical_score = physical_score

    def read_my_name(self):
        print(f"聽清楚了，我的名字是{self.name}!!!")

    def compare(self, obj):
        Student.cp_cnt += 1
        result_a = self.math_score + self.eng_score + self.physical_score
        result_b = obj.math_score + obj.eng_score + obj.physical_score
        if result_a > result_b:
            print(f"{self.name}的名字贏了！")
        elif result_a == result_b:
            print(f"什麼？竟然平手？！")
        else:
            print(f"可...可惡，難道，這就是{obj.name}的名字真正的實力嗎？")
        print(f"已經比較{Student.cp_cnt}次\n")

    def compareE(self, obj):
        Student.cp_cnt += 1
        if self > obj:
            print(f"{self.name} > {obj.name}")
        elif self == obj:
            print(f"{self.name} == {obj.name}")
        else:
            print(f"{self.name} < {obj.name}")
        print(f"已經比較{Student.cp_cnt}次\n")

    @classmethod
    def get_count(cls):
        print(f"目前的學生總數{cls.cnt}")

    def weight_score(self):
        return self.math_score * 2 + self.eng_score * 5

    def __gt__(self, other):
        print(f"self: {self.weight_score()}, other: {other.weight_score()}")
        return self.weight_score() > other.weight_score()

    def __eq__(self, other):
        return self.weight_score() == other.weight_score()


class A:
    def __init__(self):
        self.name = 'A'
        print('A')
        print('Name = ' + self.name)


class B:
    def __init__(self):
        self.name = 'B'
        print('B')
        print('Name = ' + self.name)


class C(B, A):  # 繼承class寫越前面的優先程度越大
    pass


class Nmod3:
    def __init__(self, num):
        self.num = num

    def __eq__(self, n2):  # 定義 "=="這個運算子為是否除以3的餘數相同
        return self.num % 3 == n2.num % 3

    # 這些都是兩個物件相比，我們假定後面的物件叫b。
    # __eq__ => self == b
    # __ne__ => self != b
    # __lt__ => self < b
    # __gt__ => self > b
    # __add__ => self + b
    # __mul__ => self * b # 原來字串的乘法是這麼弄出來的XD
    # __len__ => len(self) # 可以定義什麼是你的物件的"長度"


if __name__ == '__main__':
    """1."""
    # ming = Student('阿明', 55, 70, 55)
    # mei = Student('小美', 90, 88, 100)
    # how = Student("HowHow", 80, 60, 40)
    # # print(ming.name)
    # # print(mei.name)
    # # ming.read_my_name()
    # # mei.read_my_name()
    # ming.compare(mei)
    # mei.compare(how)
    # how.compare(ming)
    # Student.get_count()

    """2."""
    # test = C()

    """3."""
    # a = Nmod3(11)  # 除3餘2
    # b = Nmod3(18)  # 除3餘0
    # c = Nmod3(17)  # 除3餘2
    #
    # print(a == b)
    # print(b == c)
    # print(c == a)

    """hw"""
    ming = Student('阿明', 55, 70, 55)
    mei = Student('小美', 90, 88, 100)
    how = Student("HowHow", 80, 60, 40)

    ming.read_my_name()
    mei.read_my_name()
    how.read_my_name()

    print("pk~~")
    ming.compare(how)
    ming.compareE(how)
    mei.compare(ming)
    mei.compareE(how)
    how.compare(ming)
    how.compareE(mei)

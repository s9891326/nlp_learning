# class -> object

class Student:
    def __init__(self, name, math_score=0, eng_score=0, physical_score=0):
        self.name = name
        self.math_score = math_score
        self.eng_score = eng_score
        self.physical_score = physical_score

    def read_my_name(self):
        print(f"聽清楚了，我的名字是{self.name}!!!")

    def compare(self, obj):
        result_a = self.math_score + self.eng_score + self.physical_score
        result_b = obj.math_score + obj.eng_score + obj.physical_score
        if result_a > result_b:
            print(f"{self.name}的名字贏了！")
        elif result_a == result_b:
            print(f"什麼？竟然平手？！")
        else:
            print(f"可...可惡，難道，這就是{obj.name}的名字真正的實力嗎？")

if __name__ == '__main__':
    ming = Student('阿明', 55, 70, 55)
    mei = Student('小美', 90, 88, 100)
    how = Student("HowHow", 80, 60, 40)
    # print(ming.name)
    # print(mei.name)
    # ming.read_my_name()
    # mei.read_my_name()
    ming.compare(mei)
    mei.compare(how)
    how.compare(ming)

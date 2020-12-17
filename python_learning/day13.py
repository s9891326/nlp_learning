class Car(object):
    def __init__(self, name):
        self.name = name

    def whoami(self):
        print('My name is ' + self.name)
        print('I\'m a Car!')


class Tesla(Car):
    def __init__(self, name, mode):
        super().__init__(name)  # 使用super來對name初始化，看起來稍微多此一舉，但可以保證對於name處理的一致性，之後如果要額外針對Car這個父類別修改時，就可以一起同時影響到Tesla這邊
        self.pilotmode = mode

    def whoami(self):
        super().whoami()  # 先喊名字跟喊自己是輛車
        print('Also, I\'m a Tesla, not a trash car!')  # 這兩行再做只有Tesla會做的事情
        print('Auto-pilot mode: ' + str(self.pilotmode))

    def autopilot_switch(self):
        self.pilotmode ^= 1
        if self.pilotmode == 0:
            print('Auto-pilot mode switch off!\n')
        else:
            print('Auto-pilot mode switch on!\n')


car = Car('CC')
tla = Tesla('TT', 0)
car.whoami()
print()
tla.whoami()
tla.autopilot_switch()
tla.whoami()
tla.autopilot_switch()

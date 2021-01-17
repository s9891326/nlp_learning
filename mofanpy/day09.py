from typing import Any


class Cat:
    def __init__(self, name, color, **kwargs):
        self.name = name
        self.__color = color
        # self.__dict__.update(**kwargs)

    # def __setattr__(self, name: str, value: Any) -> None:
    #     super().__setattr__(name, value)

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    @property
    def name(self):
        return self.name

    @name.setter
    def name(self, name):
        self.name = name

    def output(self):
        print(f"name: {self.name}, color: {self.__color}")
    # name = property(__setattr__, __getattribute__)

if __name__ == '__main__':
    cat = Cat("eddy", "red")
    cat.output()
    # cat.__setattr__("__name", "eddy2")
    #     # cat.output()
    cat.name = "eddy3"
    cat.output()


from functools import wraps, partial
import time


def time_count_decorator(init_func=None, *, time_unit='sec'):
    if init_func is None:
        print("init_func is none")
        """
        @time_count_decorator(time_unit='hour') 想設置參數的時候，必須用關鍵字參數設定
        partial(time_count_decorator,time_unit=time_unit) 會回傳一個簡化func
        裡面的功能和time_count_decorator一樣但time_unit已經被固定住了='hour'
        只等待init_func的輸入
        """
        return partial(time_count_decorator, time_unit=time_unit)

    @wraps(init_func)
    def time_count(*pos_args, **kw_args):
        ts = time.time()
        return_value = init_func(*pos_args, **kw_args)
        te = time.time()

        if time_unit == 'sec':
            time_used = te - ts

        elif time_unit == 'min':
            time_used = (te - ts) / 60

        elif time_unit == 'hour':
            time_used = (te - ts) / 60 / 60

        print("{}'s time consume({}): {}".format(init_func.__name__, time_unit, time_used))

        return return_value
    return time_count


def test(a=None, *, b=1, c=2):
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}")


def test2(*args, **kwargs):
    pass


# 等於for_loop = time_count_decorator(for_loop)
@time_count_decorator
def for_loop(n):
    """ The docstring of for_loop """
    for i in range(n):
        pass
    test(1, b=2, c=3)
    return n

# init = time_count_decorator(time_unit="min")
# 等於for_loop2 = init(init_func=for_loop2) 的感覺
@time_count_decorator(time_unit="min")
def for_loop2(n):
    """ The docstring of for_loop """
    for i in range(n):
        pass
    test(1, b=2, c=3)
    return n


if __name__ == '__main__':
    for_loop2(500000)

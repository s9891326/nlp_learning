while True:  # 使用while True須謹慎，一定要留下可以離開的方法！
    mode = input('請問你要選擇什麼模式？1. 簡單模式 2. 困難模式 3. 專家模式 [輸入1, 2, 3] ')
    if mode == '3':
        print('\n選擇專家模式的難關 帶著我的夥伴 還有我的不平凡')
        break
    elif mode == '1' or mode == '2':  # 簡單來說，其他模式都不給過XD
        print('不選難一點的嗎？再給你選一次！\n')
        continue  # 所以在這邊會直接回到迴圈開始處，因而不會印出下一行
    print('請輸入正確的選項！\n')

print('恭喜你選擇專家模式，加油！')

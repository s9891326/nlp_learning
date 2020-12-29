import csv

def write(file_name):
    with open(file_name, "w", newline="") as f:
        csvw = csv.writer(f, delimiter=" ")  # delimiter預設 "," delimiter是指分隔符號
        csvw.writerow(["姓名", "數學", "英文", "物理"])
        students = [
            ['阿明',   55,  70,   55],
            ['小美',   90,  88,  100],
            ['HowHow', 80,  60,   40]
        ]
        csvw.writerows(students)  # 一次寫多個rows

def read(file_name):
    with open(file_name, "r") as f:
        csvr = csv.reader(f, delimiter=" ")
        student_from_file = [row for row in csvr]
    print(student_from_file)

def write_dic(file_name):
    with open(file_name, 'w', newline='') as f:
        field = ['姓名', '數學', '英文', '物理'] # 第一個row做為欄位名稱
        csvw = csv.DictWriter(f, delimiter=' ', fieldnames=field)
        csvw.writeheader()
        csvw.writerow({'姓名':'阿明', '數學':55, '英文':70, '物理':55})
        csvw.writerow({'姓名':'小美', '數學':90, '英文':88, '物理':100})
        csvw.writerow({'姓名':'HowHow','數學':80, '英文':60, '物理':40})

def read_dic(file_name):
    with open(file_name, 'r') as f:
        # DictReader將第一個row當做欄位名稱，所以就省略了
        csvr = csv.DictReader(f, delimiter=' ')
        student = [row for row in csvr]
        print(student)

if __name__ == '__main__':
    file_name = "student.csv"
    # write(file_name)
    # read(file_name)

    dic_file = "student_dic.csv"
    # write_dic(dic_file)
    read(dic_file)

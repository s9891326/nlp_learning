import json

how_school = {
    "校長": "How哥",
    "工友": "林阿嘉",
    "class": {
        "A": {
            "teacher": "蔡阿嘎",
            "students": {
                "阿明": {"數學": 55, "英文": 70, "物理": 55},
                "HowHow": {"數學": 80, "英文": 60, "物理": 40}
            }
        },
        "B": {
            "teacher": "二伯",
            "students": {
                "小美": {"數學": 90, "英文": 88, "物理": 100},
                "蔡哥": {"數學": 50, "英文": 50, "物理": 40}
            }
        }
    }
}


how_json = json.dumps(how_school, ensure_ascii=False)  # ensure_ascii不要ascii
print(type(how_json))
print(how_json)

how2 = json.loads(how_json)
print(f"\n{type(how2)}")
print(how2)

classA = how2["class"]["A"]
print(classA)

def write(fileName):
    with open(fileName, "w") as f:
        # json.dump(classA, f)  # ascii
        json.dump(classA, f, ensure_ascii=False)  # utf-8

def read(fileName):
    with open(fileName, "r") as f:
        reclassA = json.load(f)
        print(reclassA)

if __name__ == '__main__':
    file_name = "classA.json"
    # write(file_name)
    read(file_name)

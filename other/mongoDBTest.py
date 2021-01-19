import pymongo

my_client = pymongo.MongoClient("mongodb://root:root@localhost:27017/")

db_list = my_client.list_database_names()
print("list database = ", db_list)

my_db = my_client["testDB"]
coll_list = my_db.list_collection_names()
print("list database = ", coll_list)

my_col = my_db["sites"]

# 新增
def insert():
    # 插入單一資料
    data = {"name": "RUNOOB", "alexa": "10000", "url": "https://www.runoob.com"}
    x = my_col.insert_one(data)
    print(x.inserted_id)

    # 插入多筆資料
    my_list = [
      {"name": "Taobao", "alexa": "100", "url": "https://www.taobao.com"},
      {"name": "QQ", "alexa": "101", "url": "https://www.qq.com"},
      {"name": "Facebook", "alexa": "10", "url": "https://www.facebook.com"},
      {"name": "知乎", "alexa": "103", "url": "https://www.zhihu.com"},
      {"name": "Github", "alexa": "109", "url": "https://www.github.com"}
    ]
    x = my_col.insert_many(my_list)
    print(x.inserted_ids)

    # 插入指定_id資料
    my_list = [
      {"_id": 1, "name": "RUNOOB", "cn_name": "菜鸟教程"},
      {"_id": 2, "name": "Google", "address": "Google 搜索"},
      {"_id": 3, "name": "Facebook", "address": "脸书"},
      {"_id": 4, "name": "Taobao", "address": "淘宝"},
      {"_id": 5, "name": "Zhihu", "address": "知乎"}
    ]
    x = my_col.insert_many(my_list)
    print(x.inserted_ids)

# 查詢
def search():
    # 查詢單筆
    x = my_col.find_one()
    print(x)

    # 查詢所有資料
    for x in my_col.find():
        print(x)

    for x in my_col.find({}, {"_id": 0, "name": 1, "alexa": 1}):
        print(x)

    for x in my_col.find({}, {"alexa": 0}):
        print(x)

    # 進行條件查詢
    my_query = {"name": "RUNOOB"}
    my_doc = my_col.find(my_query)

    for x in my_doc:
        print(x)

    # 高級查詢 查詢name第一個字母ascii值大於"H"
    my_query = {"name": {"$gt": "H"}}
    my_doc = my_col.find(my_query)

    for x in my_doc:
        print(x)

    # 正規表達式查詢
    my_query = {"name": {"$regex": "^R"}}
    my_doc = my_col.find(my_query)

    for x in my_doc:
        print(x)

    # 限制回傳筆數
    my_result = my_col.find().limit(3)

    # 输出结果
    for x in my_result:
        print(x)

# 修改
def update():
    # 修改找到的第一筆資料
    my_query = {"alexa": "10000"}
    new_values = {"$set": {"alexa": "12345"}}

    my_col.update_one(my_query, new_values)

    # 输出修改后的  "sites"  集合
    for x in my_col.find():
        print(x)

    # 修改多筆資料
    my_query = {"name": {"$regex": "^F"}}
    for x in my_col.find(my_query):
        print(x)
    new_values = {"$set": {"alexa": "123"}}

    x = my_col.update_many(my_query, new_values)

    print(x.modified_count, "文档已修改")

    for x in my_col.find(my_query):
        print(x)

# 刪除
def delete():
    # 刪除多個檔案
    my_query = {"name": {"$regex": "^F"}}
    for x in my_col.find(my_query):
        print(x)

    x = my_col.delete_many(my_query)

    print(x.deleted_count, "个文档已删除")

    for x in my_col.find(my_query):
        print(x)

    # 刪除集合中的所有資料
    x = my_col.delete_many({})

    print(x.deleted_count, "个文档已删除")

    # 刪除集合
    my_col.drop()

# 排徐
def sort():
    # alexa升序排序
    my_doc = my_col.find().sort("alexa")
    for x in my_doc:
        print(x)

    # alexa降序排序
    my_doc = my_col.find().sort("alexa", -1)
    for x in my_doc:
        print(x)

import glob
print(glob.glob("*.csv"))


"""
前處理原始的訓練數據集。
"""
import os
import pandas as pd

# 解壓縮從 Kaggle 競賽下載的訓練壓縮檔案
# os.system("unzip fake-news-pair-classification-challenge.zip")

# 簡單的數據清理，去除空白標題的 examples
df_train = pd.read_csv("train.csv")
empty_title = ((df_train["title2_zh"].isnull())
               | (df_train["title1_zh"].isnull())
               | (df_train["title2_zh"] == "")
               | (df_train["title2_zh"] == "0"))
# print(empty_title)
df_train = df_train[~empty_title]
# print(df_train["title2_zh"].head())

# 剔除過長的樣本以避免 BERT 無法將整個輸入序列放入記憶體不多的 GPU
MAX_LENGTH = 30
df_train = df_train[~(df_train.title1_zh.apply(lambda x: len(x)) > MAX_LENGTH)]
df_train = df_train[~(df_train.title2_zh.apply(lambda x: len(x)) > MAX_LENGTH)]

# 只用 1% 訓練數據看看 BERT 對少量標註數據有多少幫助
SAMPLE_FRAC = 0.01
df_train = df_train.sample(frac=SAMPLE_FRAC, random_state=9527)

# 去除不必要的欄位並重新命名兩標題的欄位名
df_train = df_train.reset_index()
df_train = df_train.loc[:, ["title1_zh", "title2_zh", "label"]]
df_train.columns = ["text_a", "text_b", "label"]

# idempotence, 將處理結果另存成 tsv 供 PyTorch 使用
df_train.to_csv("train.tsv", sep="\t", index=False)

print("訓練樣本數: ", len(df_train))
print(df_train.head())

print(df_train.label.value_counts() / len(df_train))

df_test = pd.read_csv("test.csv")
df_test = df_test.loc[:, ["title1_zh", "title2_zh", "id"]]
df_test.columns = ["text_a", "text_b", "Id"]
df_test.to_csv("test.tsv", sep="\t", index=False)

print("預測樣本數:", len(df_test))
print(df_test.head())

ratio = len(df_test) / len(df_train)
print("測試集樣本數 / 訓練集樣本數 = {:.1f} 倍".format(ratio))

# BERT 的輸入編碼格式

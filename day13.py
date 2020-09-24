# # 1.去除 HTML 標籤(Tag)
# from bs4 import BeautifulSoup
# html = "<h2 class='block-title'>今日最新</h2>"
# soup = BeautifulSoup(html, "lxml")
# print(soup.get_text())
#
# 2.分詞(Tokenize)
from nltk.tokenize import word_tokenize

# 測試字句
sent = "the the the dog, dog, some other words that we do not care about"

# 取出每個單字
list = [word for word in word_tokenize(sent)]
# print(list)

# 去除重複，並排序
vacabulary = sorted(set(list))
# print(vacabulary)

# 求得每個單字的出現機率
import nltk
freq = nltk.FreqDist(list)
# print(freq)

# 作圖
# freq.plot()

# 3.Stop Words 處理
stopwords = [",", "the"]
# 去除 Stop Words
list = [word for word in word_tokenize(sent) if word not in stopwords]
# print(list)

# 4.『詞嵌入』(Word Embedding)

# 5.『詞形還原』
# 記得載入 WordNet 語料庫
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
# 要指定單字詞性(pos)
print(wnl.lemmatize('ate', pos='v'))  # 得到 eat
print(wnl.lemmatize('better', pos='a'))  # 得到 good
print(wnl.lemmatize('dogs'))  # 得到 dog





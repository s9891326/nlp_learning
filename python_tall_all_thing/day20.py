# crawler.py
import os
from pathlib import PurePath, Path

import requests
from bs4 import BeautifulSoup

main_url = "http://big5.quanben5.com/"
cat_url = "category/"
ROOT_DIR = PurePath(__file__).parent


def check_folder(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            raise OSError


def in_entry(main_url, cat_url):  # 進入入口網站採取的行為
    for cat_num in range(1, 19):  # 點進每個類別連結的for迴圈
        url = main_url + cat_url + str(cat_num) + ".html"
        print(f"in_entry = {url}")
        yield url

def in_cat(cat_url_gen):  # 進入類別頁面所採取的行為
    for cat_url in cat_url_gen:
        res = requests.get(cat_url)  # 取得其中一個類別頁面的html code
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, "lxml")
        all_page_num = soup.select(".cur_page")[0].string  # 取得顯示頁數的神祕石板(ex.'1 / 716')
        # soup.select(".cur_page")[0] -> <class 'bs4.element.Tag'>
        all_page_num = int(all_page_num.split("/")[1])  # 取得總頁數(ex.'1 / 716' -> '716')
        for page in range(1, all_page_num + 1):
            page_url = cat_url.rstrip(".html") + '_' + str(page) + '.html'
            print(f"page_url = {page_url}")
            yield page_url

def in_page(page_url_gen, main_url):  # 進入頁數頁面所採取的行為
    for page_url in page_url_gen:
        res = requests.get(page_url)  # 取得其中一個類別頁面的html code
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, "lxml")
        for title in soup.select(".pic_txt_list"):  # 尋找小說傳送點的外衣
            chpList_url = main_url + str(title.select('a')[0]['href']) + 'xiaoshuo.html'
            print(f"chpList_url = {chpList_url}")
            yield chpList_url

def in_chpList(chpList_url_gen, dir_name, main_url):  # 進入小說章節目錄所採取的行為
    for chpList_url in chpList_url_gen:
        res = requests.get(chpList_url)  # 取得其中一個類別頁面的html code
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, "lxml")
        novel_name = str(ROOT_DIR / "quanben5-trad" / soup.select("h1")[0].string)
        check_folder(novel_name)
        with open(novel_name, "wt", encoding="utf-8") as file:  # wt模式下， python會用\r\n換行
            for chapter in soup.select("li.c3"):  # 取得所有章節連結
                chap_index = chapter.select("a")[0]["href"]
                chap_url = main_url + chap_index
                # print(f"chap_url = {chap_url}")
                yield chap_url, file
        print("內容已抓完")

def in_content(content_url_gen):  # 進入其中一個章節的內容頁面所採取的行為
    for content_url, novel_file in content_url_gen:
        res = requests.get(content_url)
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text, 'lxml')
        for content in soup.select("div#content > p"):  # 取得<div id='content'>的<p>的內容
            con = str(content.string)
            novel_file.write(con + "\r\n")
            yield content.string

def reptile():
    for cat_num in range(1, 19):  # 點進每個類別連結的for迴圈
        url = main_url + cat_url + str(cat_num) + ".html"
        res = requests.get(url)  # 取得其中一個類別頁面的html code
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, "lxml")
        all_page_num = soup.select(".cur_page")[0].string  # 取得顯示頁數的神祕石板(ex.'1 / 716')
        # soup.select(".cur_page")[0] -> <class 'bs4.element.Tag'>
        all_page_num = int(all_page_num.split("/")[1])  # 取得總頁數(ex.'1 / 716' -> '716')

        for page in range(1, all_page_num + 1):
            url = main_url + cat_url + str(cat_num) + '_' + str(page) + '.html'
            res = requests.get(url)
            res.encoding = "utf-8"
            soup = BeautifulSoup(res.text, "lxml")

            for title in soup.select(".pic_txt_list"):  # 尋找小說傳送點的外衣
                novel_url = main_url + str(title.select("a")[0]["href"]) + "xiaoshuo.html"  # 結尾加上"xiaoshuo.html"繞過小說傳送點到章節頁面之間的中間頁面
                res = requests.get(novel_url)
                res.encoding = "utf-8"
                soup = BeautifulSoup(res.text, "lxml")
                novel_name = str(ROOT_DIR / "quanben5-trad" / soup.select("h1")[0].string)
                print(novel_name)
                check_folder(novel_name)
                with open(novel_name, "wt", encoding="utf-8") as file:  # wt模式下， python會用\r\n換行
                    for chapter in soup.select("li.c3"):  # 取得所有章節連結
                        chap_index = chapter.select("a")[0]["href"]
                        chap_url = main_url + chap_index
                        res = requests.get(chap_url)
                        res.encoding = "utf-8"
                        soup = BeautifulSoup(res.text, "lxml")

                        for content in soup.select("div#content > p"):  # 取得<div id='content'>的<p>的內容
                            con = str(content.string)
                            file.write(con + "\r\n")
                        break
                    break
                break
            break
        break


cat_url_generator = in_entry(main_url, cat_url)
page_url_generator = in_cat(cat_url_generator)
chpList_url_generator = in_page(page_url_generator, main_url)
content_url_generator = in_chpList(chpList_url_generator, "", main_url)
content_generator = in_content(content_url_generator)

for content in content_generator:
    pass

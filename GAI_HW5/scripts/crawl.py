# -*- coding: utf-8 -*-
# 學號：313512072

import os
import re
import json
import time
import click
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import Counter
from urllib.parse import urljoin
from multiprocessing import Pool, cpu_count
from random import uniform

# ---------- 全域變數與資料夾 ----------
PTT_URL = "https://www.ptt.cc"
INDEX_URL = "https://www.ptt.cc/bbs/Beauty/index{}.html"
HEADERS = {"User-Agent": "Mozilla/5.0"}
OUTPUT_DIR = "outputs"
HTML_DIR = os.path.join(OUTPUT_DIR, "html")
os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 建立 session 並繞過 over18 驗證 ----------
def create_session():
    s = requests.Session()
    s.post("https://www.ptt.cc/ask/over18", data={"yes": "yes"}, headers=HEADERS)
    return s

session = create_session()

# ---------- 嘗試重連 ----------
def fetch_with_retry(url, retries=3, delay=1):
    for _ in range(retries):
        try:
            res = session.get(url, headers=HEADERS, timeout=10)
            if res.status_code == 200:
                return res
        except:
            time.sleep(delay)
    return None

# ---------- 判斷有效文章 ----------
def is_valid_article(div):
    title_tag = div.select_one("div.title a")
    if not title_tag:
        return False
    title = title_tag.text.strip()
    return title and "[公告]" not in title and "Fw:[公告]" not in title

# ---------- 判斷推爆文章 ----------
def is_popular_article(div):
    mark = div.select_one("div.nrec span")
    return mark and mark.text.strip() == "爆"

# ---------- 擷取文章資訊 ----------
#拿出 {date, title, url}
def parse_article(div):
    title_tag = div.select_one("div.title a")
    href = title_tag["href"]
    url = urljoin(PTT_URL, href)
    title = title_tag.text.strip()
    date = div.select_one("div.date").text.strip().replace("/", "").zfill(4)
    return {"date": date, "title": title, "url": url}

# ---------- 抓圖片 ----------
def extract_image_urls(soup):
    urls = []
    for tag in soup.find_all(string=True):
        urls.extend(re.findall(r'(https?://[^\s]+\.(?:jpg|jpeg|png|gif))', tag, re.IGNORECASE))
    return urls

# ---------- 快取 HTML ----------
def save_html(article):
    filename = re.sub(r'\W+', '_', article['url']) + ".html"
    path = os.path.join(HTML_DIR, filename)
    if not os.path.exists(path):
        res = fetch_with_retry(article["url"])
        if res:
            with open(path, "w", encoding="utf-8") as f:
                f.write(res.text)
    return path if os.path.exists(path) else None

# ---------- CLI 主控 ----------
@click.group()
def cli():
    pass

# ----------  Crawl 抓取 2023到2024 年文章 ----------
@cli.command()
def crawl():
    articles_path = os.path.join(OUTPUT_DIR, "articles.jsonl")
    popular_path = os.path.join(OUTPUT_DIR, "popular_articles.jsonl")
    #記錄已處理過的文章網址
    seen_urls = set()

    def is_2023_2024_article(url):
        res = fetch_with_retry(url)
        if not res:
            return False
        soup = BeautifulSoup(res.text, "html.parser")
        time_tag = soup.find("span", class_="article-meta-tag", string="時間")
        if time_tag:
            time_value = time_tag.find_next_sibling("span", class_="article-meta-value")
            if time_value:
                # 只要包含 2023 或 2024
                if "2023" in time_value.text or "2024" in time_value.text:
                    return True
        return False
    # 找出 2023 年第一篇文章的頁數
    def find_start_index():
        # target_url = "/bbs/Beauty/M.1704040318.A.E87.html"
        target_url = "/bbs/Beauty/M.1672503968.A.5B5.html"
        for idx in range(3200, 3700):  
            url = INDEX_URL.format(idx)
            res = fetch_with_retry(url)
            if not res:
                continue
            soup = BeautifulSoup(res.text, "html.parser")
            for a in soup.select("div.title a"):
                if a["href"] == target_url:
                    return idx
        return None

    start_index = find_start_index()
    if not start_index:
        print("找不到第一篇文章，請確認網路狀態或 PTT 網站是否變更")
        return

    with open(articles_path, "a", encoding="utf-8") as f_all, \
         open(popular_path, "a", encoding="utf-8") as f_pop:

        index = start_index
        while True:
            url = INDEX_URL.format(index)
            res = fetch_with_retry(url)
            if not res:
                break
            soup = BeautifulSoup(res.text, "html.parser")
            divs = soup.select("div.r-ent")
            valid_divs = [div for div in divs if is_valid_article(div)]
            if not valid_divs:
                break

            found_2023_2024 = False
            for div in valid_divs:
                article = parse_article(div)
                if article["url"] in seen_urls:
                    continue
                seen_urls.add(article["url"])

                if not is_2023_2024_article(article["url"]):
                    continue
                found_2023_2024 = True

                json_line = json.dumps(article, ensure_ascii=False) + "\n"
                f_all.write(json_line)
                if is_popular_article(div):
                    f_pop.write(json_line)

            if not found_2023_2024:
                break
            # time.sleep(uniform(0.3, 0.5))
            index += 1

# ----------  Collect 從 HTML 資料夾中收集圖片 URL 並存成 jsonl 格式 ----------
@cli.command()
def extract_images():
    """
    讀取 outputs/articles.jsonl，
    對每篇文章抓取所有圖片 URL（只保留 imgur.com），
    最後寫入 outputs/article_images.jsonl，每行只是一個 imgur 連結。
    """
    input_path = os.path.join(OUTPUT_DIR, "articles.jsonl")
    output_path = os.path.join(OUTPUT_DIR, "article_images.jsonl")

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            article = json.loads(line)
            url = article["url"]

            # 抓取文章頁面
            res = fetch_with_retry(url)
            if not res:
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            all_imgs = extract_image_urls(soup)

            # 只保留 imgur.com 的連結
            imgur_imgs = [u for u in all_imgs if "imgur.com" in u]

            # 如果有找到，就把每個連結當一行寫入
            for img in imgur_imgs:
                fout.write(json.dumps(img, ensure_ascii=False) + "\n")

            # time.sleep(uniform(0.2, 0.5))

    print(f"已將所有 Imgur 圖片連結寫入：{output_path}")

# ---------- 主程式 ----------
if __name__ == "__main__":
    cli()
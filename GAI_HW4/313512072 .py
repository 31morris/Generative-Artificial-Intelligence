# -*- coding: utf-8 -*-
# 學號：313512072
# 功能：四項功能，限定抓取 2024 年文章，使用 append 寫入。

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

# ----------  Crawl 僅抓取 2024 年文章 ----------@cli.command()@cli.command()
@cli.command()
def crawl():
    articles_path = os.path.join(OUTPUT_DIR, "articles.jsonl")
    popular_path = os.path.join(OUTPUT_DIR, "popular_articles.jsonl")
    #記錄已處理過的文章網址
    seen_urls = set()

    def is_2024_article(url):
        res = fetch_with_retry(url)
        if not res:
            return False
        soup = BeautifulSoup(res.text, "html.parser")
        time_tag = soup.find("span", class_="article-meta-tag", string="時間")
        if time_tag:
            time_value = time_tag.find_next_sibling("span", class_="article-meta-value")
            if time_value and "2024" in time_value.text:
                return True
        return False
    # 找出 2024 年第一篇文章的頁數
    def find_start_index():
        target_url = "/bbs/Beauty/M.1704040318.A.E87.html"
        for idx in range(3600, 3700):  # 粗略搜尋 aespa WINTER 所在頁
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
        print("找不到 2024 年第一篇文章，請確認網路狀態或 PTT 網站是否變更")
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

            found_2024 = False
            for div in valid_divs:
                article = parse_article(div)
                if article["url"] in seen_urls:
                    continue
                seen_urls.add(article["url"])

                if not is_2024_article(article["url"]):
                    continue
                found_2024 = True

                json_line = json.dumps(article, ensure_ascii=False) + "\n"
                f_all.write(json_line)
                if is_popular_article(div):
                    f_pop.write(json_line)

            if not found_2024:
                break
            time.sleep(uniform(0.3, 0.5))
            index += 1

# ----------  Push ----------
def parse_push(path):
    push_counter, boo_counter = Counter(), Counter()
    try:
        with open(path, encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for p in soup.select("div.push"):
            tag = p.select_one("span.push-tag")
            user = p.select_one("span.push-userid")
            if tag and user:
                t = tag.text.strip()
                u = user.text.strip()
                if t == "推":
                    push_counter[u] += 1
                elif t == "噓":
                    boo_counter[u] += 1
    except:
        pass
    return push_counter, boo_counter

@cli.command()
@click.argument("start_date")
@click.argument("end_date")
def push(start_date, end_date):
    with open(f"{OUTPUT_DIR}/articles.jsonl", encoding="utf-8") as f:
        articles = [json.loads(line) for line in f if start_date <= json.loads(line)["date"] <= end_date]
    paths = [save_html(a) for a in tqdm(articles, desc="Caching HTML") if save_html(a)]
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(parse_push, paths), total=len(paths), desc="Parsing Push"))
    total_push, total_boo = Counter(), Counter()
    for p, b in results:
        total_push.update(p)
        total_boo.update(b)

    def top10(counter):
        def sort_key(x):
            # 排序邏輯：先比 count 大小，再比 user_id 字典序（較大者排前）
            return (-x["count"], tuple(-ord(c) for c in x["user_id"]))
    
        return sorted(
            [{"user_id": u, "count": c} for u, c in counter.items()],
            key=sort_key
        )[:10]

    out = {
        "push": {"total": sum(total_push.values()), "top10": top10(total_push)},
        "boo": {"total": sum(total_boo.values()), "top10": top10(total_boo)}
    }
    with open(os.path.join(OUTPUT_DIR, f"push_{start_date}_{end_date}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

# ----------  Popular ----------
def extract_images(path):
    try:
        with open(path, encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        return extract_image_urls(soup)
    except:
        return []

@cli.command()
@click.argument("start_date")
@click.argument("end_date")
def popular(start_date, end_date):
    popular_file = os.path.join(OUTPUT_DIR, "popular_articles.jsonl")
    output_file = os.path.join(OUTPUT_DIR, f"popular_{start_date}_{end_date}.json")

    # 檢查 popular_articles.jsonl 是否存在
    if not os.path.exists(popular_file):
        print(" popular_articles.jsonl 不存在，請先執行 crawl。")
        return

    # 讀取 popular articles 並過濾日期區間
    with open(popular_file, encoding="utf-8") as f:
        articles = [json.loads(line) for line in f if start_date <= json.loads(line)["date"] <= end_date]

    # 沒有符合日期的推爆文章
    if not articles:
        result = {
            "number_of_popular_articles": 0,
            "image_urls": []
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f" popular 執行成功（無推爆文章），已寫入 {output_file}")
        return

    # 快取 HTML
    paths = [save_html(article) for article in tqdm(articles, desc="Caching HTML") if save_html(article)]

    # 抓取所有圖片網址
    with Pool(cpu_count()) as pool:
        all_images = list(tqdm(pool.imap(extract_images, paths), total=len(paths), desc="Extracting Images"))

    # 合併所有圖片網址
    image_urls = [url for group in all_images for url in group]

    #  正確格式輸出 JSON
    result = {
        "number_of_popular_articles": len(articles),
        "image_urls": image_urls
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f" popular 執行成功，共 {len(articles)} 篇推爆文章，圖片數：{len(image_urls)}，輸出：{output_file}")

# ----------  Keyword ----------
def keyword_match(args):
    article, keyword = args
    try:
        path = save_html(article)
        if not path:
            return 0, []
        with open(path, encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        content = soup.find("div", id="main-content")
        if not content:
            return 0, []

        full_text = ""

        # 把 meta 資訊一起放入全文區
        for tag in soup.select("div.article-metaline, div.article-metaline-right"):
            full_text += tag.text + "\n"

        if "※ 發信站" not in content.text:
            return 0, []

        main_body = content.text.split("※ 發信站")[0]
        full_text += main_body

        # 比對 keyword 是否出現在任一處
        if keyword not in full_text:
            return 0, []

        return 1, extract_image_urls(soup)

    except Exception as e:
        return 0, []


@cli.command()
@click.argument("start_date")
@click.argument("end_date")
@click.argument("keyword")
def keyword(start_date, end_date, keyword):
    with open(os.path.join(OUTPUT_DIR, "articles.jsonl"), encoding="utf-8") as f:
        articles = [json.loads(line) for line in f if start_date <= json.loads(line)["date"] <= end_date]

    # 多進程搜尋符合關鍵字的文章並抓圖
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(keyword_match, [(a, keyword) for a in articles]), total=len(articles), desc="Searching Keyword"))

    img_urls = [url for match, urls in results for url in urls if match == 1]

    out = {
        "image_urls": img_urls  # 只輸出這個欄位
    }

    output_path = os.path.join(OUTPUT_DIR, f"keyword_{start_date}_{end_date}_{keyword}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f" keyword 成功，共找到 {len(img_urls)} 張圖片，已輸出：{output_path}")


# ---------- 主程式 ----------
if __name__ == "__main__":
    cli()
# HW6 
## 進入cripts路徑


```bash
cd scripts
```

## Data
### 從 PTT 表特板（Beauty）爬取 2023–2024 年的文章與圖片連結
crawl：從 PTT Beauty 板中自動爬取年份為 2023 或 2024 的文章，過濾掉公告與無效文章，並將文章標題、網址、日期等資訊儲存至 outputs/articles.jsonl

extract_images：讀取上述已儲存的文章清單，逐一抓取文章內的圖片連結，僅保留 imgur.com 的圖片網址，並將所有圖片連結儲存至 outputs/article_images.jsonl
```bash
python3 craw.py crawl
python3 crawl.py extract-images
```

### 下載圖片並裁剪人臉圖像，儲存為 64x64 的圖像資料集
```bash
python3 face.py
```

## Training
### 使用 Diffusers 與 Accelerate 框架從頭訓練一個 DDPM 擴散模型，並支援混合精度、EMA
```bash
source train.sh
```

## Inference 
### 使用訓練完成的 UNet 模型與 DDPM 排程器，根據 YAML 設定自動載入模型權重，透過反向去噪流程在 GPU 上批次產生共 10,000 張 64×64 的高品質圖片，並儲存於 generated_images/ 資料夾中。
```bash
python3 inference.py
```
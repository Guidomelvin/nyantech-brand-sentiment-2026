# NYANTECH Brand Sentiment 2026

Pipeline end-to-end Data Science & AI Engineering untuk menganalisis sentimen audiens channel YouTube **NYANTECH** khusus kategori **smartphone per brand** dari seluruh komentar video yang rilis sepanjang tahun **2026**.

## Arsitektur 5 Tahap
1. **Tahap 1 — Data Scraping & Collection (`src/ingestion/`)** *(Ready)*
   - Mengambil daftar video rilis `2026-01-01` s.d. `2026-12-31` (WIB) secara hemat kuota via YouTube Data API v3 (`playlistItems.list` + `videos.list`).
   - Memfilter video yang khusus membahas smartphone (menyingkirkan laptop, tablet, TWS, smartwatch, rakit PC).
   - Mengambil transkrip (`youtube-transcript-api`, prioritas `id` -> `en`) dan seluruh komentar + balasan (`commentThreads.list` + `comments.list`) dengan checkpoint/resume.
2. **Tahap 2 — Entity & Metadata Extraction (`src/extraction/`)** *(Ready)*
   - Deteksi otomatis brand smartphone (`Samsung`, `Apple`, `Xiaomi`, `POCO`, `OPPO`, `Vivo`, `iQOO`, `Realme`, `Nothing`, `Infinix`, `Tecno`, dll.) beserta status video komparasi multi-brand.
   - Ekstraksi harga IDR dari judul/deskripsi/transkrip dan klasifikasi ke segmen: `1-3 juta`, `3-6 juta`, `6-10 juta`, `>10 juta`.
   - Pengelompokan tanggal upload ke kuartal: `Q1 2026`, `Q2 2026`, `Q3 2026`, `Q4 2026`.
   - Dukungan opsional hybrid LLM refiner untuk video yang ambigu.
3. **Tahap 3 — Sentiment & Aspect Analysis (`src/sentiment/`)** *(Next)*
4. **Tahap 4 — Aggregation & 2027 Forecasting (`src/analytics/`)** *(Next)*
5. **Tahap 5 — Streamlit Interactive Dashboard (`src/dashboard/`)** *(Next)*

## Struktur Folder
```text
nyantech-brand-sentiment-2026/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/          # Output Tahap 1 (di-ignore oleh git)
│   ├── processed/    # Output Tahap 2 & 3 (di-ignore oleh git)
│   └── outputs/      # Output agregasi & forecast Tahap 4 (di-ignore oleh git)
├── scripts/
│   ├── run_stage1_scrape.py
│   └── run_stage2_extract.py
├── src/
│   ├── config.py
│   ├── utils/
│   ├── ingestion/    # Tahap 1
│   ├── extraction/   # Tahap 2
│   ├── sentiment/    # Tahap 3
│   ├── analytics/    # Tahap 4
│   └── dashboard/    # Tahap 5
└── tests/
```

## Setup & Menjalankan Tahap 1 & 2

```bash
# 1. Buat virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Siapkan .env (JANGAN PERNAH commit/push file .env!)
cp .env.example .env
# Edit .env dan isi YOUTUBE_API_KEY Anda

# 3. Jalankan Tahap 1 (Scraping 2026)
python scripts/run_stage1_scrape.py

# 4. Jalankan Tahap 2 (Brand, Harga, Kuartal Extraction)
python scripts/run_stage2_extract.py
```

## Menjalankan Unit Test
```bash
pytest
```

# Eksperimen Sentimen Wisata Bali 🏖️

Proyek eksperimen untuk submission akhir kelas **Membangun Sistem Machine Learning (MSML)** Dicoding
oleh **Eka Fanya Yohana Dasilva (NIM 2218068, Teknik Informatika ITN Malang)**.

## 📊 Deskripsi
Eksperimen ini menganalisis **sentimen wisatawan terhadap destinasi pantai di Bali**
menggunakan metode **Random Forest Classifier**.

## 🧩 Dataset
Dataset yang digunakan merupakan hasil preprocessing dari data ulasan wisatawan asli.
File dataset:
📂 `dataset_raw/ulasan_bali_raw.csv`
Berisi teks ulasan (`text`, `clean_text`, `stemmed_text`) dan label sentimen (`sentiment`).

Dataset ini diambil dari file:
`/content/drive/MyDrive/SMSML_EkaFanya/Membangun_model/namadataset_preprocessing.csv`

## ⚙️ Tahapan Eksperimen
1. **Data Loading & Cleaning** — membaca data mentah, membersihkan teks, dan menyiapkan kolom sentimen.
2. **Exploratory Data Analysis (EDA)** — visualisasi sebaran sentimen dan panjang teks.
3. **Preprocessing Otomatis** — dilakukan dengan script `automate_EkaFanya.py`.
4. **Penyimpanan Data Bersih** — hasil akhir disimpan di `preprocessing/data_bersih.csv`.

## 🧠 Teknologi
- Python 3.12.7
- pandas, scikit-learn, nltk, matplotlib, seaborn
- Google Colab

## 📁 Struktur Folder
```
Eksperimen_MSML_EkaFanya/
├── dataset_raw/
│   └── ulasan_bali_raw.csv
├── preprocessing/
│   └── data_bersih.csv
├── automate_EkaFanya.py
├── Eksperimen_MSML_EkaFanya.ipynb
└── screenshot_notebook.png
```

## ✨ Catatan
Dataset ini **asli (bukan dummy)** dan merupakan hasil preprocessing proyek utama *SMSML_EkaFanya*.
Semua cell notebook dapat dijalankan tanpa error.

---
**Eka Fanya Yohana Dasilva**  
Dicoding Academy | *Membangun Sistem Machine Learning* 🌱

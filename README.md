# Deteksi DNS Tunneling: fitur per-event vs. per-event + sliding window

Proyek Python yang modular dan reproducible untuk membandingkan dua pendekatan pada log DNS Zeek (JSON):

* **A. Baseline**: klasifikasi per DNS event, tanpa sliding window.
* **B. Sliding window**: fitur per-event **ditambah** agregasi trailing-window per `(src_ip, SLD)` untuk W ∈ {5, 10, 15, 30, 60} detik.

Empat classifier yang sama (Random Forest, LightGBM, XGBoost, MLP) dipakai pada kedua pendekatan:
4 baseline + 4 × 5 window = **24 konfigurasi utama**.

> **Peringatan.** Angka apa pun yang berasal dari `src/synthetic_data.py` adalah *contoh smoke test*, **bukan hasil penelitian**.
> Repo ini tidak memuat hasil/metrik penelitian.

Dokumen lanjutan: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) (definisi fitur, window, kebijakan leakage, keterbatasan).

---

## 1. Asumsi dan keputusan desain

| Topik | Keputusan |
|---|---|
| Label | Dari file sumber (`benign.json`=0, `tunnel.json`=1). Tidak pernah dibaca oleh kode fitur/window. Rasio aktual dilaporkan, tidak dipaksa 7:1. |
| Format input | Auto-detect JSON Lines vs JSON array (karakter non-spasi pertama `[`), atau `--input-format`. Record rusak dihitung per alasan dan tidak menghentikan program (kecuali melebihi `validation.max_invalid_fraction`). |
| Event ID | `evt_` + 16 hex pertama `sha1("<benign/tunnel>:<posisi record>")`. Unik, deterministik, dan **tidak berkorelasi dengan label** saat dipakai sebagai tie-breaker urutan stabil (ID seperti `benign_000123` akan membocorkan label pada timestamp yang sama). File sumber & nomor baris tetap disimpan sebagai metadata. |
| Field Zeek hilang | Dipetakan lewat `field_mapping` (nama kandidat; `id.orig_h` datar maupun objek bersarang). Field yang tidak ada → `None`; ketersediaan tingkat-dataset dilaporkan di `dataset_summary.json`. |
| `answers` / `TTLs` | Field tidak ada sama sekali di dataset → fitur turunan `NaN`. Field ada tetapi unset pada suatu event (Zeek menghilangkan field unset) → 0 answers / 0 TTL. Statistik TTL dari list kosong = `NaN`, diimputasi dengan median **train**. |
| SLD | Registrable domain (eTLD+1) via `tldextract` + Public Suffix List bawaan paket, tanpa jaringan (§6). |
| Duplikat | Record dengan semua field kanonik identik **dan** label sama dihapus sebelum split; jumlahnya dilaporkan. Konten identik dengan label berbeda tidak dihapus, hanya dihitung (`label_conflict_events`). |
| Split | Dihitung sekali, sebelum fitting/balancing apa pun. Manifest disimpan dan dipakai semua model & window. Strategi: `stratified_random` (default, optimistis untuk time series), `chronological` (`global`/`per_class`), dan `group_sld` (semua event satu SLD masuk satu subset; uji generalisasi ke domain baru dan tidak memotong grup window). Lihat `docs/METHODOLOGY.md` §3b. |
| Audit artefak | Tiap run mengaudit (a) field yang ketersediaannya berbeda drastis antar kelas (hanya dari train) dan secara default membuang fitur turunannya, (b) AUC satu-fitur, kemurnian kategori, vektor fitur identik train/test, dan tumpang-tindih query/SLD. Hasil di `dataset_summary.json` dan `RUN_NOTES.md`. Ablasi: `--feature-set query_only`, `--exclude-features`. |
| Balancing | Default `random_oversampling`, di dalam `imblearn.Pipeline`, hanya pada matriks fitur train (setelah fitur window terbentuk). |
| Kategorikal & SMOTENC | Kategorikal di-encode ordinal → sampler → one-hot, sehingga SMOTENC menghasilkan kategori valid. Kategori baru di test → kode `-1` → vektor one-hot nol. |
| SMOTE biasa | Tidak kompatibel dengan kategorikal nominal → **error informatif** (pakai `smotenc`, atau `features.include_categorical: false`). Flag biner 0/1 diperlakukan numerik oleh `smote` (nilai sintetis bisa pecahan; keterbatasan) dan sebagai kategorikal oleh `smotenc`. `k_neighbors` diperkecil otomatis bila minoritas sedikit; error bila < 2 sampel minoritas. |
| Class weight | Default `none`. `class_weight` + oversampling **ditolak** kecuali `balancing.allow_class_weight_with_resampling: true`. MLP tidak punya `class_weight` (diabaikan dan dicatat). |
| Scaling | Untuk MLP, dan untuk model lain hanya bila balancing = `smote`/`smotenc` (jarak kNN butuh skala sebanding). Di-fit pada train. |
| Threshold | Default tetap 0.5. `threshold.mode: validation_f1` memilih threshold pada validation split **dari train**. Test tidak dipakai. |
| Tuning | Opsional (`tuning.enabled`): `RandomizedSearchCV` pada train, seluruh pipeline (termasuk resampling) di dalam fold. Default nonaktif. |
| Seed | `random_seed=42` untuk split, sampler, dan semua classifier. |

## 2. Diagram alur

```
benign.json ─┐
             ├─► load + validasi (label dari file) ─► dedupe ─► SPLIT sekali (70/30, seed 42)
tunnel.json ─┘                                                      └─► split_manifest.csv
                                          ┌─────────────────────────────┴───────────────────────┐
                                        TRAIN                                                  TEST
                                          │                                                      │
                     fitur per-event (stateless, tanpa label)              fitur per-event (stateless)
                                          │                                                      │
                  window W dari event TRAIN saja                        window W dari event TEST saja
                                          ▼                                                      │
  imblearn.Pipeline.fit:  impute(+scale) → ordinal → oversample → one-hot → classifier           │
                          (semua fitting & resampling hanya di sini, hanya data train)            │
                                          └───────────► model terlatih ─────────────────────────►┤
                                                          transform (tanpa sampler) → predict_proba
                                                                                                  ▼
                                          metrik, kurva, laporan, importance → experiments/<id>/ ; results.csv ; figures/
```

## 3. Struktur proyek

```
dns_tunneling_detection/
├── requirements.txt
├── README.md
├── docs/METHODOLOGY.md
├── config.yaml
├── pytest.ini
├── main.py                     # CLI
├── src/
│   ├── config.py               # default, load YAML, override CLI, validasi
│   ├── data_loader.py          # JSONL/array, mapping field, label, event_id
│   ├── validation.py           # parser/validator field, ValidationReport
│   ├── domain_utils.py         # normalisasi, eTLD+1, subdomain, kasus khusus
│   ├── feature_extraction.py   # fitur per-event + definisi matematis
│   ├── window_features.py      # trailing window (deque), streaming extractor
│   ├── split.py                # dedupe, split (stratified/chronological/group_sld), manifest
│   ├── feature_policy.py       # feature_set/exclude, audit artefak skema & kebocoran
│   ├── preprocessing.py        # ColumnTransformer, sampler (ROS/SMOTE/SMOTENC)
│   ├── models.py               # classifier + imblearn Pipeline
│   ├── evaluation.py           # metrik, threshold validasi, plot
│   ├── experiment.py           # orkestrasi + penyimpanan artefak
│   └── synthetic_data.py       # data sintetis HANYA untuk smoke test
├── data/synthetic/             # contoh kecil (sintetis): benign.json (JSONL), tunnel.json (JSON array)
└── tests/                      # unit + integration test
```

Penyimpangan dari struktur contoh: ditambah `src/config.py` (default/validasi konfigurasi) dan `src/synthetic_data.py` (data smoke test),
serta `tests/test_loader.py`, `tests/test_experiment.py`.

## 4. Instalasi

```bash
cd dns_tunneling_detection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS: LightGBM/XGBoost butuh OpenMP  ->  brew install libomp
```

Diuji dengan Python 3.14, pandas 3.0, scikit-learn 1.9, imbalanced-learn 0.14, LightGBM 4.7, XGBoost 3.4, tldextract 5.3.
Untuk reproduksi ketat, simpan versi (`pip freeze > requirements.lock`); setiap run menulis `environment.json`.

## 5. Menjalankan

```bash
# Data riil: 24 konfigurasi dengan default
python main.py --benign data/benign.json --tunnel data/tunnel.json \
               --config config.yaml --output outputs/experiment_01

# Smoke test data sintetis (hasil BUKAN hasil penelitian)
python -m src.synthetic_data --out data/synthetic --n-benign 2800 --n-tunnel 400
python main.py --benign data/synthetic/benign.json --tunnel data/synthetic/tunnel.json \
               --output outputs/smoke --synthetic --n-jobs 4

# Variasi (tambahkan ke perintah di atas)
--balancing smotenc            # atau: none | random_oversampling | smote
--sampling-strategy 0.5        # rasio minoritas/mayoritas setelah resampling
--models random_forest xgboost --windows 10 30
--windows                      # tanpa nilai: baseline saja
--skip-baseline
--split-strategy chronological --chronological-mode per_class   # bila kedua file beda periode rekam
--split-strategy group_sld                                      # uji ke domain yang belum pernah dilihat
--feature-set query_only                                        # ablasi: tanpa fitur respons/protokol
--exclude-features num_ttls qtype_cat win_ttl_mean              # buang fitur tertentu
--schema-artifact-policy warn                                   # exclude (default) | warn | off
--split-strategy chronological --window-history-policy train_carryover --output outputs/experiment_01_carryover
--split-manifest outputs/experiment_01/split_manifest.csv

pytest -q                      # semua test
```

Opsi CLI: `--benign --tunnel --config --output --input-format --split-strategy --chronological-mode --test-size --random-seed --split-manifest
--feature-set --exclude-features --schema-artifact-policy --balancing --sampling-strategy --models --windows --skip-baseline --window-history-policy --n-jobs --synthetic --overwrite --log-level`.
Prioritas konfigurasi: default kode < `config.yaml` < argumen CLI. Direktori output yang tidak kosong ditolak kecuali `--overwrite`.

Default: `test_size=0.30`, `random_seed=42`, `split_strategy=stratified_random`, `balancing=random_oversampling`,
`windows=[5,10,15,30,60]`, `models=[random_forest, lightgbm, xgboost, mlp]`.
`--n-jobs` dipakai untuk menghitung window secara paralel dan untuk thread model.

## 6. SLD untuk grouping

SLD = registrable domain (eTLD+1): `a.b.example.com → example.com`, `a.example.co.id → example.co.id`.
`tldextract` dikonfigurasi `suffix_list_urls=()` dan `cache_dir=None`: **tidak pernah mengakses jaringan**, hanya memakai snapshot PSL di dalam paket
(versi paket dan SHA-256 snapshot ditulis ke `dataset_summary.json → suffix_list`). Bagian privat PSL (`github.io`, dll.) tidak dipakai secara default (`domain.include_private_suffixes`).

| Kasus | Perlakuan SLD |
|---|---|
| `registrable` | eTLD+1 normal |
| `public_suffix` (`co.id`) | query itu sendiri |
| `single_label` (`wpad`) | label itu sendiri |
| `local` (`x.office.local`; suffix di `domain.local_suffixes`) | satu label + suffix lokal |
| `unlisted_suffix` (TLD tak dikenal PSL) | dua label terakhir |
| `ip` | literal IP itu sendiri (satu grup per IP) |
| `reverse_dns` | `in-addr.arpa`: 2 label terdekat suffix (/16); `ip6.arpa`: 4 nibble |
| `invalid` (label kosong/>63, panjang >253, karakter di luar `[a-z0-9_*-]`) | karakter diganti `_`, label kosong dibuang, lalu aturan PSL dipakai pada hasil *salvage*; **tidak** dijadikan satu grup raksasa |
| `empty` | `__empty__`; karena grup = `(src_ip, SLD)`, menjadi satu grup per src_ip |

`get_registrable_domain()` dan `get_subdomain()` adalah fungsi terpisah di `src/domain_utils.py`.

## 7. Penjelasan output

```
outputs/experiment_01/
├── results.csv                 # satu baris per konfigurasi (kolom wajib + kolom tambahan)
├── dataset_summary.json        # validasi file, distribusi kelas, dedupe, split, domain kind, versi PSL
├── split_manifest.csv          # event_id, subset(train/test), label, source_file, source_line, timestamp
├── config_used.yaml            # konfigurasi final (setelah override CLI)
├── environment.json            # versi Python & paket
├── RUN_NOTES.md                # ringkasan kebijakan + caveat run ini
├── figures/compare_<metrik>.png
└── experiments/<approach>_w<WW>_<model>/      # mis. sliding_window_w10_lightgbm
    ├── model_pipeline.joblib   # imblearn Pipeline ter-fit (preprocessing + classifier)
    ├── model_params.json       # parameter classifier/sampler, threshold, hasil tuning
    ├── feature_schema.json     # urutan fitur input & nama fitur setelah one-hot
    ├── metrics.json            # semua metrik, classification report, distribusi train sebelum/sesudah
    ├── classification_report.txt, confusion_matrix.csv/.png
    ├── roc_curve.png, precision_recall_curve.png
    ├── predictions.csv         # event_id, y_true, proba_tunnel, y_pred (n baris = n_test)
    └── feature_importance.csv/.png   # hanya RF, LightGBM, XGBoost
```

Kolom `results.csv` wajib (sesuai spesifikasi) ditambah: `n_features_transformed`, `threshold`, `threshold_mode`, `class_weight_policy`,
`preprocessing_seconds`, `model_inference_seconds`, `feature_extraction_test_seconds`, `batch_end_to_end_ms_per_event`, `tuning_enabled`,
`tuning_seconds`, `roc_auc_note`, `train_{neg,pos}_{before,after}`, `data_origin`, `experiment_dir`.

Definisi waktu (**bukan latensi streaming end-to-end**):

| Kolom | Isi |
|---|---|
| `feature_extraction_seconds` | fitur per-event + (untuk window) fitur window, train + test |
| `training_seconds` | satu kali `pipeline.fit` (impute/scale/encode + resampling + classifier) dengan hyperparameter final; waktu tuning/pemilihan threshold terpisah |
| `preprocessing_seconds` | transform test (tanpa sampler) |
| `model_inference_seconds` | `predict_proba` classifier pada matriks test terproses |
| `inference_seconds` | `preprocessing_seconds + model_inference_seconds` |
| `inference_ms_per_event` | `inference_seconds * 1000 / n_test` |
| `batch_end_to_end_ms_per_event` | `(feature_extraction_test + inference_seconds) * 1000 / n_test` (batch offline) |

## 8. Pengujian

```bash
pytest -q
```

Mencakup: loader JSONL & JSON array, label dari file, mapping/field hilang/record rusak, eTLD+1 bertingkat (`example.co.id`),
query kosong/IP/reverse/lokal/invalid, entropy, batas window tepat `t-W`, pemisahan grup `(src_ip, SLD)`, tidak ada event masa depan,
timestamp sama, kesetaraan dengan implementasi naif O(n²), kesetaraan batch vs streaming, isolasi window train/test, carry-over,
test set tidak berubah karena oversampling, preprocessing hanya di-fit pada train, kategori baru di test, SMOTE/SMOTENC, `class_weight` + oversampling,
jumlah prediksi = jumlah event test, split identik untuk semua model/window, dan reproducibility dengan seed sama.

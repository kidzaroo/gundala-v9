# Catatan metodologi

## 1. Fitur per-event (`src/feature_extraction.py`)

Notasi: `q` = query ternormalisasi (lowercase, tanpa trailing dot), `n = |q|` karakter (titik dihitung), `s` = string subdomain.

* **Entropy Shannon**: `H(x) = − Σ_{c ∈ set(x)} p(c) · log₂ p(c)`, dengan `p(c) = count_x(c) / |x|`; `H("") = 0`.
  `query_entropy` memakai x = q, `subdomain_entropy` memakai x = s (titik ikut dihitung sebagai karakter).
* **Rasio**: `ratio_X = count_X / n` untuk X ∈ {digit `0-9`, huruf `a-z`, hyphen}; `unique_char_ratio = |set(q)| / n`. Semua rasio = 0 bila `n = 0`.
* `mean_label_length = Σ len(label) / #label` (0 bila tanpa label).
* `longest_digit_run` / `longest_alpha_run`: panjang run maksimal `[0-9]+` / `[a-z]+` pada q.
* Fitur leksikal lain: `query_length`, `num_labels`, `subdomain_length`, `subdomain_num_labels`, `max_label_length`,
  `digit_count`, `letter_count`, `hyphen_count`, `unique_char_count`; indikator `is_ip_query`, `is_reverse_dns`, `is_single_label`, `is_valid_domain`.
* Protokol/respons: kategorikal `qtype_cat`, `rcode_cat`, `proto_cat` (hilang → `"MISSING"`); `is_nxdomain` (rcode == NXDOMAIN, 0 bila rcode hilang);
  `num_answers`, `num_ttls`, `ttl_min`, `ttl_max`, `ttl_mean`; flag `flag_aa/tc/rd/ra` dan `rejected_flag` (hilang → `NaN`, diimputasi modus train).

Metadata **tidak pernah** menjadi fitur classifier: `src_ip`, `dst_ip`, `query`, `sld`, `subdomain`, `timestamp`, `event_id`, `source_file`, `label`
(`FORBIDDEN_FEATURE_COLUMNS`; dicek oleh `experiment._feature_groups_for` dan test). Metadata hanya untuk grouping, pengurutan, pelacakan, dan pelaporan.

Ekstraksi fitur per-event bersifat *stateless* (tidak belajar dari data, tidak membaca label), sehingga aman dijalankan setelah split untuk train dan test terpisah.

## 2. Sliding window (`src/window_features.py`)

**Definisi.** Untuk event pada waktu `t` dan window `W` detik, window adalah interval setengah-terbuka **`(t − W, t]`**.

* Event tepat pada `t − W` **tidak** dihitung (batas bawah terbuka). Event saat ini dihitung (batas atas tertutup). Event masa depan tidak pernah dihitung.
* Event diproses dalam urutan total stabil `(timestamp, event_id)` di dalam grup `(src_ip, SLD)`. Untuk timestamp yang sama hanya event yang **lebih awal dalam urutan itu** plus event itu sendiri yang berkontribusi; dua event dengan timestamp identik dapat memperoleh fitur window berbeda.
* Timestamp dikonversi ke mikrodetik integer (resolusi Zeek), sehingga perbandingan `ts ≤ t − W` eksak.
* Unit prediksi tetap satu DNS event; label event tidak diubah; tidak ada majority vote. Hanya kolom `WINDOW_INPUT_COLUMNS` yang dibaca, jadi label tidak mungkin masuk ke fitur window.

**Kompleksitas.** Satu pass per grup; `deque` event + `dict` berpenghitung untuk keunikan + monotonic deque untuk maksimum + jumlah bergulir (integer eksak untuk panjang query dan IAT).
Tidak ada scan seluruh dataset per event.

| Fitur (n = jumlah event di window, termasuk event saat ini) | Definisi |
|---|---|
| `win_count` | `n` |
| `win_query_rate` | `n / W` (query per detik) |
| `win_unique_queries`, `win_unique_query_ratio` | jumlah query unik; dibagi `n` |
| `win_unique_subdomains`, `win_unique_subdomain_ratio` | jumlah subdomain unik tak-kosong; dibagi `n` |
| `win_query_len_mean/std/max` | rata-rata, std populasi (ddof=0), maksimum panjang query |
| `win_entropy_mean/std` | rata-rata / std populasi entropy query per-event |
| `win_subdomain_len_mean` | rata-rata panjang subdomain |
| `win_digit_ratio_mean` | rata-rata rasio digit per-event |
| `win_nxdomain_count/ratio` | jumlah event NXDOMAIN; dibagi `n` |
| `win_unique_qtypes`, `win_unique_dst_ips` | jumlah query type / destination IP (tak-null) unik |
| `win_answers_mean` | rata-rata `num_answers` atas event yang terdefinisi (`NaN` bila tidak ada) |
| `win_ttl_mean` | rata-rata `ttl_mean` per-event atas event yang terdefinisi (`NaN` bila tidak ada) |
| `win_iat_mean/std` | rata-rata / std populasi selisih waktu (detik) antar event berurutan **dalam grup & window yang sama** |

Konvensi pengisian (konsisten): std dengan < 2 nilai = `0.0`; bila tidak ada IAT (hanya satu event di window) `win_iat_mean = W`
(event sebelumnya, jika ada, paling sedikit W detik sebelumnya) dan `win_iat_std = 0`. `NaN` pada rata-rata answers/TTL diimputasi dengan statistik train.

Implementasi online `StreamingWindowFeatureExtractor` memakai kelas `GroupWindow` yang sama dan diuji setara dengan versi batch.

## 3. Kebijakan window dan data leakage

**Eksperimen utama (`window_history_policy = split_isolated`)**

1. Dataset dipisah dulu menjadi train/test.
2. Fitur window train dihitung hanya dari event train; fitur window test hanya dari event test.
3. Kebijakan sama untuk semua ukuran window dan classifier. Test `test_train_windows_do_not_read_test_events_and_vice_versa` memverifikasi bahwa setiap pemanggilan agregasi hanya melihat event dari satu subset.

Konsekuensi:

* Pemisahan ini mencegah pencampuran event train/test dalam agregasi (tidak ada event test yang ikut membentuk fitur train, atau sebaliknya).
* Pada **stratified random split**, pemisahan mengambil event secara acak dari deret waktu, sehingga *history* window di tiap subset **tidak lengkap**.
  Train hanya mempertahankan ~70% event tetangga di setiap window, sedangkan test hanya ~30%. Akibatnya fitur berbasis hitungan (`win_count`, `win_query_rate`, `win_unique_*`, `win_nxdomain_count`)
  pada test secara sistematis sekitar 0,43× nilai pada train: **pergeseran distribusi train→test yang merupakan artefak split**, dan berbeda dari deployment dengan history lengkap.
  Pergeseran ini hanya mengenai pendekatan B (window), bukan baseline, sehingga perbandingan A vs B di bawah stratified random split ikut terdistorsi.
* **`group_sld` tidak mengalami masalah ini**: kunci grup window adalah `(src_ip, SLD)`, dan split per SLD tidak pernah memotong satu grup window. Fitur window tiap subset identik dengan yang dihitung pada aliran penuh.
* Karena itu, hasil stratified random split **tidak boleh diklaim sebagai simulasi deployment streaming**.

**Opsi chronological split.** Default tetap menghitung window terpisah pada train dan test.
`--window-history-policy train_carryover` adalah **eksperimen tambahan yang terpisah**: window test boleh memuat event train, tetapi karena window hanya melihat ke belakang,
hanya event train yang terjadi sebelum event test yang berkontribusi; label tidak dipakai. Opsi ini ditolak bila split bukan `chronological`.
Hasilnya ditandai pada kolom `window_history_policy` di `results.csv`; jalankan di direktori output berbeda dan jangan mencampurnya di satu tabel utama tanpa penanda.
Bila `benign.json` dan `tunnel.json` direkam pada rentang waktu berbeda, split chronological global dapat menghasilkan subset satu kelas: program berhenti dengan pesan
informatif; gunakan `split.chronological_mode: per_class` (pembagian kronologis 70/30 di dalam tiap kelas) atau stratified random.

**Kontrol leakage lain**

* Split sebelum augmentasi, balancing, encoding ter-fit, scaling, imputasi, dan feature selection. Test set tidak pernah di-resample.
* Seluruh fitting (imputer, scaler, encoder, sampler, classifier) berada di dalam `imblearn.Pipeline` dan hanya memakai train; sampler tidak aktif saat `predict`/`transform`.
* Oversampling dilakukan pada matriks fitur **setelah** fitur window terbentuk. Tidak ada event sintetis/duplikat yang masuk ke perhitungan window (itu akan mengubah pola trafik).
* Tuning (opsional) dan pemilihan threshold (opsional) hanya memakai train; resampling berada di dalam fold CV.
* Hanya satu manifest split untuk semua 24 konfigurasi.

## 3b. Strategi split: kapan memakai yang mana

| Strategi | Mengukur | Kelemahan |
|---|---|---|
| `stratified_random` | generalisasi i.i.d. (optimistis untuk time series) | near-duplicate & domain berulang di train/test; window train/test menipis berbeda (lihat §3) |
| `chronological` (`global` / `per_class`) | generalisasi temporal; window tetap utuh | bila sesi tunneling sedikit, test hanya memuat 1–2 sesi (varians besar); `global` gagal bila kedua file direkam di periode berbeda |
| `group_sld` | generalisasi ke **domain yang belum pernah dilihat**; window tidak terpotong | butuh beberapa SLD per kelas; bila satu kelas hanya punya sedikit SLD, fraksi test kelas itu bisa melebihi target (warning) atau split gagal dengan pesan jelas |

Rekomendasi: laporkan `chronological` (`per_class` bila periode rekam berbeda) dan `group_sld` sebagai hasil utama; `stratified_random` hanya sebagai pembanding optimistis.

## 3c. Audit artefak dan kebocoran (`src/feature_policy.py`)

Skor mendekati sempurna perlu dijelaskan, bukan dirayakan. Setiap run menulis ke `dataset_summary.json` dan `RUN_NOTES.md`:

* `schema_artifact_audit`: proporsi field terisi per kelas, dihitung **hanya pada train**. Field dengan selisih ≥ `features.schema_artifact_threshold` (default 0,5) ditandai; bila `schema_artifact_policy=exclude` (default) fitur turunannya dibuang dari train *dan* test. Alasannya: jika `answers`/`TTLs`/flag hanya ada pada satu file, keberadaan field itu membedakan kelas karena setup perekaman, bukan perilaku.
* `leakage_audit`: AUC satu-fitur dan kemurnian kategori (train saja); proporsi event test yang vektor fiturnya identik dengan event train; vektor fitur berlabel ganda; tumpang-tindih query dan SLD test-di-train; ditambah daftar peringatan.
* Ablasi: `features.feature_set=query_only` membuang semua fitur turunan respons/protokol (TTL, answers, rcode/NXDOMAIN, qtype, proto, flag, dst IP, dan agregat window turunannya); `features.exclude` membuang fitur tertentu. Keputusan tiap fitur yang dibuang tercatat di `feature_schema.json → features_dropped_by_policy`.

Interpretasi: bila skor tetap ~1 pada `query_only` + `group_sld`/`chronological`, datasetnya memang mudah dipisahkan secara leksikal (hasil sah tetapi tidak informatif untuk trafik nyata). Bila skor turun drastis, sebelumnya skor dipengaruhi artefak atau memorisasi.
Audit ini diagnostik: hasilnya tidak dipakai untuk memilih model atau threshold.

## 4. Metrik dan pelaporan

* Kelas positif = 1 (DNS tunneling). Precision/recall/F1 memakai `pos_label=1`, `zero_division=0`.
* ROC-AUC dan average precision (PR-AUC) memakai probabilitas kelas 1 (`predict_proba`, kolom diambil dari `classes_`), bukan `predict()`.
  Bila test hanya punya satu kelas, ROC-AUC = `NaN` dengan alasan di kolom `roc_auc_note`.
* Threshold 0.5 default. Confusion matrix (TN, FP, FN, TP), false positive rate, classification report disimpan per eksperimen.
* Jangan menyimpulkan model terbaik dari accuracy saja (data tidak seimbang). Bandingkan recall, FPR, F1, ROC-AUC/PR-AUC, dan waktu secara bersama (trade-off).
* Memilih konfigurasi (model/window) berdasarkan skor **test** lalu melaporkannya sebagai hasil final menimbulkan bias seleksi optimistis.
  Untuk klaim final gunakan validation set dari train (atau nested CV) untuk memilih, dan evaluasi independen (mis. periode/jaringan lain) untuk melaporkan.
* Hasil berasal dari satu split dan satu seed; untuk estimasi variansi, ulangi dengan beberapa seed (`--random-seed`) dan laporkan sebaran.

## 5. Artefak model vs state inference

`model_pipeline.joblib` hanya berisi preprocessing ter-fit dan classifier. Fitur window membutuhkan **state tambahan**:
riwayat event W detik terakhir per `(src_ip, SLD)`. Saat deployment state ini harus dijaga oleh komponen streaming
(mis. `StreamingWindowFeatureExtractor`, termasuk `prune()` berkala), dan urutan event harus sesuai waktu. Model baseline tidak memerlukan state semacam itu.
Latensi per-event yang dilaporkan (`inference_ms_per_event`) hanya mencakup transform + `predict_proba` pada batch test dan **bukan** latensi end-to-end.

## 6. Keterbatasan

* **Split acak**: event dari klien/domain yang sama tersebar di train dan test; domain tunneling yang berulang membuat test memiliki "kerabat dekat" di train, sehingga skor cenderung optimistis
  dibanding trafik masa depan atau jaringan lain. Dependensi temporal diabaikan dan history window terpotong (lihat §3). Gunakan chronological split untuk menilai generalisasi temporal.
* **Kualitas label**: label berasal dari file sumber. Jika `benign.json` mengandung query tunneling atau `tunnel.json` mengandung query normal (mis. lookup resolver), label menjadi noisy.
  `label_conflict_events` hanya mendeteksi konten identik dengan label berbeda.
* **Duplikasi**: hanya record identik penuh (termasuk timestamp) yang dihapus. Query yang sama dengan timestamp berbeda tetap dianggap event berbeda. Domain/pola yang dominan tetap menciptakan near-duplicate.
* **Kebocoran artefak dataset**: jika benign dan tunnel direkam di lingkungan berbeda, fitur seperti `proto`, `rcode`, flag DNS, TTL, atau pola waktu dapat membedakan kelas karena artefak perekaman, bukan perilaku tunneling.
* **Informasi log terbatas**: log Zeek `dns.log` tidak memuat ukuran payload DNS, sehingga fitur ukuran paket tidak tersedia; `answers`/`TTLs` kosong pada query tanpa respons.
* **Grouping**: klien di balik NAT/resolver bersama bergabung pada satu `src_ip`; tunneling via banyak SLD tidak tertangkap oleh agregasi per-SLD; ketergantungan pada PSL (sufiks privat tidak dipakai default).
* **Resampling**: oversampling acak menduplikasi baris minoritas (risiko overfit); SMOTE pada flag biner menghasilkan nilai pecahan; `sampling_strategy` bukan rasio 7:1 otomatis.
* **Pemilihan threshold**: 0.5 belum tentu optimal untuk data tidak seimbang; gunakan `threshold.mode: validation_f1` bila perlu, bukan pemilihan berdasarkan test.
* **Reproducibility**: seed tetap untuk semua komponen, tetapi LightGBM/XGBoost dapat sedikit berbeda antar versi/platform/jumlah thread. Versi paket dicatat di `environment.json`.

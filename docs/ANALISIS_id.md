# Analisis — Jawaban atas Pertanyaan Tugas

Dataset EuroSAT, PyTorch, dua tier hardware. Semua angka berasal dari `results/`.

> **Catatan:** Layer B untuk MS masih berjalan (n=3). Angka MS Layer B di bawah berasal dari smoke test (n=1, 1.280 gambar) dan akan diperbarui.

---

## Ringkasan dua tier

| | Sandbox (tanpa GPU) | Kaggle (Tesla T4) |
|---|---|---|
| Core | 4 fisik / 8 logis | 2 fisik / 4 logis |
| Yang diukur | preprocessing saja | preprocessing + training |
| Bottleneck | **CPU** | **GPU** (99,8% terpakai) |
| Speedup terbaik | **5,03×** | **1,00×** |
| Konfigurasi terbaik | process ×8 | sequential |

Kode decode identik. Hasilnya berlawanan karena tahap terlambatnya berbeda.

---

## 1. Apakah multiprocessing selalu lebih cepat daripada sequential?

**Tidak.** Ada kasus di mana multiprocessing justru **lebih lambat**, dan keduanya terukur:

| Kasus | Hasil |
|---|---|
| Kaggle, `process ×6` | 0,99× — **lebih lambat secara statistik** (p = 0,034) |
| Kaggle, `process ×8` | 0,99× — **lebih lambat secara statistik** (p < 0,001) |
| Sandbox, MS, `process ×1`, payload array | **0,55×** — hampir setengah kecepatan sequential |

Kasus `0,55×` paling jelas: satu worker tidak menambah paralelisme sama sekali, tetapi tetap membayar penuh biaya serialisasi data antar-proses.

Multiprocessing hanya membantu jika **tahap yang diparalelkan memang bottleneck-nya**, dan jika biayanya (spawn, IPC, memori) lebih kecil daripada keuntungannya.

---

## 2. Apakah semakin banyak worker selalu semakin cepat?

**Tidak.** Kedua platform menunjukkan puncak lalu penurunan.

Sandbox, RGB, payload scalar:

| Worker | 1 | 2 | 4 | **8** | 16 |
|---|---|---|---|---|---|
| Speedup | 1,00× | 1,91× | 3,32× | **5,03×** | 4,97× |

Kaggle, RGB, 27.000 gambar, n=3:

| Worker | 0 | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| Speedup | 1,00× | 0,99× | 1,00× | 1,00× | 0,99× | 0,99× |

Efisiensi paralel (speedup ÷ worker) di Kaggle turun persis sebagai 1/w: **1,00 → 0,50 → 0,25 → 0,12**. Pola klasik ketika worker tambahan tidak memberi apa pun.

---

## 3. Mengapa terlalu banyak worker justru memperlambat?

Empat mekanisme, semuanya terukur:

**a. Oversubscription dan context switching.** `thread ×4` melakukan **144.103 context switch/detik** dibanding `process ×8` yang hanya 66.593 — dua kali lebih banyak sambil 5,6× lebih lambat.

**b. Biaya transfer data antar-proses (IPC).** Beban CPU identik, hanya payload berbeda:

| Dataset | Worker | payload array | payload scalar | selisih |
|---|---|---|---|---|
| MS | 4 | 1,84× | 3,37× | **1,83×** |
| MS | 8 | 2,83× | 4,54× | 1,60× |
| RGB | 8 | 4,23× | 5,03× | 1,19× |

Mengembalikan data alih-alih satu angka memakan **hingga 45%** speedup yang bisa dicapai. Thread adalah kontrolnya: payload tidak berpengaruh untuk thread (0,91× vs 0,90×) karena thread berbagi memori.

**c. Konsumsi memori.** MS `process ×16` dengan payload array mencapai **+847 MB**, dibanding +144 MB payload scalar dan +34 MB untuk thread.

**d. Kompetisi dengan proses utama.** Di Kaggle, `thread ×8` pada MS menurunkan utilisasi GPU ke **78,5%** (sequential 94%), karena thread decode berebut GIL dengan thread yang mengirim perintah ke GPU.

---

## 4. Apakah multithreading memberikan peningkatan performa?

**Tergantung jenis pekerjaannya — dan kami membuktikan aturannya.**

Eksperimen kontrol GIL (16 tugas, dibagi ke N worker):

| Tugas | thread ×4 | process ×4 |
|---|---|---|
| Loop Python murni (menahan GIL) | **1,00×** | 3,73× |
| SHA-256 (melepas GIL) | **3,04×** | 3,58× |

Thread **tidak memberi apa pun** pada pekerjaan yang menahan GIL, dan hampir linear pada pekerjaan yang melepasnya. Mesin dan pool sama; satu-satunya variabel adalah GIL.

Pada decode nyata:

| Kondisi | thread ×8 |
|---|---|
| Cache panas (CPU-bound) | **0,89×** — kalah dari sequential |
| Cache kosong (I/O-bound) | **1,18×** — menang |

**Thread berbalik tanda** ketika pekerjaan berubah dari CPU-bound ke I/O-bound, karena operasi baca berkas melepas GIL.

Bukti paling ringkas: **thread tertahan di 2,4 core aktif** dari 8 yang tersedia, berapa pun jumlah thread (2, 4, 8, 16 identik). Process mencapai **7,9 core**. Thread secara fisik tidak mampu memakai mesin.

**Kesimpulan: thread hanya membantu jika pekerjaan melepas GIL.** Satu mekanisme ini menjelaskan keempat hasil di atas.

---

## 5. Apa bottleneck utama: CPU, RAM, storage, atau GPU?

**Berbeda di setiap kondisi**, dan masing-masing ada angkanya:

| Kondisi | Bottleneck | Bukti |
|---|---|---|
| Kaggle, T4 | **GPU** | utilisasi 99,8%, idle 0,2%; 12 konfigurasi dalam rentang 0,68% |
| Sandbox, cache panas | **CPU** (GIL untuk thread) | thread mentok 2,4 core, process 7,9 core |
| Sandbox, cache kosong, worker sedikit | **Latensi I/O** | sequential kena penalti 1,35× |
| Sandbox, cache kosong, worker banyak | **CPU lagi** | penalti turun ke 1,09× |
| RAM | **tidak pernah** | puncak +847 MB dari 30,8 GB |
| Bandwidth storage | **tidak pernah jenuh** | MS 605 dari 1.315 MB/s (46%) |

Catatan: RGB cache kosong pada 16 worker mencapai **83% dari batas bacanya**, lebih dekat ke jenuh daripada MS. Batas itu berasal dari **jumlah operasi per detik** untuk 27.000 berkas kecil (3,32 KB), bukan dari bandwidth.

---

## 6. Apakah GPU sempat idle karena menunggu data dari CPU?

**Pada hasil akhir: tidak.** Utilisasi GPU 99,3–99,8% di **semua** 12 konfigurasi, termasuk sequential. Idle hanya 0,2–0,7%.

Alasannya: CUDA berjalan asinkron. Thread utama mengirim perintah ke GPU, langsung kembali, lalu men-decode batch berikutnya sementara GPU bekerja. Jadi CPU dan GPU sudah tumpang-tindih **tanpa worker tambahan sama sekali**. Per batch, GPU butuh 207 ms sedangkan satu core men-decode dalam 139 ms — decode sudah lebih cepat daripada training.

**Dua catatan penting untuk kejujuran laporan:**

1. Pengukuran awal kami sempat menunjukkan GPU idle **37,3%** dan speedup 1,50× dari satu worker. Setelah ditelusuri, itu **artefak urutan page cache**: baseline sequential berjalan pertama sebelum berkas masuk cache, jadi yang terukur adalah waktu baca disk, bukan masalah pipeline. Versi final membaca semua berkas sekali sebelum pengukuran, dan efeknya hilang.

2. GPU **memang bisa** kelaparan data — tetapi penyebabnya bukan kekurangan worker. `thread ×8` pada MS menurunkan utilisasi GPU ke **78,5%**, karena kontensi GIL menunda pengiriman perintah ke GPU. Jadi menambah thread justru **menyebabkan** GPU menunggu.

---

## 7. Berapa jumlah worker yang menghasilkan performa terbaik?

**Tergantung bottleneck-nya:**

| Kondisi | Worker optimal | Speedup |
|---|---|---|
| Kaggle, GPU-bound | **0 (sequential)** | 1,00× |
| Sandbox, cache panas | **8 process** = jumlah core logis | 5,03× |
| Sandbox, cache kosong | **16 process** | 5,69× |

Pada kondisi GPU-bound, sequential menyamai konfigurasi tercepat (`thread ×1`, selisih 0,03 detik, p = 0,91) sambil memakai **CPU paling sedikit**: 0,88 core dibanding 1,47 core (**+67%**) untuk `thread ×1` yang tidak memberi keuntungan apa pun.

Pada cache kosong, optimalnya bergeser **lebih tinggi** (16, bukan 8), karena semakin banyak worker semakin banyak latensi I/O yang bisa disembunyikan.

---

# Kesimpulan

## Apakah multithreading atau multiprocessing dapat meningkatkan performa pipeline Deep Learning? Konfigurasi mana yang terbaik dan mengapa?

**Tergantung sepenuhnya pada tahap mana yang menjadi bottleneck.**

**Pada pipeline training di GPU: tidak bisa ditingkatkan.** GPU sudah terpakai 99,8%, menyisakan 0,2% ruang. Ke-12 konfigurasi berada dalam rentang 0,68%, dan empat di antaranya **lebih lambat secara statistik** daripada sequential. Konfigurasi terbaik adalah **sequential**, karena mencapai performa yang sama dengan CPU dan memori paling sedikit.

**Pada pipeline preprocessing (tanpa GPU): ya, sangat besar.** **Multiprocessing dengan 8 worker memberi 5,03×**, sementara multithreading tidak pernah melampaui sequential.

Mengapa 8 worker: sama dengan jumlah core logis mesin. Mengapa process dan bukan thread: thread tertahan di 2,4 core oleh GIL sedangkan process mencapai 7,9. Mengapa bukan 16: penambahan di atas jumlah core hanya menambah context switching dan memori tanpa menambah kapasitas komputasi.

**Prinsipnya: paralelisme hanya membantu pada bottleneck.** Memparalelkan tahap yang bukan bottleneck tetap memakan CPU dan memori, tetapi tidak menghasilkan apa-apa — terbukti dari `thread ×1` yang memakai 67% CPU lebih banyak untuk peningkatan 0,0%.

## Apakah hasil eksperimen mendukung atau menolak asumsi "semakin banyak thread atau process, semakin cepat proses Deep Learning"?

**Menolak.** Empat bukti independen:

1. **Menambah thread hampir selalu memperburuk.** Rentang 0,84–1,06×, dan tertahan di 2,4 core berapa pun jumlah thread.
2. **Process memuncak lalu menurun.** 8 worker 5,03×, 16 worker 4,97×.
3. **Di pipeline GPU, menambah worker apa pun tidak memberi keuntungan**, dan 6–8 worker **lebih lambat secara signifikan** (p = 0,034 dan p < 0,001).
4. **Satu worker bisa lebih lambat daripada nol worker.** MS `process ×1` = **0,55×**.

Asumsi tersebut hanya benar dalam rentang sempit: ketika tahap yang diparalelkan memang bottleneck, **dan** jumlah worker belum melebihi jumlah core, **dan** biaya transfer datanya kecil. Di luar ketiga syarat itu, menambah worker tidak berpengaruh atau justru merugikan.

---

## Catatan metodologi yang memengaruhi hasil

Empat hal yang sempat menghasilkan angka salah sebelum dikendalikan:

1. **Page cache OS.** Menghasilkan speedup palsu 1,50×. Diatasi dengan membaca semua berkas sekali sebelum pengukuran.
2. **Clock GPU yang menurun karena panas.** Pada T4 yang dingin, clock turun 1044 → 870 MHz sambil memanas 51 → 72 °C — lebih besar daripada perbedaan antar-konfigurasi. Diatasi dengan pemanasan GPU sampai suhu stabil, ditambah pengacakan urutan.
3. **Jumlah thread PyTorch.** Default mengambil semua core, sehingga efek worker tidak bisa ditafsirkan. Di-pin eksplisit dan dilaporkan.
4. **Variasi antar-sesi Kaggle.** Kecepatan decode satu core terukur 459–590 img/s pada sesi berbeda dengan pengaturan identik. **Angka hanya dibandingkan dalam satu sesi.**

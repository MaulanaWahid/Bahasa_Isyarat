Pengenalan Bahasa Isyarat Berbasis Landmark Tangan

Aplikasi Python ini melatih model klasifikasi dari CSV landmark tangan dan
menjalankan prediksi huruf secara real-time melalui webcam.

## Persyaratan

- Python 3.10–3.14 (tergantung versi wheel yang tersedia untuk platform Anda)
- Webcam untuk mode real-time
- File `Datafull train.csv` di folder project (atau jalur CSV lain)

Instalasi (Windows)

Buka PowerShell di folder project:

powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Training

```powershell
python main.py train
```

Secara default, program membaca `Datafull train.csv`, membersihkan label
kosong/whitespace dan baris yang fiturnya bukan angka atau tidak finite,
membagi data secara stratified, menampilkan akurasi dan classification report,
lalu menyimpan model ke `hand_sign_model.joblib`.

Jalur dataset/model dan ukuran data evaluasi dapat diubah:

```powershell
python main.py train --data "D:\dataset\Datafull train.csv" --model "D:\models\hand_sign_model.joblib" --test-size 0.2
```

Kolom fitur selalu dipilih eksplisit sebagai `0x, 0y, ..., 20x, 20y`; kolom
tambahan CSV tidak mengubah urutannya. Baik data training maupun landmark
webcam dinormalisasi dengan menggeser titik wrist ke origin dan membagi dengan
rentang maksimum tangan. Transformasi ini mengurangi perbedaan posisi dan
ukuran tangan, dan memakai fungsi yang sama pada kedua jalur.

## Jalankan webcam

Setelah training berhasil:

```powershell
python main.py webcam
```

Tekan `Q` untuk keluar. Gunakan `--camera 1` bila webcam yang digunakan bukan
kamera index 0. Prediksi menampilkan label dan probabilitas kelas tertinggi.
Probabilitas adalah skor model, bukan jaminan akurasi.

Aplikasi mendukung MediaPipe API lama (`solutions`) dan API Tasks pada versi
baru. Jika memakai Tasks API dan file `hand_landmarker.task` belum tersedia,
aplikasi akan mengunduh model resmi MediaPipe satu kali ke folder project.

## Catatan

- Model hanya mengenali kelas yang tersedia dalam dataset training.
- Hasil nyata dipengaruhi kualitas dan konsistensi dataset, pencahayaan, serta
  posisi tangan.
- File model Joblib sebaiknya hanya dimuat jika dibuat oleh sumber tepercaya.

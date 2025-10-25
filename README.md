# Tugas Individu IF5152 - Computer Vision Pipeline

**Nama:** Adril Putra Merin  
**NIM:** 13522068
**Kelas:** IF5152 Computer Vision

## 📋 Description

Aplikasi Computer Vision terintegrasi yang menerapkan materi minggu 3-6:
- Image Filtering (Gaussian, Median, Sobel)
- Edge Detection (Sobel, Canny dengan berbagai threshold)
- Feature/Interest Points (Harris, FAST, SIFT)
- Camera Geometry & Calibration (Transformasi geometri, kalibrasi kamera)

## 🎯 Unique Features

1. **Automated Pipeline**: Semua komponen dapat dijalankan sekaligus atau individual
2. **Multi-parameter Analysis**: Analisis otomatis dengan berbagai parameter
3. **Statistical Reporting**: Generate CSV dan statistik untuk setiap komponen
4. **Visual Comparison**: Side-by-side comparison untuk semua hasil
5. **Modular Design**: Setiap komponen independent dan reusable

## 📁 Folder Structure

```
├── 01_filtering/
│   ├── image_filtering.py
│   └── generated/
│       ├── *_filtering_comparison.png
|       └── filtering_parameters.csv
├── 02_edge/
│   ├── edge_detection.py
│   └── generated/
|       ├── *_edge_comparison.png
|       ├── *_sampling_analysis.png
|       └── edge_parameters.csv
├── 03_featurepoints/
│   ├── feature_point_detection.py
│   └── generated/
|       ├── *_feature_marking.png
|       ├── feature_statistics.csv
|       └── feature_comparison.png
├── 04_geometry/
│   ├── camera_geometry.py
│   └── generated/
│       ├── *_calibration.png
│       ├── *_transformations.png
│       ├── *_matrices.txt
│       └── geometry_parameters.csv
├── 05_laporan.pdf
├── README.md
└── main.py
```

## 🔧 Installation

### Requirements

Python 3.7 (64-bit) atau lebih baru diperlukan. 

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Important Note
Pada awalnya, saya menggunakan Python 3.10.11 (32-bit), tetapi saya mengalami beberapa error saat melakukan _dependencies installation_. Jadi, saya mengganti versi Python ke versi 3.13.9 (64-bit). Proses error ini kemungkinan karena pada kasus ini _library_ yang digunakan memiliki _binary wheels_ yang tersedia hanya untuk Python 64-bit.  

## 🚀 How to Run

### Option 1: Run Complete Pipeline

Menjalankan semua 4 komponen sekaligus:

```bash
python main.py
```

Pilih opsi `1` untuk menjalankan complete pipeline atau `2` dilanjutkan dengan nomor komponen (misalnya `1` untuk image filtering) untuk memilih individual component untuk dijalankan. 

### Option 2: Run Individual Components

Jalankan setiap komponen secara terpisah:

```bash
# Image Filtering
cd 01_filtering
python image_filtering.py

# Edge Detection
cd 02_edge
python edge_detection.py

# Feature Point Detection
cd 03_featurepoints
python feature_point_detection.py

# Camera Geometry & Calibration
cd 04_geometry
python camera_geometry.py
```

## 📊 Generated Outputs

### 01_filtering/generated/
- **Gambar**: Comparison before-after untuk setiap filter
- **CSV**: Parameter filter yang digunakan (sigma, kernel size, dll)

### 02_edge/generated/
- **Gambar**: Edge detection results dengan berbagai threshold
- **Gambar**: Analisis efek sampling
- **CSV**: Threshold values dan edge pixel counts

### 03_featurepoints/generated/
- **Gambar**: Feature points marking (Harris, FAST, SIFT)
- **Gambar**: Bar chart comparison jumlah features
- **CSV**: Statistik lengkap (count, response values, descriptor dimensions)

### 04_geometry/generated/
- **Gambar**: Checkerboard corner detection dan calibration
- **Gambar**: Geometric transformations (rotation, perspective, affine)
- **TXT**: Transformation matrices dan camera matrix
- **CSV**: Calibration parameters (focal length, principal point)

## 🖼️ Dataset

### Gambar Standar (Wajib)
1. **cameraman.png** - Grayscale 512x512
2. **coins.png** - Coin detection test image
3. **checkerboard.png** - Calibration pattern
4. **astronaut.png** - RGB color image

Semua gambar diload dari `skimage.data` library.

## ⚙️ Parameters

### Filtering
- **Gaussian**: sigma = 1.5
- **Median**: kernel size = 5x5
- **Sobel**: default parameters

### Edge Detection
- **Sobel**: 3x3 kernel
- **Canny**: 
  - Low threshold: 30, 50, 100
  - High threshold: 90, 150, 200
- **Sampling**: 1x, 1/2x, 1/4x

### Feature Points
- **Harris**: threshold = 0.01 * max_response
- **FAST**: threshold = 10
- **SIFT**: default parameters

### Geometry
- **Checkerboard**: 9x6 corners
- **Rotation**: 30 degrees
- **Transformations**: rotation, perspective, affine


## 📚 References

- OpenCV Documentation: https://docs.opencv.org/
- Scikit-image Documentation: https://scikit-image.org/
- Computer Vision Course Materials: IF5152

## 👤 Author

<pre>
  Name  : Adril Putra Merin
  NIM   : 13522068
  Email : <a href="mailto:13522068@std.stei.itb.ac.id">13522068@std.stei.itb.ac.id</a>
</pre>
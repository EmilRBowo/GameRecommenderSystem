# Laporan Proyek Machine Learning - Emil RB

## Project Overview

Dalam industri game yang terus berkembang, jumlah game yang tersedia di berbagai platform semakin meningkat, membuat pemain kesulitan menemukan game yang sesuai dengan preferensi mereka. Tanpa sistem yang efektif, pencarian game yang menarik bisa menjadi proses yang memakan waktu dan kurang efisien. Oleh karena itu, sistem rekomendasi game menjadi solusi penting untuk membantu pemain menemukan game yang sesuai berdasarkan genre, rating, popularitas, atau ulasan pengguna. Sistem rekomendasi dapat memberikan saran yang lebih personal dan relevan, meningkatkan pengalaman bermain, serta mendorong eksplorasi game baru yang mungkin tidak ditemukan secara manual. Penelitian yang ada menunjukkan bahwa pemain mendapat manfaat dari saran yang disesuaikan berdasarkan preferensi dan interaksi mereka sebelumnya, yang pada akhirnya meningkatkan pengalaman bermain mereka [1](https://doi.org/10.1109/cig.2018.8490456). Selain itu, bagi pengembang dan penerbit, sistem ini juga dapat meningkatkan visibilitas game mereka dan membantu meningkatkan angka penjualan dengan menargetkan audiens yang tepat[2](https://doi.org/10.3390/info10120379).

## Business Understanding

Pengembangan sistem rekomendasi game memiliki potensi untuk memberikan banyak manfaat bagi pemain dan platform distribusi game. Sistem ini dapat membantu pemain menemukan game yang sesuai dengan preferensi mereka dengan lebih mudah dan efisien, serta membantu platform meningkatkan keterlibatan pengguna, kepuasan pemain, dan efisiensi dalam penyajian konten game.

### Problem Statements
- Bagaimana cara membuat sistem rekomendasi game yang merekomendasikan pengguna berdasarkan genre game?
- Bagimana membuat model sistem rekomendasi Cosine Similarity?
- Bagaimanna cara mengukur nilai perfoma model sistem rekomendasi yang telah dibangun?

### Goals
- Membuat sistem rekomendasi game yang merekomendasikan pengguna berdasarkan genre game
- Membuat model sistem rekomendasi Cosine Similarity
- Mengukur nilai perfoma model sistem rekomendasi yang telah dibangun

### Solution Approach
Menganalisis data dengan melakukan Exploratory Data Analysis dan melakukan visualisasi. Agar didapatkan model prediksi yang baik maka dilakukanlah data preparation berupa menghapus missing value, memeriksa apakah ada data yang duplikat, mengubah ke list, dan memasukkan ke dictionary. Menggunakan sklearn untuk membuat model cosine similarity.

## Data Understanding
### EDA-Deskripsi Variabel

[Sumber data](https://www.kaggle.com/datasets/thedevastator/video-game-sales-and-ratings)
### Video Game Sales and Ratings Dataset

| Jenis         | Keterangan |
|--------------|------------|
| **Title**    | Video Game Sales and Ratings |
| **Description** | Global Video Game Sales, Ratings, and User Insights |
| **Source**   | Kaggle |
| **Maintainer** | Sumit Kumar Shukla |
| **License**  | Other |
| **Visibility** | Publik |
| **Tags**     | Video Games, Sales, Ratings, User Insights, Gaming Industry |
| **Usability** | 10.00 |

Terdapat 16928 data dan 17 kolom dalam dataframe.
Variabel-variabel pada Video Game Sales and Ratings dataset adalah sebagai berikut:
- **Platform**: Platform tempat game tersedia, seperti PC, PS4, Xbox, dll. (String)  
- **Year_of_Release**: Tahun di mana game dirilis. (Integer)  
- **Genre**: Genre game, seperti Action, Sports, dll. (String)  
- **Publisher**: Perusahaan yang menerbitkan game. (String)  
- **NA_Sales**: Penjualan game di Amerika Utara, dalam juta unit. (Float)  
- **EU_Sales**: Penjualan game di Eropa, dalam juta unit. (Float)  
- **JP_Sales**: Penjualan game di Jepang, dalam juta unit. (Float)  
- **Other_Sales**: Penjualan game di wilayah lain, dalam juta unit. (Float)  
- **Global_Sales**: Total penjualan global game, dalam juta unit. (Float)  
- **Critic_Score**: Skor rata-rata yang diberikan oleh kritikus. (Float)  
- **Critic_Count**: Jumlah kritikus yang mengulas game. (Integer)  
- **User_Score**: Skor rata-rata yang diberikan oleh pengguna. (Float)  
- **User_Count**: Jumlah pengguna yang memberikan ulasan. (Integer)  
- **Developer**: Perusahaan yang mengembangkan game. (String)  
- **Rating**: Rating ESRB dari game, seperti E untuk Everyone, T untuk Teen, atau M untuk Mature. (String)  

## Data Preparation
Pada data terdapat 5365 game dengan nama yang sama. Untuk mengatasi hal tersebut dilakukan  metode dropping. Setelah mendrop data duplicate, missing value dicek dan ditemukan sebanyak 7444. Metode **dropping** diterapkan menggunakan fungsi `drop()` untuk menangani data yang memiliki **missing value** dan **Duplicated Value**. Alasan penggunaan metode ini adalah karena data yang dihapus tidak memiliki dampak signifikan terhadap model.Setelah menghapus data yang mengandung missing value dan duplicated value, jumlahnya berkurang menjadi **4119**.Lalu mengubah dari data series menjadi list dengan to_list(). Setelah itu membuat dictionary untuk menentukan pasangan key-value pada data game_name dan game_genre yang telah disiapkan sebelumnya. Lalu menggunakan fungsi TfidfVectorizer dari library scikit-learn untuk vektorisasi. Selain melakukan vektorisasi, TF-IDF juga melakukan proses tokenisasi pada data. Sehingga, tidak perlu melakukan tokenisasi lagi. TFIDF digunakan pada sistem rekomendasi untuk menemukan representasi fitur penting dari macam-macam genre dalam data yaitu ['action', 'adventure', 'fighting', 'misc', 'platform', 'playing', 'puzzle', 'racing', 'role', 'shooter', 'simulation', 'sports', 'strategy'].Matriks yang dimiliki berukuran (4119, 13). Nilai 4119 merupakan ukuran data dan 13 merupakan matrik kategori game. Untuk menghasilkan vektor tf-idf dalam bentuk matriks, digunakan fungsi todense().

## Modeling
### Cosine Similarity
*Cosine Similarity** adalah metode yang digunakan untuk mengukur tingkat kemiripan antara dua vektor dalam ruang multidimensi. Metode ini menghitung **kosinus sudut** antara dua vektor, di mana dimensi dan magnitudonya direpresentasikan sebagai titik dalam suatu ruang. Nilai **cosine similarity** berkisar antara **-1 hingga 1**:

- **1** menunjukkan bahwa kedua vektor sejajar atau memiliki kemiripan **100%**.  
- **0** berarti vektor saling tegak lurus, menandakan **tidak ada keterkaitan**.  
- **-1** menunjukkan bahwa kedua vektor berlawanan arah secara sempurna atau **100% tidak mirip**.  

Metode ini banyak digunakan dalam **pemrosesan teks** dan **pengelompokan data** untuk menilai kesamaan antara dokumen atau fitur dalam suatu dataset.  


Cosine Similarity dihitung dengan rumus:

    CosineSimilarity(A, B) = (A ⋅ B) / (||A|| × ||B||)


di mana:  

- \( A \cdot B \) adalah **produk titik** antara vektor **A** dan **B**.  
- \( ||A|| \) adalah **norma Euclidean** (magnitudo) dari vektor **A**.  
- \( ||B|| \) adalah **norma Euclidean** (magnitudo) dari vektor **B**.  

Lalu untuk percobaan rekomendasi
```
game_recommendations('Mario Kart Wii')
```
| Name                         | Genre   |
|------------------------------|-------- |
| GRID 2                       | Racing  |
| Pro Race Driver              | Racing  |
| RalliSport Challenge 2       | Racing  |
| Colin McRae Rally 2005       | Racing  |
| Test Drive Unlimited         | Racing  |
| Pure                         | Racing  |
| MotoGP                       | Racing  |
| Rock 'N Roll Racing         | Racing  |
| F1 Career Challenge          | Racing  |
| Midnight Club: Los Angeles   | Racing  |


#### Kelebihan dan Kekurangan Cosine Similarity

##### Kelebihan
- **Mudah Dihitung**: Cosine similarity hanya membutuhkan operasi dot product dan norma vektor, sehingga komputasinya relatif ringan.
- **Tidak Terpengaruh Skala**: Metode ini mengukur kesamaan berdasarkan arah vektor, bukan besarnya nilai, sehingga tetap efektif meskipun terdapat perbedaan skala.
- **Cocok untuk Data Teks**: Sering digunakan dalam pemrosesan bahasa alami (NLP) untuk mengukur kemiripan antar dokumen berbasis vektor TF-IDF atau word embeddings.
- **Efektif untuk Data Sparse**: Dapat digunakan pada data berdimensi tinggi dengan banyak nol (sparse data), seperti representasi teks atau fitur pengguna dalam sistem rekomendasi.

##### Kekurangan
- **Tidak Memperhitungkan Magnitudo**: Karena hanya melihat sudut antara vektor, metode ini tidak memperhitungkan perbedaan panjang vektor, yang bisa menjadi kelemahan dalam beberapa kasus.
- **Kurang Optimal untuk Data dengan Informasi Kuantitatif**: Jika dua vektor memiliki perbedaan skala yang signifikan tetapi pola yang sama, cosine similarity mungkin tidak menangkap perbedaannya secara optimal.
- **Tidak Menangani Nilai Negatif dengan Baik**: Pada beberapa kasus, terutama dalam machine learning atau ekonomi, data bisa memiliki nilai negatif yang memiliki arti penting, tetapi cosine similarity tidak memperhitungkannya dengan baik.
- **Kurang Efektif untuk Data yang Tidak Berbentuk Vektor**: Memerlukan representasi data dalam bentuk vektor, sehingga tidak selalu bisa langsung diterapkan pada semua jenis data tanpa preprocessing.


## Evaluation
Precision adalah metrik penting dalam mengevaluasi kinerja model pengelompokan. Metrik ini membantu memahami seberapa akurat model dalam mengidentifikasi data positif. Nilai precision yang tinggi menunjukkan bahwa model jarang membuat prediksi positif yang salah, sehingga prediksi positifnya lebih dapat dipercaya. 

#### Rumus Precision

Precision dapat dihitung menggunakan rumus berikut:

```
Precision = {TP}/{TP + FP}
```

Dimana:
- **TP (True Positive)**: Jumlah data yang diprediksi positif dan memang benar-benar positif.  
- **FP (False Positive)**: Jumlah data yang diprediksi positif, tetapi sebenarnya negatif.  

Berdasarkan tabel hasil rekomendasi, 10 dari 10 rekomendasi merupakan game dengan genre yang sama dengan Mario Kart WII yaitu Racing, maka tingkat presisinya adalah 100%.

## Referensi
[1] Bertens, P., Guitart, A., Chen, P. P., & Periáñez, Á. (2018). A machine-learning item recommendation system for video games. 2018 IEEE Conference on Computational Intelligence and Games (CIG), 1-4. https://doi.org/10.1109/cig.2018.8490456
[2] Lee, Y. and Jung, Y. (2019). A mapping approach to identify player types for game recommendations. Information, 10(12), 379. https://doi.org/10.3390/info10120379

**---Ini adalah bagian akhir laporan---**

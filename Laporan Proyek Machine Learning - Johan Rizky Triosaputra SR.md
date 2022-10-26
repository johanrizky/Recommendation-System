# Laporan Proyek Machine Learning Sistem Rekomendasi - Johan Rizky Triosaputra


## Project Overview

Sekarang ini kita dapat *browsing* film atau membeli barang pada toko *online*. Juga dalam *browsing* dalam *web* film atau toko *online* tersebut kita biasanya mendapatkan rekomendasi sesuai popularitas menurut pengunjung yang menonton atau membeli barang. Rekomendasi tersebut tentu menggunakan bumbu kecerdasasan buatan yang dibuat oleh manusia. Sistem rekomendasi merupakan bagian yang dipelajari pada *Machine Learning*. Sistem rekomendasi sekarang merupakan bidang yang menarik untuk dipelajari. Mengutip dari dari prosiding [*Comparative Analysis of Machine Learning based Filtering Techniques using MovieLens dataset*](https://www.sciencedirect.com/science/article/pii/S1877050921021165) oleh Mohammed Talha Alam, menurut penelitian Badrul Sarwar dan timnya mengakui bahwa sistem rekomendasi telah menjadi alat penting untuk *e-commerce* saat ini dari data penggunaan di beberapa *web* dan karena kualitas sistem rekomendasi diperlukan dengan skalabilitas yang dapat membantu pelanggan menemukan produk atau sesuatu yang dibutuhkan dan menguntungkan penyedia. Dalam laporan ini akan menjelaskan tahapan membuat sistem rekomendasi berdasarkan proyek saya, dimana didalamnya akan menerapkan model *Machine Learning Recommendation System*. Dimana dalam proyek ini diharapkan dapat menghasilkan *output* untuk membantu rekomendasi *user* untuk menonton film berdasarkan film kesukaan *user*, dari *rating* yang diberikannya seperti *output* yang diharapkan, proyek ini sendiri menggunakan referensi dari web [*movielens.org*](https://movielens.org/). Karena untuk memenuhi ketentuan dari laporan ini, juga akan menjelaskan sistem rekomendasi menggunakan pendekatan *content based filtering* dan *collaborative filtering*.


## Business Understanding

### Problem Statements :

- Apakah pengguna bisa mendapatkan rekomendasi berdasarkan judul film yang dicari pengguna?
- Bagaimana jika pengguna hanya menyukai beberapa film saja sehingga pengguna bingung jika menentukan judul dari yang dicari?
- Manakah yang paling efektif, pendekatan *Content Based Model* atau *Collaborative Filtering*?

### Goals

- Pengguna dapat mengetahui bahwa memasukkan judul saja bisa mendapatkan rekomendasi film yang hampir sama dengan yang dicari.
- Pengguna bisa mendapatkan rekomendasi berdasarkan film yang disukai pengguna.
- Mengetahui perbedaan dari kedua pernyataan masalah tersebut pada solution statments.
- Mengetahui efektif kedua sistem tersebut dari evaluasi.

    ### Solution statements :
    - Menggunakan pendekatan menggunakan sistem *content based filtering* dan *collaborative filtering*.
    - Menggunakan metric evaluasi dari kedua sistem tersebut.


## Data Understanding

Data yang digunakan oleh proyek ini adalah dari website [Kaggle](https://www.kaggle.com/datasets) bernama [Movielens dataset](https://www.kaggle.com/datasets/ayushimishra2809/movielens-dataset) dimana didalam folder movielens tersebut terdapat 2 file .csv yaitu movies.csv yang mempunyai 10.330 baris 3 kolom dan rating.csv yang mempunyai 105.340 baris 4 kolom.

Variabel-variabel pada Movielens dataset adalah sebagai berikut:

- movieId : merupakan ID yang diberikan pada judul film tersebut.
- title : merupakan judul dari film.
- genres : merupakan jenis atau genre pada film.
- userId : merupakan ID pengguna pada website movielens.
- rating : merupakan penilaian dari pengguna pada film yang ditontonnya.
- timestamp : merupakan variabel waktu saat pengguna memberikan penilaian tersebut.


### Exploratory data analysis

Disini akan dijelaskan EDA yang digunakan pada masing- masing model pendekatan yaitu Content Based Filtering dan Collaborative Filtering:

#### EDA Model Content Based Filtering

| # 	| Column 	| Non-Null Count 	| Dtype 	|
|:---:	|:---:	|:---:	|:---:	|
| 0 	| movieId 	| 10329 non-null 	| int64 	|
| 1 	| title 	| 10329 non-null 	| object 	|
| 2 	| genres 	| 10329 non-null 	| object 	|

tabel 1. fungsi info()

| index 	|       userId       	|       movieId      	|       rating       	|      timestamp     	|
|:-----:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|
| count 	|      105339.0      	|      105339.0      	|      105339.0      	|      105339.0      	|
|  mean 	| 364.92453886974437 	| 13381.312476860421 	| 3.5168503593161127 	| 1130423971.9742546 	|
|  std  	| 197.48690452013048 	| 26170.456869194284 	|  1.044872179249583 	| 180266031.94107547 	|
|  min  	|         1.0        	|         1.0        	|         0.5        	|     828564954.0    	|
|  25%  	|        192.0       	|       1073.0       	|         3.0        	|     971100797.5    	|
|  50%  	|        383.0       	|       2497.0       	|         3.5        	|    1115154056.0    	|
|  75%  	|        557.0       	|       5991.0       	|         4.0        	|    1275495998.5    	|
|  max  	|        668.0       	|      149532.0      	|         5.0        	|    1452404919.0    	|

tabel 2. fungsi describe()

| index 	| userId_x 	| movieId 	| rating_x 	| timestamp_x 	|     title     	|    genres    	| userId_y 	| rating_y 	|  timestamp_y 	|
|:-----:	|:--------:	|:-------:	|:--------:	|:-----------:	|:-------------:	|:------------:	|:--------:	|:--------:	|:------------:	|
|   0   	|     1    	|    16   	|    4.0   	|  1217897793 	| Casino (1995) 	| Crime\|Drama 	|    NaN   	|    NaN   	|      NaN     	|
|   1   	|     1    	|    16   	|    4.0   	|  1217897793 	|      NaN      	|      NaN     	|    1.0   	|    4.0   	| 1217897793.0 	|
|   2   	|     1    	|    16   	|    4.0   	|  1217897793 	|      NaN      	|      NaN     	|    9.0   	|    4.0   	|  842686699.0 	|
|   3   	|     1    	|    16   	|    4.0   	|  1217897793 	|      NaN      	|      NaN     	|   12.0   	|    1.5   	| 1144396284.0 	|
|   4   	|     1    	|    16   	|    4.0   	|  1217897793 	|      NaN      	|      NaN     	|   24.0   	|    4.0   	|  963468757.0 	|

tabel 3. cek jumlah rating

| # 	| userId 	| movieId 	| rating_x 	|  timestamp 	| rating_y 	|     title     	|    genres    	|
|:-:	|:------:	|:-------:	|:--------:	|:----------:	|:--------:	|:-------------:	|:------------:	|
| 0 	|    1   	|    16   	|    4.0   	| 1217897793 	|    4.0   	| Casino (1995) 	| Crime\|Drama 	|
| 1 	|    1   	|    16   	|    4.0   	| 1217897793 	|    4.0   	| Casino (1995) 	| Crime\|Drama 	|
| 2 	|    1   	|    16   	|    4.0   	| 1217897793 	|    1.5   	| Casino (1995) 	| Crime\|Drama 	|
| 3 	|    1   	|    16   	|    4.0   	| 1217897793 	|    4.0   	| Casino (1995) 	| Crime\|Drama 	|
| 4 	|    1   	|    16   	|    4.0   	| 1217897793 	|    3.0   	| Casino (1995) 	| Crime\|Drama 	|

tabel 4. Menggabungkan movie dengan rating

  1. Mendefinisikan variabel pada data agar movies.csv dan ratings.csv bisa dipanggil dengan movies dan rating
  2. Cek variabel movies dengan fungsi info() (seperti pada tabel 1) untuk mengetahui tipe datanya. Setelah itu hitung menggunakan len dan genre yang tercatat adalah 938 tipe.
  3. Cek nilai minimal dan maksimal rating yang diberikan pengguna menggunakan describe() (seperti pada tabel 2). Setelah itu hitung jumlah userId, movieId, dan data rating dengan len.
  4. setelah movie dan user dibuat menjadi variabel yang berbeda. Selanjutnya akan dicek untuk mengetahui jumlah rating berdasarkan gabungan dari variabel movie dan user. Seperti pada tabel 3, rating_x adalah rating movie berdasarkan rata- rata dari rating user yaitu rating_y.
  5. Menggabungkan dataframe movie dengan rating, dan genre seperti pada tabel 4.

#### EDA Model Collaborative Filtering

| # 	| userId 	| movieId 	| rating 	|  timestamp 	|
|:-:	|:------:	|:-------:	|:------:	|:----------:	|
| 0 	|    1   	|    16   	|   4.0  	| 1217897793 	|
| 1 	|    1   	|    24   	|   1.5  	| 1217895807 	|
| 2 	|    1   	|    32   	|   4.0  	| 1217896246 	|
| 3 	|    1   	|    47   	|   4.0  	| 1217896556 	|
| 4 	|    1   	|    50   	|   4.0  	| 1217896523 	|

tabel 5. visualisasi variabel df

  1. Karena pada tahap EDA Content Based filtering ratings.csv sudah didefinikan variabelnya menjadi rating. Kita akan mendefinisikan variabel rating disini dengan df, agar lebih mudah seperti tabel 5.


## Data Preparation

Disini akan dijelaskan Data Preparation dari masing- masing pendekatan yaitu Content Based Filtering dan Collaborative Filtering untuk masuk ke tahap modeling:

### Data Preparation Content Based Filtering

![1 cek missing value](https://user-images.githubusercontent.com/81506579/195044715-51ffe9a3-d8d2-4c23-8135-8c3f4ac8ef85.jpg)

gambar 1. cek missing value

![2 menyamakan jenis movie](https://user-images.githubusercontent.com/81506579/195044891-a26c5b92-6fbe-4501-8649-31b6cebb9d89.jpg)

gambar 2. menyamakan jenis movie

![3 mengurutkan movie](https://user-images.githubusercontent.com/81506579/195045076-f5333c43-0319-46b3-83f2-1631df571a3e.jpg)

gambar 3. mengurutkan movie

| # 	| id 	|             movie_name             	|                      genre                      	|
|:-:	|:--:	|:----------------------------------:	|:-----------------------------------------------:	|
| 0 	|  1 	|          Toy Story (1995)          	| Adventure\|Animation\|Children\|Comedy\|Fantasy 	|
| 1 	|  2 	|           Jumanji (1995)           	|           Adventure\|Children\|Fantasy          	|
| 2 	|  3 	|       Grumpier Old Men (1995)      	|                 Comedy\|Romance                 	|
| 3 	|  4 	|      Waiting to Exhale (1995)      	|              Comedy\|Drama\|Romance             	|
| 4 	|  5 	| Father of the Bride Part II (1995) 	|                      Comedy                     	|

tabel 6. membuat dictionary dari movieId, movie_name, dan genres

1. Cek missing value menggunakan isnull(). Agar bisa mengidentifikasi apakah data tersebut masih belum sama. Karena pada gambar 1 tidak ada missing jadi akan lanjut. 
2. Menyamakan movie berdasarkan movieId menggunakan sort_values. Menyamakan movie dengan movieId diperlukan, agar movie dan movieId tidak bias pada data. Karena pada gambar 2 tidak ada yang janggal sehingga bisa lanjut ke tahap berikutnya.
3. Masuk ke tahap preparation dengan mengurutkan fix_movie berdasarkan movieId. Untuk masuk ke tahap pemodelan perlu untuk mengetahui data duplikat, seperti pada gambar 2.
4. Karena pada gambar 3 ada data duplikat, maka selanjutnya buang duplikat data menggunakan drop_duplicate. Agar bisa lanjut ke tahap pemodelan
5. Selanjutnya adalah mengkonversi data series menjadi list, menggunakan fungsi tolist() dari library numpy dan mendefinisikan variabel baru untuk tahap membuat dictionary. Disini variabel movieId akan menjadi movie_id, title akan menjadi  variabel movie_name, dan genres akan menjadi variabel genre
6. Selanjutnya membuat dictionary untuk data movie_id, movie_name, dan movie_genre lalu menyimpannya kedalam variabel movie_new. Membuat dictionary diperlukan untuk menentukan pasangan key-value pada movie_id, movie_name, dan genre berdasarkan variabel pada tahap 5 sebelumnya, sehingga dataframe menjadi seperti tabel 6.

### Data Preparation Collaborative Filtering

![5 hasil sebagian dari encoding](https://user-images.githubusercontent.com/81506579/195045296-ba50efa4-87e8-4941-8f80-b87f22aae2c3.jpg)

gambar 4. hasil sebagian dari encoding userId

![6 cek jumlah data](https://user-images.githubusercontent.com/81506579/195045393-651b0061-7758-495f-b399-77da5f021655.jpg)

gambar 5. cek jumlah data

1. Tahap awal yang dilakukan adalah menyandikan (encode) fitur 'user' dan 'movieId' dalam indeks integer. Hasil dari sebagian encoding fitur 'user' bisa dilihat pada gambar 4. Dalam proses encoding ini dapat memudahkan saat proses mapping userId ke dataframe user dan movieId ke dalam movie, yang selanjutnya 
2. Selanjutnya kita cek data jumlah user, untuk mengetahui jumlah movie dan nilai minimal maksimal dari rating, seperti pada gambar 5.
3. Selanjutnya adalah mengacak dataset terlebih dahulu agar distribusi datanya random, menggunakan fungsi sample(frac=1).
3. Selanjutnya membagi data train dan validasi agar bisa digunakan untuk melatih data dan menguji model. Disini saya menggunakan komposisi 80:20, tetapi sebelum itu diperlukan memetakan data user dan movie menjadi satu value dahulu dengan membuat rating dalam skala 0 sampai 1 agar mudah saat proses training.


## Modeling
Dalam tahap modeling ini akan dijelaskan masing- masing dua pendekatan yang berbeda yaitu Content Based Filtering dan Collaborative Filtering. 

### Content Based Filtering

Sebelum masuk ke algoritma yang digunakan, perlu terlebih assign dataframe dari tahap Data Preparation ke dalam variabel data. Dua algoritma yang digunakan adalah sebagai berikut: 
1. Algoritma TF-IDF Vectorizer
Algoritma ini akan menghasilkan korelasi antara movie dengan genrenya. Algoritma ini menggunakan fungsi tfidfvectorizer() dari library sklearn. TF-IDF Vectorizer ini nantinya akan melakukan mapping terhadap fitur genre, selanjutnya TF-IDF Vectorize akan melakukan fit dan transformasi genre ke dalam bentuk matriks. Untuk menghasilkan vektor tf-idf dalam bentuk matriks, perlu menggunakan fungsi todense(). Selanjutnnya matriks tf-idf akan menghasilkan sebuah baris judul film dan kolom judul, dimana dalam baris kolom tersebut judul yang memiliki kesamaan akan memiliki nilai matriks yang sama.
2. Algoritma Cosine Similarity
Algoritma ini digunakan untuk menghitung derajat kesamaan (similarity degree) antar movie dengan teknik cosine similarity. Algoritma cosine Cosine Similarity akan menggunakan fungsi cosine_similarity dari library sklearn. Pada tahap ini tentu kita akan memanggil fungsi Cosine Similarity dengan dataframe yang dipanngil dari tfidf_matrix TF-IDF Vectorizer, sehingga menghasilkan keluaran berupa matriks kesamaan dalam bentuk array.

| # 	| id 	|    movie_name    	|                      genre                      	|
|:-:	|:--:	|:----------------:	|:-----------------------------------------------:	|
| 0 	|  1 	| Toy Story (1995) 	| Adventure\|Animation\|Children\|Comedy\|Fantasy 	|

tabel 7. memasukkan data Toy Story

| # 	|                     movie_name                    	|                      genre                      	|
|:-:	|:-------------------------------------------------:	|:-----------------------------------------------:	|
| 0 	|               Shrek the Third (2007)              	| Adventure\|Animation\|Children\|Comedy\|Fantasy 	|
| 1 	| Asterix and the Vikings (Astérix et les Viking... 	| Adventure\|Animation\|Children\|Comedy\|Fantasy 	|
| 2 	|               Boxtrolls, The (2014)               	| Adventure\|Animation\|Children\|Comedy\|Fantasy 	|
| 3 	|   Adventures of Rocky and Bullwinkle, The (2000)  	| Adventure\|Animation\|Children\|Comedy\|Fantasy 	|
| 4 	|           Tale of Despereaux, The (2008)          	| Adventure\|Animation\|Children\|Comedy\|Fantasy 	|

tabel 8. mendapatkan rekom dari data toy story

Dalam kedua algoritma tersebut saya menggunakan Cosine Similarity dengan mengambil dataframe cosine_sim_df yang dimasukkan kedalam kode program untuk mendapatkan rekomendasi. Selanjutnya kita akan menerapkan kode tersebut untuk menemukan movie yang mempunyai genre yang sama dengan Toy Story (1995), seperti pada tabel 7., lalu kita akan mendapatkan rekomendasi 5 movie dengan genre yang sama seperti tabel 8.

### Collaborative Filtering

Sebelum memasuki tahap ke Collaborative Filtering, perlu untuk membuat class *RecommenderNet* dengan keras Model class, yang terinsipirasi dari tutorial dalam situs Keras dengan beberapa adaptasi menyesuaikan kasus ini. Tahap yang dilakukan adalah membuat class RecommenderNet(tf.keras.Model) yang didalamnya menginilasasi fungsinya, dan tidak lupa melakukan proses embedding terhadap data 'user' dan 'movie'. Selanjutnya melakukan operasi perkalian dot antara embedding user dengan movie, juga menambahkan bias untuk setiap 'user' dan 'resto' dengan skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.
Setelah menerapkan kode dalam class RecommenderNet selanjutnya menginisialisasi model dengan memanggil RecommenderNet yang didalamnya berisi num_users, dan num_resto (sesuai pada class RecommenderNet). Selanjutnya memanggil model.compile yang didalamnya terdapat parameter:
1. loss = tf.keras.losses.BinaryCrossentropy(), merupakan fungsi loss default dari library tensorflow Keras.
2. optimizer = keras.optimizers.Adam(learning_rate=0.001), merupakan optimizers yang dipanggil dari library Keras. Menggunakan learning_rate=0.001, dimana semakin besar nilai yang diberikan akan semakin cepat proses trainingnya.
3. metrics=[tf.keras.metrics.RootMeanSquaredError(), membuat metric RootMeanSquaredError dari Keras.

|         Judul Film         	|               Genre               	|
|:--------------------------:	|:---------------------------------:	|
|        Scream (1996)       	| Comedy\|Horror\|Mystery\|Thriller 	|
|        Benny & Joon        	|          Comedy\|Romance          	|
| Grosse Pointe Blank (1997) 	|       Comedy\|Crime\|Romance      	|
|  Dangerous Liaisons (1988) 	|           Drama\|Romance          	|
|       Superman (1978)      	|     Action\|Adventure\|Sci-Fi     	|

tabel 9. *Top 5 high ratings from user* 560

|              Judul Film              	|            Genre            	|
|:------------------------------------:	|:---------------------------:	|
|               Godfather              	|         Crime\|Drama        	|
|       Full Metal Jacket (1987)       	|          Drama\|War         	|
|            Henry V (1989)            	| Action\|Drama\|Romance\|War 	|
|            Amadeus (1984)            	|            Drama            	|
|           Annie Hall (1977)          	|       Comedy\|Romance       	|
|     Boot, Das (Boat, The) (1981)     	|      Action\|Drama\|War     	|
|           Sting, The (1973)          	|        Comedy\|Crime        	|
|       Dead Poets Society (1989)      	|            Drama            	|
|         Graduate, The (1967)         	|    Comedy\|Drama\|Romance   	|
| Bridge on the River Kwai, The (1957) 	|    Adventure\|Drama\|War    	|

tabel 10. *top 10 movie recommendation for user* 560

Sebelum melanjutkan ke tahapan mendapatkan rekomendasi movie menggunakan Collaborative Filtering, perlu mengambil user secara acak dan mendefinisikan variabel movie_not_watched yang merupakan daftar movie yang belum pernah ditonton oleh pengguna. karena daftar movie_not_watched yang akan direkomendasikan. Juga sebelumnya pengguna memberi rating pada movie yang pernah ditontonnya, karena akan menggunakan data dari rating tersebut untuk merekomendasikan movie_not_watched. Setelah menerapkan kode untuk mendapatkan rekomendasi, selanjutnya menggunakan fungsi model.predict() untuk mendapatkan top 10 movie rekomendasi untuk pengguna 560 yang menyukai beberapa movie yang mempunyai genre comedy, horror, drama seperti pada tabel 9. Lalu user 560 mendapatkan hasil rekomendasi movie yang hampir serupa dengan riwayat genre movie yang ditonton, hasil dapat dilihat pada tabel 10.


## Evaluation

### Evaluasi Content Based Model

Pada evaluasi Content Based Model saya menggunakan metric precision dengan formula seperti berikut:

recommender system precision = p

**p = #of recommendation that are relevant / #of item we recommend**

sesuai pada tabel 8, lima genre yang keluar sama dengan genre Toy Story (1995) (sesuai tabel 7). Sehinggan precision yang didapat adalah 5/5 atau 100%.

### Evaluasi Collaborative Filtering

![model RMSE](https://user-images.githubusercontent.com/81506579/195130669-7464c6e8-1245-4207-9e36-01781dc4a8b8.jpg)

gambar 6. *plot model metric root_mean_squared_error*

Pada evaluasi Collaborative Filtering disini menggunakan Roor Mean Squared Error (RMSE) sebagai metrics evaluasinya. Dalam metric ini mentraining data, mengetahui jumlah loss, dan mengetahui jumlah RMSE-nya dengan memanggil model fit dengan data latih dan data uji yang sudah didefinisikan pada data preparation sebelumnya. Disini akhir rmse dalam model collaborative mendapatkan nilai error akhir sebesar sekitar 0.20 dan error pada data validasi sebesar 0.21 seperti yang terlihat pada gambar 6. Untuk melihat nilai tersebut cukup bagus atau tidak tergantung dari hasil sesuai pada tabel 10.


# Kesimpulan dari Pendekatan Content Based Filtering dan Collaborative Filtering

Menurut saya kedua pendekatan tersebut memiliki kelebihan masing- masing karena kedua pendekatan tersebut sangat berguna bila digunakan sesuai kebutuhan yang diperlukan. 
Pertama adalah Content Based model. Pendekatan ini sangat berguna disini bila pengguna akan mencari genre yang sama dari sebuah film maka pengguna akan diberi langsung daftar genre sesuai yang ingin dicari, misalnya pengguna mencari sebuah genre drama maka pengguna akan diberi rekomendasi beberapa genre drama yang tersedia. Kekurangan dari pendekatan ini adalah pengguna bisa saja tidak cocok dengan film yang direkomendasikan oleh sistem tersebut, karena tidak semua genre drama sesuai yang ingin dicari. Misalnya pengguna menyukai genre drama yang dipadukan dengan action tanpa romansa, sistem bisa saja memberi rekomendasi genre drama dengan romansa yang kental dan tanpa genre action.
kedua adalah pendekatan Collaborative Filtering. Pendekatan ini sangat berguna bila pengguna sudah mempunyai riwayat menonton film sehingga sistem bisa merekomendasikan film yang sesuai bahkan hampir sama genrenya dengan riwayat yang pengguna tonton. Kekurangan dari pendekatan ini adalah pengguna harus menonton beberapa film lalu memberi rating pada film yang dia tonton, sehingga sistem dapat merekam jejak genre yang pengguna sukai.
Kalau untuk memilih saya lebih memilih ke Collaborative Filtering, karena pengguna bisa direkomendasikan film sesuai riwayat yang ditonton. Sehingga saat pengguna baru masuk aplikasi tersebut pengguna dapat direkomendasikan menggunakan pendekatan Content Based Model.


# Referensi
1. M. T. Alam, S. Ubaid, Shakil, S. S. Sohail, M. Nadeem, S. Hussain and J. Siddiqui, "Comparative Analysis of Machine Learning based," in Procedia Computer Science, New Delhi, 2021. 
2. B. Sarwar, G. Karypis, J. Konstan and J. Riedl, "Analysis of Recommendation Algorithms for E-Commerce," in Conference on Electronics Commerce, Minneapolis, 2000. 




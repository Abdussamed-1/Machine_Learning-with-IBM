# IBM ile Makine Öğrenmesi ve Derin Öğrenme

IBM Machine Learning eğitimi kapsamında kullanılan laboratuvarları, uygulamaları ve beyin MR görüntülerini sınıflandıran bir final projesini içeren öğrenme deposu. Python, Jupyter Notebook ve TensorFlow/Keras üzerinden temel sinir ağlarından evrişimli ağlara, sıralı veri modellerinden üretici modellere uzanan konular ele alınır.

Depoda **29 Jupyter notebook** bulunur. Final projede ImageNet ağırlıklarıyla başlatılan **ResNet50** modeli kullanılarak transfer öğrenme uygulanır.

## İçindekiler

- [Projenin amacı](#projenin-amacı)
- [Depo yapısı](#depo-yapısı)
- [Eğitim içerikleri](#eğitim-içerikleri)
- [Final proje: Beyin MR görüntüsü sınıflandırma](#final-proje-beyin-mr-görüntüsü-sınıflandırma)
- [Kurulum](#kurulum)
- [Notebook’ları çalıştırma](#notebookları-çalıştırma)
- [Sonuçları değerlendirme](#sonuçları-değerlendirme)
- [Bilinen sınırlamalar ve geliştirme alanları](#bilinen-sınırlamalar-ve-geliştirme-alanları)
- [Katkı ve lisans](#katkı-ve-lisans)

## Projenin amacı

Bu çalışma, makine öğrenmesi ve derin öğrenme kavramlarını uygulamalı örneklerle pekiştirmek için hazırlanmıştır. Notebook’lar üzerinden şu becerilerin geliştirilmesi hedeflenir:

- Gradyan inişi, geri yayılım ve optimizasyon yöntemlerini anlamak.
- Keras ile sinir ağları kurmak, eğitmek ve karşılaştırmak.
- Görüntü verilerini hazırlamak ve CNN tabanlı sınıflandırma yapmak.
- Önceden eğitilmiş modellerden transfer öğrenme ile yararlanmak.
- Metin ve sıralı veriler için embedding, RNN, LSTM ve GRU yapılarını incelemek.
- Autoencoder, varyasyonel autoencoder ve GAN örneklerini keşfetmek.
- Model performansını eğitim eğrileri ve sınıflandırma metrikleriyle yorumlamak.

## Depo yapısı

```text
Machine_Learning-with-IBM/
├── README.md
├── Final_Project_IBM/
│   ├── Project-IBM.ipynb
│   ├── Brain_Tumor.jpg
│   └── archive/
│       ├── Training/
│       │   ├── glioma/
│       │   ├── meningioma/
│       │   ├── notumor/
│       │   └── pituitary/
│       └── Testing/
│           ├── glioma/
│           ├── meningioma/
│           ├── notumor/
│           └── pituitary/
└── IBM - Machine Learning/
    ├── Machine-Learning-Notebooks/
    └── Keras-With-IBM/
        └── Lab_Keras_İntro/
            ├── 05d_LAB_Keras_Intro.ipynb
            └── diabetes.csv
```

`Brain_Tumor.jpg`, final notebook’ta kullanılan tanıtım görselidir. Eğitim ve test görüntüleri `archive/` altındaki sınıf klasörlerinde bulunur.

## Eğitim içerikleri

### Makine öğrenmesi ve sinir ağı temelleri

`IBM - Machine Learning/Machine-Learning-Notebooks/` klasörü altındaki 6 notebook:

| Notebook | Konu |
| --- | --- |
| `05a_DEMO_Gradient_Descent.ipynb` | Doğrusal regresyon, gradyan inişi, öğrenme oranı ve SGD |
| `05b_LAB_Intro_NN.ipynb` | Nöronlar, mantık kapıları ve sinir ağlarına giriş |
| `05c_DEMO_Backpropagation.ipynb` | İleri hesaplama, geri yayılım ve ağırlık güncelleme |
| `lab_jupyter_optimizers.ipynb` | Optimizasyon algoritmalarının karşılaştırılması |
| `types-of-sequential-data.ipynb` | Zaman serisi, metin ve ses gibi sıralı veri türleri |
| `05i_DEMO_Reinforcement.ipynb` | FrozenLake ortamı üzerinden pekiştirmeli öğrenme |

### Keras ve derin öğrenme uygulamaları

`IBM - Machine Learning/Keras-With-IBM/` klasörü, alt klasörüyle birlikte 22 notebook içerir:

| Alan | İlgili notebook’lar |
| --- | --- |
| Keras temelleri | `Lab_Keras_İntro/05d_LAB_Keras_Intro.ipynb`, `introduction_to_models_in_keras.ipynb` |
| Kayıp fonksiyonu ve düzenlileştirme | `CategoricalCrossEntropy.ipynb`, `Regularization_techniques.ipynb` |
| Hiperparametreler ve donanım | `lab_paramopt_keras.ipynb`, `lab_GPU_keras.ipynb` |
| Görüntü yükleme ve evrişim | `LoadingImages.ipynb`, `image_convolution_lab.ipynb`, `Pad_Stride_Pool_Act.ipynb` |
| CNN mimarisi | `05e_DEMO_CNN.ipynb`, `BuildingCNN_Filters_Layers_Flattening_and_Receptive_Field.ipynb` |
| Transfer öğrenme | `05f_DEMO_Transfer_Learning.ipynb`, `TransferLearningExampleLab.ipynb` |
| Metin ve sıralı modeller | `Embeddings.ipynb`, `05g_DEMO_RNN.ipynb`, `LSTM_GRU_demo.ipynb`, `pre-trained-RNNs.ipynb` |
| Autoencoder modelleri | `Autoencoders.ipynb`, `05h_LAB_Autoencoders.ipynb`, `Variational_Autoencoders.ipynb` |
| GAN modelleri | `GANS_lab1.ipynb`, `GANsPart2.ipynb` |

Önerilen öğrenme sırası: gradyan inişi → temel sinir ağları → geri yayılım → Keras → CNN → transfer öğrenme → sıralı modeller → autoencoder ve GAN. Pekiştirmeli öğrenme örneği ayrı olarak incelenebilir.

## Final proje: Beyin MR görüntüsü sınıflandırma

[Final proje notebook’unu aç](Final_Project_IBM/Project-IBM.ipynb)

Final çalışma, beyin MR görüntülerini dört kategoriye ayıran bir görüntü sınıflandırma uygulamasıdır. Sınıf etiketleri görüntülerin bulunduğu klasör adlarından alınır.

### Veri kümesi

Depodaki JPG dosyalarının sınıflara göre dağılımı:

| Sınıf klasörü | Açıklama | Training | Testing | Toplam |
| --- | --- | ---: | ---: | ---: |
| `glioma` | Glioma sınıfı | 1.321 | 300 | 1.621 |
| `meningioma` | Meningioma sınıfı | 1.339 | 306 | 1.645 |
| `notumor` | Tümör bulunmayan görüntü sınıfı | 1.595 | 405 | 2.000 |
| `pituitary` | Hipofiz tümörü sınıfı | 1.457 | 300 | 1.757 |
| **Toplam** | | **5.712** | **1.311** | **7.023** |

Notebook, `Training/` içeriğinin tamamını eğitim için kullanır. `Testing/` içeriği ise `train_test_split(test_size=0.5, random_state=43)` ile **655 doğrulama** ve **656 test** görüntüsüne ayrılır. Bu bölmede `stratify` kullanılmadığı için sınıf oranları birebir korunmaz.

### İşlem akışı

1. Görüntü yolları ve klasörlerden türetilen etiketler Pandas DataFrame’lerine aktarılır.
2. Doğrulama ve test kümeleri oluşturulur.
3. `ImageDataGenerator.flow_from_dataframe` ile görüntüler batch’ler halinde yüklenir; 224 × 224 piksel RGB biçimine dönüştürülür ve kategorik etiketler hazırlanır.
4. Örnek görüntüler sınıf adlarıyla görselleştirilir.
5. Dondurulmuş ResNet50 tabanının üzerine sınıflandırma katmanları eklenir.
6. Model eğitilir; eğitim ve doğrulama kaybı ile doğruluk eğrileri çizilir.
7. Eğitim, doğrulama ve test kümeleri değerlendirilir; test tahminlerinden karışıklık matrisi ve sınıflandırma raporu üretilir.

### Model mimarisi ve eğitim ayarları

```text
Girdi: 224 × 224 × 3
    ↓
ResNet50 (ImageNet ağırlıkları, include_top=False, dondurulmuş)
    ↓
Flatten
    ↓
Dense(256, ReLU) → Dropout(0.5)
    ↓
Dense(64, ReLU) → Dropout(0.5)
    ↓
Dense(4, Softmax)
```

| Parametre | Notebook’taki değer |
| --- | --- |
| Batch boyutu | 16 |
| Epoch sayısı | 10 |
| Optimizasyon | Adamax |
| Öğrenme oranı | 0.001 |
| Kayıp fonksiyonu | `categorical_crossentropy` |
| Eğitim metriği | `accuracy` |
| Test üretecinde karıştırma | `shuffle=False` |

Çıkış katmanının boyutu kodda sınıf sayısından dinamik olarak hesaplanır. ResNet50 tabanı dondurulduğundan eğitim sırasında eklenen sınıflandırma katmanlarının ağırlıkları güncellenir.

## Kurulum

### 1. Depoyu klonlayın

```bash
git clone https://github.com/Abdussamed-1/Machine_Learning-with-IBM.git
cd Machine_Learning-with-IBM
```

### 2. Sanal ortam oluşturun

Python 3 ve seçtiğiniz Python sürümüyle uyumlu bir TensorFlow kurulumu gerekir. Depoda sabitlenmiş bir bağımlılık veya doğrulanmış sürüm matrisi bulunmaz.

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Sisteminizde Python komutu `python3` ise ortam oluştururken `python` yerine `python3` kullanın.

### 3. Temel paketleri yükleyin

Final proje ve temel örnekler için başlangıç kurulumu:

```bash
python -m pip install --upgrade pip
python -m pip install jupyterlab ipykernel tensorflow numpy pandas matplotlib scikit-learn pillow seaborn
python -m ipykernel install --user --name ibm-ml --display-name "Python (IBM ML)"
```

Depoda `requirements.txt` bulunmadığı için paketler doğrudan kurulmaktadır. Bu liste bütün laboratuvarların ek bağımlılıklarını kapsamaz. Seçtiğiniz notebook’un import ve kurulum hücrelerini inceleyin; bazı örnekler `skillsnetwork`, `mlxtend`, `opencv-python`, `nltk`, `statsmodels`, `scipy`, `tqdm` veya `gym` gibi ek paketler kullanır.

## Notebook’ları çalıştırma

```bash
jupyter lab
```

Jupyter arayüzünde ilgili notebook’u açın, **Python (IBM ML)** kernel’ini seçin ve hücreleri yukarıdan aşağıya sırayla çalıştırın. Göreli dosya yolları kullanan laboratuvarlarda çalışma dizininin notebook’un bulunduğu klasör olduğundan emin olun.

### Final proje için dosya yollarını düzenleme

Final notebook’ta `train_data` ve `test_data` değişkenleri, geliştiricinin bilgisayarına ait mutlak Windows yollarını içerir. Çalıştırmadan önce bu iki atamayı kendi ortamınıza göre değiştirin.

Kernel’in çalışma dizini `Final_Project_IBM/` olduğunda:

```python
from pathlib import Path

project_dir = Path.cwd()
train_data = str(project_dir / "archive" / "Training")
test_data = str(project_dir / "archive" / "Testing")

assert Path(train_data).is_dir(), train_data
assert Path(test_data).is_dir(), test_data
```

Kernel depo kökünde çalışıyorsa `project_dir = Path.cwd() / "Final_Project_IBM"` kullanın. `Path.cwd()` değerini kontrol ederek doğru dizini seçebilirsiniz.

İlk çalıştırmada ResNet50’nin ImageNet ağırlıkları indirilebilir. Eğitim laboratuvarlarının bir kısmı da verileri veya görselleri internetten indirir; bu hücreler için internet bağlantısı gerekir. GPU, görüntü modellerinin eğitimini hızlandırabilir; eğitim süresi donanıma göre değişir.

## Sonuçları değerlendirme

Final notebook aşağıdaki çıktıları üretir:

- Etiketli örnek MR görüntüleri ve model özeti.
- Epoch bazında eğitim/doğrulama kaybı ve doğruluk grafikleri.
- Eğitim, doğrulama ve test kümeleri için kayıp ve doğruluk değerleri.
- Sınıflar arasındaki hataları gösteren karışıklık matrisi.
- Her sınıf için precision, recall, F1-score ve support değerlerini içeren sınıflandırma raporu.

README’de sabit bir başarı oranı taahhüt edilmez. Sonuçları kendi çalıştırmanızın çıktıları üzerinden değerlendirin; veri sıralaması, rastgele başlangıçlar, paket sürümleri ve donanım sonuçları etkileyebilir.

## Bilinen sınırlamalar ve geliştirme alanları

Aşağıdaki noktalar mevcut notebook’un davranışını ve gelecekte yapılabilecek iyileştirmeleri açıklar:

- **Ön işleme:** Final projedeki `ImageDataGenerator()` eğitim girdilerine `rescale` veya ResNet50’ye özgü `preprocess_input` uygulamaz. Örnekleri çizdirirken yapılan `/255` işlemi yalnızca görselleştirme içindir. Ön işleme tercihi ayrı bir deney olarak değerlendirilebilir.
- **Veri artırma:** Final projedeki üreteçte döndürme, yakınlaştırma veya benzeri veri artırma ayarları etkin değildir.
- **Callback’ler:** `ReduceLROnPlateau` ve `EarlyStopping` nesneleri tanımlanmıştır; ancak `model.fit` çağrısına `callbacks` verilmediği için mevcut eğitimde çalışmazlar.
- **Tekrarlanabilirlik:** Doğrulama/test bölmesi için `random_state` belirlenmiştir; tüm rastgelelik kaynakları sabitlenmemiştir.
- **Sürüm uyumu:** Laboratuvarlar farklı dönemlere ait Keras/TensorFlow kullanımları ve eski Gym API çağrıları içerir. İlgili notebook için paket sürümü seçimi veya kod uyarlaması gerekebilir. Kurulum hücrelerindeki toplu yükseltmeleri çalıştırmadan önce inceleyin.
- **Model kaydetme:** Final notebook’ta eğitilmiş modeli diske kaydeden veya servis olarak sunan bir adım bulunmaz.

Geliştirme seçenekleri arasında bağımlılık sürümlerinin sabitlenmesi, taşınabilir veri yolları, sınıf oranlarını koruyan veri bölme, callback’lerin eğitime eklenmesi ve model kaydetme adımı yer alır.

## Katkı ve lisans

Hataları bildirmek veya geliştirme önermek için GitHub üzerinde issue açabilirsiniz. Katkı sunarken değişikliğin amacını, etkilenen notebook’u ve nasıl doğrulandığını pull request açıklamasında belirtin. Notebook çıktılarında kişisel dosya yolları veya gereksiz büyük çıktılar bırakmamaya özen gösterin.

Eğitim notebook’larında IBM / Skills Network kaynak ve telif bildirimleri bulunur. Bu bildirimler korunmalıdır. Depoda ayrı bir `LICENSE` dosyası bulunmadığından proje için MIT veya başka bir lisans tanımlandığı varsayılmamalıdır. Veri kümesinin yeniden kullanım koşulları da ilgili kaynağından ayrıca doğrulanmalıdır.

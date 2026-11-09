import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)

RANDOM_STATE = 42          # Sonuçların tekrarlanabilir olması için sabit tohum
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
FIG_DIR = "figs"
import os; os.makedirs(FIG_DIR, exist_ok=True)

def kaydet(ad):
    """Grafiği rapor için PNG olarak kaydeder ve gösterir."""
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{ad}.png", dpi=200, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## A. Veri Seti Seçimi ve İş Probleminin Tanımı
# Veri seti: **Titanic** (OpenML ID 40945; Kaggle "Titanic" yarışmasının tam 1309 yolculu sürümü).
# İş problemi: *Bir yolcunun demografik ve bilet bilgilerinden hayatta kalıp kalmadığını tahmin etmek*
# ve hayatta kalmayı belirleyen faktörleri ortaya koymak (ikili sınıflandırma, hedef: `survived`).

# %%
# ---------------------------------------------------------------
# A.1) Veriyi yükleme
# titanic.csv bu dosyayla aynı klasörde olmalıdır. Dosya yoksa
# veri seti OpenML'den indirilir (internet gerekir).
# ---------------------------------------------------------------
try:
    df_raw = pd.read_csv("titanic.csv")
except FileNotFoundError:
    from sklearn.datasets import fetch_openml
    df_raw = fetch_openml("titanic", version=1, as_frame=True).frame
    df_raw["survived"] = df_raw["survived"].astype(int)

print("Satır sayısı :", df_raw.shape[0])
print("Sütun sayısı :", df_raw.shape[1])
df_raw.head()

# %%
# A.2) Özniteliklerin veri tipleri ve doluluk durumu
df_raw.info()

# %%
# A.3) Sayısal özniteliklerin betimsel istatistikleri
df_raw.describe().T.round(2)

# %%
# A.4) Hedef değişkenin dağılımı (sınıf dengesi)
print(df_raw["survived"].value_counts())
print("\nHayatta kalma oranı: %{:.1f}".format(100 * df_raw["survived"].mean()))

# %% [markdown]
# ## B. Veri Ön İşleme ve Temizleme

# %%
# ---------------------------------------------------------------
# B.1) Eksik veri analizi
# ---------------------------------------------------------------
eksik = pd.DataFrame({
    "Eksik Sayı": df_raw.isna().sum(),
    "Eksik %": (100 * df_raw.isna().mean()).round(1)
}).sort_values("Eksik %", ascending=False)
print(eksik[eksik["Eksik Sayı"] > 0])

plt.figure(figsize=(7, 3.6))
e = eksik[eksik["Eksik Sayı"] > 0]
sns.barplot(x=e["Eksik %"], y=e.index, color="#4C72B0")
for i, v in enumerate(e["Eksik %"]):
    plt.text(v + 1, i, f"%{v}", va="center", fontsize=9)
plt.xlabel("Eksik değer oranı (%)"); plt.ylabel("")
plt.title("Özniteliklere göre eksik değer oranları")
plt.xlim(0, 100)
kaydet("01_eksik_degerler")

# %%
# ---------------------------------------------------------------
# B.2) Öznitelik seçimi ve türetme
# - boat, body : Kaza SONRASI oluşan bilgiler (filika no, ceset no).
#   Hedefi doğrudan ele verdikleri için (veri sızıntısı / leakage) çıkarılır.
# - cabin      : %77 eksik. Değerin kendisi yerine "kabin bilgisi var mı?"
#   (has_cabin) bayrağı türetilir; eksikliğin kendisi bilgi taşır.
# - name       : Doğrudan kullanılamaz; içindeki unvan (Mr, Mrs, Miss, Master...)
#   yaş/cinsiyet/sosyal statü bilgisi taşıdığı için 'title' olarak çıkarılır.
# - sibsp+parch: Aile büyüklüğü (family_size) olarak birleştirilir.
# - ticket, home.dest: Çok yüksek kardinaliteli serbest metin; çıkarılır.
# ---------------------------------------------------------------
df = df_raw.copy()

df["title"] = df["name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
# Nadir unvanları anlamlı gruplara toplama
unvan_map = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Rare", "the Countess": "Rare",
             "Dona": "Rare", "Capt": "Rare", "Col": "Rare", "Don": "Rare", "Dr": "Rare",
             "Major": "Rare", "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare"}
df["title"] = df["title"].replace(unvan_map)
print(df["title"].value_counts())

df["has_cabin"] = df["cabin"].notna().astype(int)
df["family_size"] = df["sibsp"] + df["parch"] + 1

df = df.drop(columns=["boat", "body", "cabin", "name", "ticket", "home.dest"])
df["pclass"] = df["pclass"].astype(int)
df.head()

# %%
# ---------------------------------------------------------------
# B.3) Eğitim / test ayrımı (%80 / %20)
# Ayrım, eksik değer atama ve ölçekleme gibi "öğrenen" işlemlerden ÖNCE
# yapılır. Böylece imputer ve scaler yalnızca eğitim verisinden öğrenir;
# test kümesinin bilgisi modele sızmaz. stratify ile sınıf oranı korunur.
# ---------------------------------------------------------------
X = df.drop(columns="survived")
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
X_train, X_test = X_train.copy(), X_test.copy()
print("Eğitim:", X_train.shape, " Test:", X_test.shape)
print("Hayatta kalma oranı – eğitim: %.3f, test: %.3f" % (y_train.mean(), y_test.mean()))

# %%
# ---------------------------------------------------------------
# B.4) Eksik değerlerin giderilmesi
# - embarked (2 eksik, kategorik) : eğitim kümesinin MOD'u ile doldurulur.
# - fare (1 eksik)                : yolcunun sınıfına (pclass) göre eğitim MEDYAN'ı.
#   Ücret çarpık dağıldığı için ortalama yerine medyan seçilmiştir.
# - age (%20 eksik)               : KNN Imputation (k=5). Yaş; unvan, sınıf,
#   aile büyüklüğü ve ücretle ilişkili olduğu için benzer yolculardan tahmin
#   etmek, tek bir medyan atamaktan daha gerçekçi bir dağılım üretir.
# ---------------------------------------------------------------
emb_mod = X_train["embarked"].mode()[0]
fare_med = X_train.groupby("pclass")["fare"].median()
for d in (X_train, X_test):
    d["embarked"] = d["embarked"].fillna(emb_mod)
    d["fare"] = d["fare"].fillna(d["pclass"].map(fare_med))

# KNN, mesafe tabanlı olduğu için komşuluk öznitelikleri önce ölçeklenir.
# Unvan bilgisi yaşın en güçlü göstergesi olduğundan kukla değişken olarak eklenir.
def knn_girdisi(d):
    t = pd.get_dummies(d["title"], prefix="t").reindex(columns=[f"t_{c}" for c in ["Mr", "Mrs", "Miss", "Master", "Rare"]], fill_value=0)
    return pd.concat([d[["age", "pclass", "sibsp", "parch", "fare"]],
                      (d["sex"] == "male").astype(int).rename("male"), t.astype(int)], axis=1)

knn_tr, knn_te = knn_girdisi(X_train), knn_girdisi(X_test)
knn_scaler = StandardScaler().fit(knn_tr)
imputer = KNNImputer(n_neighbors=5).fit(knn_scaler.transform(knn_tr))
yas_idx = list(knn_tr.columns).index("age")

def yas_doldur(d, k):
    dolu = knn_scaler.inverse_transform(imputer.transform(knn_scaler.transform(k)))
    d["age"] = dolu[:, yas_idx].round(1)

yas_once = X_train["age"].copy()
yas_doldur(X_train, knn_tr); yas_doldur(X_test, knn_te)
print("Kalan eksik değer (eğitim):", int(X_train.isna().sum().sum()), "| (test):", int(X_test.isna().sum().sum()))
print("Yaş medyanı – atama öncesi: %.1f, sonrası: %.1f" % (yas_once.median(), X_train["age"].median()))
print("Unvana göre atanan yaş ortalamaları:")
print(X_train.loc[yas_once.isna(), :].groupby("title")["age"].mean().round(1))

# %%
# KNN atamasının yaş dağılımını bozmadığını kontrol edelim
plt.figure(figsize=(7, 3.4))
sns.kdeplot(yas_once.dropna(), label="Atama öncesi (gözlenen)", fill=True, alpha=.35)
sns.kdeplot(X_train["age"], label="KNN ataması sonrası", fill=True, alpha=.35)
plt.xlabel("Yaş"); plt.ylabel("Yoğunluk"); plt.legend()
plt.title("KNN Imputation öncesi ve sonrası yaş dağılımı (eğitim kümesi)")
kaydet("02_knn_yas")

# %%
# ---------------------------------------------------------------
# B.5) Aykırı değer analizi – IQR ve Z-skoru
# IQR: Q1 - 1.5*IQR altı veya Q3 + 1.5*IQR üstü değerler aykırıdır.
# Z  : |z| > 3 olan değerler aykırıdır.
# Sınırlar yalnızca eğitim kümesinden hesaplanır.
# ---------------------------------------------------------------
sayisal = ["age", "fare", "sibsp", "parch", "family_size"]  # sibsp/parch yalnızca raporlama için
satirlar = []
for c in sayisal:
    s = X_train[c]
    q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    alt, ust = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z = (s - s.mean()) / s.std()
    satirlar.append([c, round(alt, 2), round(ust, 2), int(((s < alt) | (s > ust)).sum()), int((z.abs() > 3).sum()), round(s.skew(), 2)])
aykiri = pd.DataFrame(satirlar, columns=["Öznitelik", "IQR alt", "IQR üst", "IQR aykırı", "Z>3 aykırı", "Çarpıklık"])
print(aykiri.to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
for a, c in zip(ax, ["age", "fare", "family_size"]):
    sns.boxplot(y=X_train[c], ax=a, color="#8DA0CB", flierprops=dict(markerfacecolor="#D55E00", markersize=4))
    a.set_title(c); a.set_ylabel("")
plt.suptitle("Aykırı değer tespiti – kutu grafikleri (eğitim kümesi)", y=1.02)
kaydet("03_aykiri_boxplot")

# %%
# ---------------------------------------------------------------
# B.6) Aykırı değerlerin yönetimi
# Aykırı değerler veri giriş hatası değildir: yüksek ücretler 1. sınıf
# lüks kabinleri, kalabalık aileler gerçek yolcuları temsil eder. Bu
# satırları SİLMEK, hayatta kalma ile güçlü ilişkili grupları (zengin
# yolcular, büyük aileler) veri setinden atmak anlamına gelir. Bu yüzden:
#  - fare : log(1+x) dönüşümü ile çarpıklık azaltılır, ardından log-ücret
#           eğitim kümesi IQR sınırlarında baskılanır (capping/winsorizing).
#  - age, family_size : IQR sınırlarında baskılanır (değer kırpılır, satır kalır).
# ---------------------------------------------------------------
for d in (X_train, X_test):
    d["log_fare"] = np.log1p(d["fare"])

sinirlar = {}
for c in ["log_fare", "age", "family_size"]:
    q1, q3 = X_train[c].quantile([.25, .75]); iqr = q3 - q1
    sinirlar[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
# family_size tam sayıdır: üst sınır yukarı yuvarlanır (3.5 → 4, yani "4 ve üzeri" grubu)
sinirlar["family_size"] = (1, np.ceil(sinirlar["family_size"][1]))
print({k: tuple(round(x, 2) for x in v) for k, v in sinirlar.items()})
for d in (X_train, X_test):
    for c, (alt, ust) in sinirlar.items():
        d[c] = d[c].clip(lower=max(alt, d[c].min() if c != "age" else 0), upper=ust)

print("Ücret çarpıklığı – ham: %.2f, log + baskılama sonrası: %.2f" % (X_train["fare"].skew(), X_train["log_fare"].skew()))
X_train = X_train.drop(columns=["fare"]); X_test = X_test.drop(columns=["fare"])

# %%
# ---------------------------------------------------------------
# B.7) Kategorik değişken dönüşümü
# - sex      : iki kategorili → Label Encoding (male=1, female=0)
# - embarked, title : sırasız (nominal) çok kategorili → One-Hot Encoding.
#   Label Encoding bunlara yapay bir sıra (C<Q<S) yükleyeceği için uygun değildir.
#   drop_first=True ile kukla değişken tuzağı (çoklu doğrusallık) önlenir.
# - pclass   : doğal sıralı (1>2>3) olduğu için sayısal bırakılır.
# Ayrıca sibsp ve parch çıkarılır: family_size = sibsp + parch + 1 olduğundan
# üçü birlikte tam doğrusal bağımlılık (çoklu doğrusallık) yaratır ve
# lojistik regresyon katsayılarını yorumlanamaz hâle getirir.
# ---------------------------------------------------------------
def kodla(d):
    d = d.drop(columns=["sibsp", "parch"])
    d["sex"] = (d["sex"] == "male").astype(int)
    d = pd.get_dummies(d, columns=["embarked", "title"], drop_first=True, dtype=int)
    return d

X_train_enc = kodla(X_train)
X_test_enc = kodla(X_test).reindex(columns=X_train_enc.columns, fill_value=0)
print("Kodlama sonrası öznitelikler:", list(X_train_enc.columns))

# %%
# ---------------------------------------------------------------
# B.8) Ölçeklendirme – StandardScaler (z = (x - μ) / σ)
# Lojistik regresyon katsayıları ve düzenlileştirme (C) ölçeğe duyarlıdır.
# Min-Max yerine Standardization seçildi: baskılama sonrası dağılımlar
# hâlâ tam simetrik değil ve Min-Max uç değerlere daha duyarlıdır.
# Scaler yalnızca eğitim kümesinde fit edilir. Ağaç tabanlı modeller
# ölçekten etkilenmez; tutarlılık için aynı matris kullanılır.
# ---------------------------------------------------------------
scaler = StandardScaler().fit(X_train_enc)
X_train_s = pd.DataFrame(scaler.transform(X_train_enc), columns=X_train_enc.columns, index=X_train_enc.index)
X_test_s = pd.DataFrame(scaler.transform(X_test_enc), columns=X_train_enc.columns, index=X_test_enc.index)
X_train_s.describe().T[["mean", "std"]].round(2).head(8)

# %% [markdown]
# ## C. Keşifsel Veri Analizi (EDA) ve Görselleştirme
# EDA, test kümesini "görülmemiş" tutmak için temizlenmiş **eğitim kümesi** üzerinde yapılır.

# %%
# C.1) Korelasyon matrisi – ısı haritası
eda = X_train_enc.copy(); eda["survived"] = y_train
kor = eda.corr()
plt.figure(figsize=(9, 7.5))
maske = np.triu(np.ones_like(kor, dtype=bool), k=1)
sns.heatmap(kor, mask=maske, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot_kws={"size": 7}, cbar_kws={"shrink": .7})
plt.title("Öznitelikler arası Pearson korelasyon matrisi")
kaydet("04_korelasyon")
print("Hedef ile korelasyonlar:\n", kor["survived"].drop("survived").sort_values().round(3))

# %%
# C.2) Kategorik değişkenlere göre hayatta kalma oranları
eda_raw = X_train.copy(); eda_raw["survived"] = y_train
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for a, c in zip(ax, ["sex", "pclass", "title"]):
    o = eda_raw.groupby(c)["survived"].mean().sort_values(ascending=False)
    sns.barplot(x=o.index.astype(str), y=o.values, ax=a, color="#55A868")
    for i, v in enumerate(o.values): a.text(i, v + .02, f"{v:.2f}", ha="center", fontsize=9)
    a.set_title(f"{c} değişkenine göre"); a.set_xlabel(""); a.set_ylim(0, 1.05)
ax[0].set_ylabel("Hayatta kalma oranı")
kaydet("05_kategorik_oranlar")

# %%
# C.3) Histogram – yaş dağılımı ve hayatta kalma
plt.figure(figsize=(7.5, 3.6))
h = eda_raw.assign(Durum=eda_raw["survived"].map({0: "Öldü (0)", 1: "Kurtuldu (1)"}))
sns.histplot(data=h, x="age", hue="Durum", bins=30, multiple="stack",
             hue_order=["Öldü (0)", "Kurtuldu (1)"], palette={"Öldü (0)": "#C44E52", "Kurtuldu (1)": "#4C72B0"})
plt.xlabel("Yaş"); plt.ylabel("Yolcu sayısı")
plt.title("Yaşa göre hayatta kalma (yığılmış histogram)")
kaydet("06_yas_histogram")

# %%
# C.4) Boxplot – log-ücret ve aile büyüklüğünün hedefe göre dağılımı
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
sns.boxplot(data=eda_raw, x="survived", y="log_fare", ax=ax[0], palette=["#C44E52", "#4C72B0"])
ax[0].set_title("log(1+ücret) – hedefe göre"); ax[0].set_xlabel("Hayatta kaldı (0/1)")
sns.boxplot(data=eda_raw, x="survived", y="age", ax=ax[1], palette=["#C44E52", "#4C72B0"])
ax[1].set_title("Yaş – hedefe göre"); ax[1].set_xlabel("Hayatta kaldı (0/1)")
kaydet("07_boxplot_hedef")

# %%
# C.5) Scatter plot – yaş × log-ücret, renk: hayatta kalma, şekil: cinsiyet
plt.figure(figsize=(7.5, 4.2))
sns.scatterplot(data=eda_raw, x="age", y="log_fare", hue="survived", style="sex", alpha=.7,
                palette={0: "#C44E52", 1: "#4C72B0"}, s=28)
plt.xlabel("Yaş"); plt.ylabel("log(1+ücret)")
plt.title("Yaş ve ücrete göre hayatta kalma")
kaydet("08_scatter")

# %%
# C.6) Sınıf × cinsiyet etkileşimi (tablo)
print(pd.crosstab([eda_raw["pclass"]], eda_raw["sex"], values=eda_raw["survived"], aggfunc="mean").round(3))
print("\nAile büyüklüğüne göre hayatta kalma:")
print(eda_raw.groupby("family_size")["survived"].agg(["mean", "count"]).round(3))

# %% [markdown]
# ## D. Modelleme ve Algoritma Uygulaması
# Üç sınıflandırıcı karşılaştırılır: **Lojistik Regresyon**, **Karar Ağacı**, **Random Forest**.
# Hiperparametreler eğitim kümesinde 5 katlı tabakalı çapraz doğrulama ile (GridSearchCV, ölçüt: F1) seçilir.

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

modeller = {
    "Lojistik Regresyon": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                           {"C": [0.01, 0.1, 0.3, 1, 3, 10]}),
    "Karar Ağacı": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                    {"max_depth": [3, 4, 5, 6, 8, None], "min_samples_leaf": [1, 5, 10, 20],
                     "criterion": ["gini", "entropy"]}),
    "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                      {"n_estimators": [200, 400], "max_depth": [4, 6, 8, None],
                       "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.5]}),
}

en_iyi = {}
for ad, (model, grid) in modeller.items():
    gs = GridSearchCV(model, grid, scoring="f1", cv=cv, n_jobs=-1).fit(X_train_s, y_train)
    en_iyi[ad] = gs.best_estimator_
    print(f"{ad:20s} en iyi parametreler: {gs.best_params_} | CV F1 = {gs.best_score_:.3f}")

# %%
# D.2) Çapraz doğrulama ile kararlılık karşılaştırması (eğitim kümesi, doğruluk)
cv_sonuc = {ad: cross_val_score(m, X_train_s, y_train, cv=cv, scoring="accuracy") for ad, m in en_iyi.items()}
for ad, s in cv_sonuc.items():
    print(f"{ad:20s} CV doğruluk = {s.mean():.3f} ± {s.std():.3f}")

# %% [markdown]
# ## E. Model Değerlendirme ve Sonuç

# %%
# E.1) Test kümesinde performans metrikleri
satir = []
tahmin, olasilik = {}, {}
for ad, m in en_iyi.items():
    p = m.predict(X_test_s); pr = m.predict_proba(X_test_s)[:, 1]
    tahmin[ad], olasilik[ad] = p, pr
    satir.append([ad, accuracy_score(y_test, p), precision_score(y_test, p),
                  recall_score(y_test, p), f1_score(y_test, p), roc_auc_score(y_test, pr)])
sonuc = pd.DataFrame(satir, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]).set_index("Model")
print(sonuc.round(3))
for ad in en_iyi:
    print(f"\n--- {ad} ---")
    print(classification_report(y_test, tahmin[ad], target_names=["Öldü (0)", "Kurtuldu (1)"], digits=3))

# %%
# E.2) Karışıklık matrisleri
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for a, ad in zip(ax, en_iyi):
    cm = confusion_matrix(y_test, tahmin[ad])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                xticklabels=["Öldü", "Kurtuldu"], yticklabels=["Öldü", "Kurtuldu"], annot_kws={"size": 12})
    a.set_title(ad); a.set_xlabel("Tahmin"); a.set_ylabel("Gerçek")
kaydet("09_confusion")

# %%
# E.3) Metriklerin grafikle karşılaştırılması
plt.figure(figsize=(8.5, 3.8))
uzun = sonuc.reset_index().melt(id_vars="Model", var_name="Metrik", value_name="Değer")
g = sns.barplot(data=uzun, x="Metrik", y="Değer", hue="Model", palette="Set2")
for c in g.containers: g.bar_label(c, fmt="%.2f", fontsize=7, padding=1)
plt.ylim(0.5, 1.0); plt.ylabel(""); plt.legend(loc="lower right", fontsize=8)
plt.title("Test kümesinde model performanslarının karşılaştırılması")
kaydet("10_metrik_karsilastirma")

# %%
# E.4) ROC eğrileri
plt.figure(figsize=(5.2, 4.4))
for ad in en_iyi:
    fpr, tpr, _ = roc_curve(y_test, olasilik[ad])
    plt.plot(fpr, tpr, label=f"{ad} (AUC={roc_auc_score(y_test, olasilik[ad]):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("Yanlış pozitif oranı"); plt.ylabel("Doğru pozitif oranı (Recall)")
plt.title("ROC eğrileri"); plt.legend(fontsize=8, loc="lower right")
kaydet("11_roc")

# %%
# E.5) Öznitelik önemi – Random Forest ve Lojistik Regresyon katsayıları
rf = en_iyi["Random Forest"]; lr = en_iyi["Lojistik Regresyon"]
onem = pd.Series(rf.feature_importances_, index=X_train_s.columns).sort_values()
kats = pd.Series(lr.coef_[0], index=X_train_s.columns).sort_values()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
onem.plot.barh(ax=ax[0], color="#4C72B0"); ax[0].set_title("Random Forest – öznitelik önemi")
kats.plot.barh(ax=ax[1], color=np.where(kats > 0, "#4C72B0", "#C44E52")); ax[1].set_title("Lojistik Regresyon – standart katsayılar")
ax[1].axvline(0, color="k", lw=.8)
kaydet("12_oznitelik_onemi")
print(onem.sort_values(ascending=False).round(3))
print("\nOdds oranları (1 std artış için):\n", np.exp(kats).sort_values().round(2))

# %% [markdown]
# ### Sonuç
# Bulguların iş kararları açısından yorumu raporun "Sonuç ve Tartışma" bölümündedir.
# Özetle: cinsiyet, unvan ve yolcu sınıfı hayatta kalmanın en güçlü belirleyicileridir;
# üç model de benzer doğrulukta olup en iyi genel denge Random Forest/Lojistik Regresyon ile elde edilmiştir.

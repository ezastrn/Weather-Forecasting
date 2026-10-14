import site
from pathlib import Path

site.addsitedir(r"C:\Assesment\.venv\Lib\site-packages")
import nbformat

template = nbformat.read('kc-assasement-template.ipynb', as_version=4)
current = nbformat.read('nama-nim-ml.ipynb', as_version=4)

def source(cell):
    return cell.source if isinstance(cell.source, str) else ''.join(cell.source)

def find_code(token):
    for cell in current.cells:
        if cell.cell_type == 'code' and token in source(cell):
            return source(cell)
    raise KeyError(token)

def find_markdown(token):
    for cell in current.cells:
        if cell.cell_type == 'markdown' and token in source(cell):
            return source(cell)
    raise KeyError(token)

setup_code = find_code("TARGET = 'target_temperature_6h_c'")
eda_code = find_code('eda_df = df_clean')
quality_code = find_code('full_duplicate_count')
feature_code = find_code('X_feature = df.drop')
validation_code = find_code('train_end = int(n * 0.70)')
baseline_code = find_code('def make_pipeline')
tuning_code = find_code('search_specs =')
comparison_code = find_code('validation_comparison =')
final_code = find_code('X_train_valid =')
error_code = find_code('test_eval =')

template.cells[0].source = """# Weather Forecasting Challenge

**Name:** Faeyza Safa Izz Deyardi  
**NIM:** 255150207111060

Target forecasting adalah `target_temperature_6h_c`, yaitu temperatur enam jam ke depan. Semua evaluasi menggunakan urutan waktu dan hanya memakai model linear yang diizinkan."""

template.cells[1].source = """This notebook follows the official assessment template. Markdown explanation is placed before each code block, while results, tables, and charts are generated from `weather.csv` when the notebook is run.

### Preserved preliminary benchmark

Eksperimen chronological holdout 80/20 sebelumnya tetap dicatat: Ridge RMSE 0.893393, Linear Regression 0.893480, Lasso 0.893521, dan Elastic Net 0.893534. Benchmark tersebut tidak dihapus, tetapi eksperimen utama di bawah memakai train/validation/test terpisah agar test set tetap independen."""

template.cells[2].source = """## Setup

Memuat library yang diperlukan, membaca data, memparse timestamp, serta menerapkan clipping anomali fisik sebelum EDA dan feature engineering. Model yang diimpor hanya Linear Regression, Ridge, Lasso, dan Elastic Net."""
template.cells[3].source = setup_code

template.cells[4].source = """## Data Overview

Bagian ini mencakup data understanding, EDA, dan data quality: ukuran/nama/dtype/statistik data, rentang dan frekuensi timestamp, missing values, duplicate, gap waktu, pemeriksaan batas fisik, distribusi target, pola waktu, heatmap korelasi, dan hubungan fitur penting dengan target. Outlier tidak otomatis dihapus karena dapat mewakili kondisi cuaca yang valid."""
template.cells[5].source = eda_code + "\n\n# Data-quality checks\n" + quality_code

template.cells[6].source = find_markdown('## 5. Feature Engineering').replace('## 5. Feature Engineering', '## Feature Engineering')
template.cells[7].source = feature_code

template.cells[8].source = find_markdown('## 6. Validation Strategy').replace('## 6. Validation Strategy', '## Validation Strategy')
template.cells[9].source = validation_code

template.cells[10].source = """## Modeling

Linear Regression dipakai sebagai baseline terlatih. Naive Persistence (`T(t+6) = T(t)`) ditambahkan sebagai pembanding non-trained. Ridge, Lasso, dan Elastic Net dituning dengan `TimeSeriesSplit` lima fold pada data training. Imputer median dan `StandardScaler` berada dalam pipeline sehingga di-fit hanya pada training fold. Validation memilih model; test tidak diakses saat tuning."""
template.cells[11].source = baseline_code + "\n\n# Time-series tuning of allowed linear models\n" + tuning_code + "\n\n# Validation comparison, including Naive Persistence\n" + comparison_code

template.cells[12].source = """# Findings

Temuan berikut dihasilkan dari nilai aktual yang dihitung pada cell evidence, bukan dari contoh atau asumsi."""
template.cells[13].source = """## Finding A

### Chart / Bukti

Tabel korelasi dan heatmap pada Data Overview menjadi bukti untuk finding ini."""
template.cells[14].source = """top_corr_feature = corr_target.drop(TARGET).abs().idxmax()
print(f'Feature with the largest absolute correlation to target: {top_corr_feature} (r={corr_target[top_corr_feature]:.3f})')"""
template.cells[15].source = """### Key point

Nilai korelasi aktual dicetak di atas.

### Why

Korelasi memberi bukti hubungan linear univariat, tetapi model tetap dievaluasi memakai seluruh fitur dan bukan korelasi saja."""
template.cells[16].source = """## Finding B

### Chart / Bukti

Plot rata-rata target menurut jam pada Data Overview menjadi bukti pola temporal."""
template.cells[17].source = """peak_hour, trough_hour = hourly_target.idxmax(), hourly_target.idxmin()
print(f'Mean target is highest at {peak_hour:02d}:00 and lowest at {trough_hour:02d}:00.')"""
template.cells[18].source = """### Key point

Jam dengan rata-rata target tertinggi dan terendah dicetak di atas.

### Why

Pola intrahari yang terlihat mendukung fitur hour sin/cos, yang menjaga kedekatan antara jam 23 dan 00."""
template.cells[19].source = """## Finding C

### Chart / Bukti

Tabel model comparison menunjukkan RMSE validation untuk pemilihan model dan RMSE test untuk evaluasi akhir."""
template.cells[20].source = """print(f'Selected by validation: {selected_name} with {selected_params}')
print('Validation comparison (RMSE):')
display(validation_comparison[['model', 'rmse']].sort_values('rmse'))"""
template.cells[21].source = """### Key point

Model dipilih berdasarkan validation, bukan test set.

### Why

Memisahkan selection dan final test menjaga estimasi generalisasi tetap lebih jujur serta mencegah test-set leakage."""

template.cells[22].source = """## Results

Setelah model/hyperparameter dipilih di validation, seluruh model linear yang diizinkan dievaluasi pada test set. Model terpilih di-fit ulang pada gabungan train+validation sebelum test. Plot Actual vs Predicted menggunakan model terpilih berdasarkan validation."""
template.cells[23].source = final_code + """

plt.figure(figsize=(14, 4))
plt.plot(y_test.index[:24 * 21], y_test.iloc[:24 * 21], label='Actual', alpha=0.85)
plt.plot(y_test.index[:24 * 21], test_pred[:24 * 21], label=f'Predicted ({selected_name})', alpha=0.85)
plt.title('Actual vs Predicted - first 21 days of test period')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()
"""

template.cells[24].source = """## Error Analysis

Residual pada test set dianalisis melalui residual-vs-predicted, distribusi residual, MAE menurut jam, dan observasi dengan 5% absolute error tertinggi. Interpretasi dibatasi pada pola yang muncul pada output."""
template.cells[25].source = error_code

template.cells[26].source = """# Discussion & Limitations

- Dataset bersifat sintetis; pola, noise, dan missingness dapat berbeda dari pengukuran stasiun nyata.
- Assessment membatasi model pada regresi linear, sehingga interaksi non-linear cuaca tidak dieksplorasi.
- Lag dan rolling feature membutuhkan observasi historis; deployment perlu mempertahankan penanganan missing value yang kausal.
- Performa test hanya mewakili periode pada dataset ini dan dapat berubah bila distribusi cuaca berubah."""

template.cells[27].source = """# Summary

Notebook selesai dijalankan dengan split kronologis 70% train, 15% validation, dan 15% test. Audit leakage: target tidak masuk feature matrix; lag/rolling memakai masa lalu; imputer/scaler di-fit dalam pipeline training; tuning memakai `TimeSeriesSplit`; test tidak dipakai untuk memilih model; dan tidak ada random split atau model di luar empat estimator linear yang diizinkan.

Lihat bagian **Results** untuk metrik final aktual dan bagian **Error Analysis** untuk batasan empiris model."""

nbformat.write(template, Path('nama-nim-ml.ipynb'))
print('Aligned notebook to official template:', len(template.cells), 'cells')

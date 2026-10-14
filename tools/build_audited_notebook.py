import site
from pathlib import Path

site.addsitedir(r"C:\Assesment\.venv\Lib\site-packages")
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))

md("""
# ML Assessment Lab IS 2026

**Name:** Faeyza Safa Izz Deyardi  
**NIM:** 255150207111060

## 1. Problem Definition

Tujuan eksperimen ini adalah memprediksi `target_temperature_6h_c`, temperatur enam jam ke depan, dari data cuaca multivariat per jam. Karena observasi berurutan dalam waktu, seluruh evaluasi memakai urutan waktu - tidak ada random split.

Model yang digunakan **hanya** `LinearRegression`, `Ridge`, `Lasso`, dan `ElasticNet`. RMSE menjadi metrik utama karena satuannya tetap derajat Celsius dan kesalahan besar diberi penalti lebih tinggi; MAE dan R² hanya digunakan sebagai metrik pendukung.

### Audit dan status benchmark sebelumnya

Notebook sebelumnya sudah memiliki parsing timestamp, EDA dasar, fitur lag/rolling, chronological holdout 80/20, empat model yang diizinkan, tuning `TimeSeriesSplit`, plot residual, findings, dan limitations. Namun, audit menemukan beberapa celah: data understanding/data quality belum lengkap, belum ada split test yang benar-benar terpisah, baseline belum dipisahkan secara eksplisit, serta EDA/error analysis belum mencakup semua bukti yang diminta.

Hasil eksperimen pendahuluan **dipertahankan** di bawah ini dan tidak dihapus. Eksperimen tersebut menggunakan chronological holdout 80/20 tanpa test set terpisah; karenanya tabel ini tidak dipakai untuk klaim generalisasi final.

| Model | RMSE holdout pendahuluan |
|---|---:|
| Ridge | 0.893393 |
| Linear Regression | 0.893480 |
| Lasso | 0.893521 |
| Elastic Net | 0.893534 |

Benchmark pendahuluan terbaik adalah Ridge (`alpha=10`, MAE 0.714367, R² 0.851462). Analisis berikut mempertahankan aturan model, tetapi menambahkan validation dan test period yang terpisah.
""")

md("""
## 2. Data Understanding

Data dibaca dari `weather.csv`. Kelompok fitur mencakup: kondisi termal (`temperature_c`, `dew_point_c`, `feels_like_c`), kelembapan/tekanan (`humidity_pct`, `pressure_hpa`, `soil_moisture_pct`), angin (`wind_speed_ms`, `wind_direction_deg`), awan/radiasi/presipitasi (`cloud_cover_pct`, `solar_radiation_wm2`, `precipitation_mm`, `uv_index`), dan visibilitas (`visibility_km`). Target adalah `target_temperature_6h_c`.
""")
code("""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = 'target_temperature_6h_c'
df_raw = pd.read_csv('weather.csv')
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], format='%Y-%m-%d %H', errors='raise')

# Physical-range cleaning is performed before EDA and feature construction.
df_clean = df_raw.copy()
anomaly_before = {
    'humidity_outside_0_100': int(((df_clean['humidity_pct'] < 0) | (df_clean['humidity_pct'] > 100)).sum()),
    'negative_wind_speed': int((df_clean['wind_speed_ms'] < 0).sum()),
    'wind_direction_outside_0_359': int(((df_clean['wind_direction_deg'] < 0) | (df_clean['wind_direction_deg'] >= 360)).sum()),
}
df_clean['humidity_pct'] = df_clean['humidity_pct'].clip(0, 100)
df_clean['wind_speed_ms'] = df_clean['wind_speed_ms'].clip(lower=0)
df_clean['wind_direction_deg'] = df_clean['wind_direction_deg'] % 360

print(f'Rows x columns: {df_raw.shape[0]:,} x {df_raw.shape[1]}')
print(f'Timestamp range: {df_raw.timestamp.min()} to {df_raw.timestamp.max()}')
print('Columns:', df_raw.columns.tolist())
display(df_raw.head())
display(df_raw.dtypes.to_frame('dtype'))
display(df_raw.describe(include='all').T)
print('Physical anomalies before clipping:', anomaly_before)
print('Cleaning applied: humidity clipped to [0, 100], wind speed clipped at 0, and wind direction mapped with modulo 360.')
""")

md("""
## 3. Exploratory Data Analysis

EDA berikut menampilkan distribusi target, urutan suhu/target, korelasi numerik, hubungan dua fitur dengan korelasi absolut tertinggi, dan pola temporal. Interpretasi dicetak langsung dari nilai yang dihitung agar tidak mengklaim pola yang tidak ada pada data.
""")
code("""
eda_df = df_clean.sort_values('timestamp').set_index('timestamp')
numeric_cols = eda_df.select_dtypes(include='number').columns.tolist()
corr_target = eda_df[numeric_cols].corr()[TARGET].sort_values()
top_features = corr_target.drop(TARGET).abs().sort_values(ascending=False).head(2).index.tolist()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
eda_df[TARGET].hist(bins=45, ax=axes[0, 0], color='tab:red', edgecolor='white')
axes[0, 0].set(title='Distribution of target temperature', xlabel='Target temperature (°C)')

eda_df[["temperature_c", TARGET]].iloc[:24 * 30].plot(ax=axes[0, 1], alpha=0.8)
axes[0, 1].set(title='Observed and 6-hour-ahead temperature (first 30 days)', ylabel='°C')

for feature, color in zip(top_features, ['tab:blue', 'tab:green']):
    sample = eda_df[[feature, TARGET]].dropna().sample(min(3000, eda_df[[feature, TARGET]].dropna().shape[0]), random_state=42)
    axes[1, 0].scatter(sample[feature], sample[TARGET], alpha=0.2, s=10, label=feature, color=color)
axes[1, 0].set(title='Two strongest univariate relationships', xlabel='Feature value', ylabel='Target (°C)')
axes[1, 0].legend()

hourly_target = eda_df.groupby(eda_df.index.hour)[TARGET].mean()
hourly_target.plot(ax=axes[1, 1], marker='o', color='tab:purple')
axes[1, 1].set(title='Mean target temperature by hour', xlabel='Hour', ylabel='Mean target (°C)')
plt.tight_layout()
plt.show()

display(corr_target.to_frame('correlation_with_target'))
plt.figure(figsize=(10, 7))
sns.heatmap(eda_df[numeric_cols].corr(), cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Correlation among numeric variables')
plt.show()

peak_hour, trough_hour = hourly_target.idxmax(), hourly_target.idxmin()
print(f'Interpretation: {top_features[0]} has the largest absolute Pearson correlation with the target (r={corr_target[top_features[0]]:.3f}).')
print(f'The hourly mean target is highest at {peak_hour:02d}:00 and lowest at {trough_hour:02d}:00; this supports cyclic time features, not a claim of causality.')
""")

md("""
## 4. Data Quality

Pemeriksaan dilakukan sebelum modeling untuk duplicate, urutan dan duplikasi timestamp, frekuensi/interval/gap, missing value, serta batas fisik sederhana. Observasi tidak dihapus hanya karena terdeteksi outlier IQR karena cuaca ekstrem dapat menjadi sinyal. Data diurutkan; bila timestamp duplikat ditemukan, baris terakhir akan dipertahankan agar satu timestamp hanya memiliki satu observasi.
""")
code("""
full_duplicate_count = int(df_clean.duplicated().sum())
timestamp_duplicate_count = int(df_clean['timestamp'].duplicated().sum())
is_sorted_before = df_clean['timestamp'].is_monotonic_increasing
time_sorted = df_clean.sort_values('timestamp').drop_duplicates('timestamp', keep='last').copy()
intervals = time_sorted['timestamp'].diff().dropna()
inferred_frequency = pd.infer_freq(time_sorted['timestamp'])
gap_count = int((intervals != pd.Timedelta(hours=1)).sum())

missing_table = time_sorted.isna().sum().sort_values(ascending=False).to_frame('missing_count')
missing_table['missing_pct'] = 100 * missing_table['missing_count'] / len(time_sorted)
display(missing_table)

plausibility_rules = {
    'humidity_pct': lambda s: (s < 0) | (s > 100),
    'cloud_cover_pct': lambda s: (s < 0) | (s > 100),
    'precipitation_mm': lambda s: s < 0,
    'solar_radiation_wm2': lambda s: s < 0,
    'wind_speed_ms': lambda s: s < 0,
    'visibility_km': lambda s: s < 0,
    'uv_index': lambda s: s < 0,
}
plausibility = {col: int(rule(time_sorted[col].dropna()).sum()) for col, rule in plausibility_rules.items()}

print(f'Exact duplicate rows: {full_duplicate_count}')
print(f'Duplicate timestamps: {timestamp_duplicate_count}')
print(f'Input sorted before handling: {is_sorted_before}; sorted after handling: {time_sorted.timestamp.is_monotonic_increasing}')
print(f'Inferred frequency: {inferred_frequency}; non-hourly intervals/gaps: {gap_count}')
print('Out-of-range values under simple physical checks:', plausibility)
print('Safe treatment: chronological forward-fill will only use earlier observations; remaining missing values are median-imputed inside each training pipeline.')
""")

md("""
## 5. Feature Engineering

`hour`, `day_of_week`, dan `month` represent kalender. `hour_sin`/`hour_cos` menjaga kedekatan jam 23 dan 0. Arah angin juga diubah ke sin/cos karena bersifat melingkar. Lag suhu 1, 2, 3, 6, 12, dan 24 jam serta rolling mean/std 3, 6, 12, dan 24 jam menangkap persistensi suhu jangka pendek hingga harian.

Rolling feature memakai `shift(1)` sebelum `rolling`, sehingga nilai pada waktu t hanya memakai suhu sebelum t; target tidak pernah dijadikan feature. Forward-fill bersifat kausal (hanya nilai masa lalu), sedangkan median imputer dan scaler nanti di-fit di pipeline training saja.
""")
code("""
df = time_sorted.set_index('timestamp')
X_feature = df.drop(columns=TARGET).copy()  # target explicitly excluded from X

X_feature['hour'] = X_feature.index.hour
X_feature['day_of_week'] = X_feature.index.dayofweek
X_feature['month'] = X_feature.index.month
X_feature['is_weekend'] = (X_feature.index.dayofweek >= 5).astype(int)
X_feature['hour_sin'] = np.sin(2 * np.pi * X_feature['hour'] / 24)
X_feature['hour_cos'] = np.cos(2 * np.pi * X_feature['hour'] / 24)
X_feature['wind_dir_sin'] = np.sin(2 * np.pi * X_feature['wind_direction_deg'] / 360)
X_feature['wind_dir_cos'] = np.cos(2 * np.pi * X_feature['wind_direction_deg'] / 360)

for lag in [1, 2, 3, 6, 12, 24]:
    X_feature[f'temperature_lag_{lag}h'] = X_feature['temperature_c'].shift(lag)
for window in [3, 6, 12, 24]:
    past_temperature = X_feature['temperature_c'].shift(1).rolling(window=window, min_periods=window)
    X_feature[f'temperature_roll_mean_{window}h'] = past_temperature.mean()
    X_feature[f'temperature_roll_std_{window}h'] = past_temperature.std()

# Causal missing-value propagation; no backward fill and no interpolation from future timestamps.
X_feature = X_feature.ffill()
y = df[TARGET].copy()
assert TARGET not in X_feature.columns
assert X_feature.index.equals(y.index)

print(f'Engineered feature count: {X_feature.shape[1]}')
print(f'Rows still containing feature NaN (handled later by train-fitted median imputer): {X_feature.isna().any(axis=1).sum()}')
display(X_feature.head())
""")

md("""
## 6. Validation Strategy

Data dibagi secara kronologis menjadi 70% training, 15% validation, dan 15% test. Proporsi ini memberikan periode training yang panjang dan dua periode masa depan yang cukup untuk seleksi dan evaluasi. Validation dipakai untuk membandingkan model; test disimpan dan hanya dibuka sekali untuk evaluasi final setelah model/hyperparameter dipilih. `TimeSeriesSplit` lima fold digunakan hanya di training untuk tuning.
""")
code("""
n = len(X_feature)
train_end = int(n * 0.70)
valid_end = int(n * 0.85)

X_train, y_train = X_feature.iloc[:train_end].copy(), y.iloc[:train_end].copy()
X_valid, y_valid = X_feature.iloc[train_end:valid_end].copy(), y.iloc[train_end:valid_end].copy()
X_test, y_test = X_feature.iloc[valid_end:].copy(), y.iloc[valid_end:].copy()

assert X_train.index.max() < X_valid.index.min() < X_test.index.min()
for name, x_part in [('Train', X_train), ('Validation', X_valid), ('Test', X_test)]:
    print(f'{name:10s}: {len(x_part):,} rows | {x_part.index.min()} to {x_part.index.max()}')

tscv = TimeSeriesSplit(n_splits=5)
for fold, (tr, va) in enumerate(tscv.split(X_train), start=1):
    assert X_train.index[tr].max() < X_train.index[va].min()
    print(f'CV fold {fold}: train ends {X_train.index[tr].max()} | validation starts {X_train.index[va].min()}')
""")

md("""
## 7. Baseline Model

Linear Regression adalah baseline yang transparan: performa model regularisasi harus dibandingkan dengannya, bukan hanya dibandingkan satu sama lain. Imputer dan scaler berada dalam pipeline; keduanya baru di-fit ketika `fit(X_train, y_train)` dipanggil.
""")
code("""
def make_pipeline(estimator):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', estimator),
    ])

baseline = make_pipeline(LinearRegression())
baseline.fit(X_train, y_train)
baseline_valid_pred = baseline.predict(X_valid)
naive_valid_pred = X_valid['temperature_c'].to_numpy()  # persistence: predict T(t+6) as T(t)
baseline_metrics = {
    'model': 'Linear Regression',
    'rmse': np.sqrt(mean_squared_error(y_valid, baseline_valid_pred)),
    'mae': mean_absolute_error(y_valid, baseline_valid_pred),
    'r2': r2_score(y_valid, baseline_valid_pred),
    'best_params': {},
}
naive_validation_metrics = {
    'model': 'Naive Persistence',
    'rmse': np.sqrt(mean_squared_error(y_valid, naive_valid_pred)),
    'mae': mean_absolute_error(y_valid, naive_valid_pred),
    'r2': r2_score(y_valid, naive_valid_pred),
    'best_params': 'not applicable',
}
print('Baseline validation metrics:', baseline_metrics)
print('Naive Persistence validation metrics:', naive_validation_metrics)
""")

md("""
## 8. Model Development

Tiga model regularisasi menggunakan pipeline preprocessing yang sama dengan baseline. Tidak ada estimator lain yang dibuat atau dilatih.

## 9. Hyperparameter Tuning

`GridSearchCV` menggunakan lima chronological fold dari training set dan RMSE negatif sebagai score. Alpha Ridge/Lasso serta alpha dan `l1_ratio` Elastic Net dituning tanpa mengakses validation atau test. Setelah tuning, setiap kandidat dibandingkan pada validation period yang sama.
""")
code("""
search_specs = {
    'Ridge': (Ridge(), {'model__alpha': [0.01, 0.1, 1, 10, 100]}),
    'Lasso': (Lasso(max_iter=20000), {'model__alpha': [0.001, 0.01, 0.1, 1]}),
    'Elastic Net': (ElasticNet(max_iter=20000), {'model__alpha': [0.001, 0.01, 0.1, 1], 'model__l1_ratio': [0.1, 0.5, 0.9]}),
}

fitted_validation_models = {'Linear Regression': baseline}
comparison_rows = [baseline_metrics]
for name, (estimator, grid) in search_specs.items():
    search = GridSearchCV(
        make_pipeline(estimator), grid, cv=tscv,
        scoring='neg_root_mean_squared_error', n_jobs=1,
    )
    search.fit(X_train, y_train)
    pred = search.predict(X_valid)
    fitted_validation_models[name] = search
    comparison_rows.append({
        'model': name,
        'rmse': np.sqrt(mean_squared_error(y_valid, pred)),
        'mae': mean_absolute_error(y_valid, pred),
        'r2': r2_score(y_valid, pred),
        'best_params': search.best_params_,
        'cv_rmse': -search.best_score_,
    })

comparison = pd.DataFrame(comparison_rows).sort_values('rmse').reset_index(drop=True)
display(comparison)
selected_name = comparison.loc[0, 'model']
selected_params = comparison.loc[0, 'best_params']
print(f'Selected using validation only: {selected_name} with {selected_params}')
""")

md("""
## 10. Model Comparison

Tabel di atas adalah perbandingan pada validation period. RMSE terendah menentukan kandidat final; MAE dan R² memberi konteks tambahan. Test set belum digunakan pada langkah ini.
""")
code("""
validation_comparison = pd.concat([pd.DataFrame([naive_validation_metrics]), comparison], ignore_index=True)
display(validation_comparison[['model', 'rmse', 'mae', 'r2', 'best_params']].style.format({'rmse': '{:.6f}', 'mae': '{:.6f}', 'r2': '{:.6f}'}))
print('Naive Persistence is a benchmark only; selection among permitted trained models uses the four linear models above.')
""")

md("""
## 11. Final Evaluation

Setelah model dan hyperparameter dipilih pada validation, model final di-fit ulang pada gabungan train+validation untuk memanfaatkan seluruh data sebelum test. Imputer dan scaler juga di-fit ulang hanya pada gabungan tersebut. Test period tetap tidak dipakai untuk tuning maupun seleksi.
""")
code("""
X_train_valid = pd.concat([X_train, X_valid])
y_train_valid = pd.concat([y_train, y_valid])

if selected_name == 'Linear Regression':
    final_estimator = LinearRegression()
elif selected_name == 'Ridge':
    final_estimator = Ridge(alpha=selected_params['model__alpha'])
elif selected_name == 'Lasso':
    final_estimator = Lasso(alpha=selected_params['model__alpha'], max_iter=20000)
else:
    final_estimator = ElasticNet(
        alpha=selected_params['model__alpha'],
        l1_ratio=selected_params['model__l1_ratio'],
        max_iter=20000,
    )

final_model = make_pipeline(final_estimator)
final_model.fit(X_train_valid, y_train_valid)
test_pred = final_model.predict(X_test)
final_metrics = {
    'RMSE (°C)': np.sqrt(mean_squared_error(y_test, test_pred)),
    'MAE (°C)': mean_absolute_error(y_test, test_pred),
    'R²': r2_score(y_test, test_pred),
}
display(pd.DataFrame([final_metrics], index=[f'Final test - {selected_name}']).T)
print(f"Interpretation: RMSE {final_metrics['RMSE (°C)']:.3f} °C is the typical error scale with extra penalty for larger temperature misses; it does not mean the model is perfectly accurate.")

# Evaluate the naive baseline and every permitted linear model on the untouched test period.
def estimator_from_row(row):
    if row['model'] == 'Linear Regression':
        return LinearRegression()
    if row['model'] == 'Ridge':
        return Ridge(alpha=row['best_params']['model__alpha'])
    if row['model'] == 'Lasso':
        return Lasso(alpha=row['best_params']['model__alpha'], max_iter=20000)
    return ElasticNet(alpha=row['best_params']['model__alpha'], l1_ratio=row['best_params']['model__l1_ratio'], max_iter=20000)

test_rows = [{
    'model': 'Naive Persistence',
    'rmse': np.sqrt(mean_squared_error(y_test, X_test['temperature_c'])),
    'mae': mean_absolute_error(y_test, X_test['temperature_c']),
    'r2': r2_score(y_test, X_test['temperature_c']),
}]
for _, row in comparison.iterrows():
    candidate = make_pipeline(estimator_from_row(row))
    candidate.fit(X_train_valid, y_train_valid)
    candidate_pred = candidate.predict(X_test)
    test_rows.append({
        'model': row['model'],
        'rmse': np.sqrt(mean_squared_error(y_test, candidate_pred)),
        'mae': mean_absolute_error(y_test, candidate_pred),
        'r2': r2_score(y_test, candidate_pred),
    })
test_comparison = pd.DataFrame(test_rows).sort_values('rmse').reset_index(drop=True)
display(test_comparison.style.format({'rmse': '{:.6f}', 'mae': '{:.6f}', 'r2': '{:.6f}'}))
""")

md("""
## 12. Error Analysis

Error analysis menggunakan prediksi final pada test set saja. Grafik menilai kecocokan actual-vs-predicted, pola residual, distribusi residual, dan MAE menurut jam. Observasi error besar didefinisikan sebagai 5% error absolut tertinggi pada test set, sehingga threshold berasal dari data aktual.
""")
code("""
test_eval = pd.DataFrame({'actual': y_test, 'predicted': test_pred}, index=y_test.index)
test_eval['residual'] = test_eval['actual'] - test_eval['predicted']
test_eval['abs_error'] = test_eval['residual'].abs()
test_eval['hour'] = test_eval.index.hour
test_eval = test_eval.join(df[['humidity_pct', 'precipitation_mm', 'wind_speed_ms', 'temperature_c']], how='left')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
test_eval[['actual', 'predicted']].iloc[:24 * 21].plot(ax=axes[0, 0], alpha=0.85)
axes[0, 0].set(title='Actual vs predicted (first 21 test days)', ylabel='Temperature (°C)')
axes[0, 1].scatter(test_eval['predicted'], test_eval['residual'], alpha=0.3, s=10)
axes[0, 1].axhline(0, color='black', linestyle='--')
axes[0, 1].set(title='Residual vs predicted', xlabel='Predicted (°C)', ylabel='Actual - predicted (°C)')
axes[1, 0].hist(test_eval['residual'], bins=40, color='tab:orange', edgecolor='white')
axes[1, 0].axvline(0, color='black', linestyle='--')
axes[1, 0].set(title='Residual distribution', xlabel='Residual (°C)')
hourly_mae = test_eval.groupby('hour')['abs_error'].agg(['mean', 'median', 'count'])
hourly_mae['mean'].plot(kind='bar', ax=axes[1, 1], color='tab:purple')
axes[1, 1].set(title='Mean absolute error by hour', xlabel='Hour', ylabel='MAE (°C)')
plt.tight_layout()
plt.show()

large_error_threshold = test_eval['abs_error'].quantile(0.95)
large_errors = test_eval[test_eval['abs_error'] >= large_error_threshold].sort_values('abs_error', ascending=False)
display(large_errors.head(15))
print(f'Largest mean hourly MAE: {hourly_mae["mean"].idxmax():02d}:00, {hourly_mae["mean"].max():.3f} °C.')
print(f'Large-error threshold (test P95): {large_error_threshold:.3f} °C; observations at/above threshold: {len(large_errors)}.')
""")

md("""
## 13. Findings

Setiap finding berikut mengambil angka dari output eksperimen yang baru dijalankan, bukan dari contoh atau asumsi.
""")
code("""
top_corr_feature = corr_target.drop(TARGET).abs().idxmax()
highest_error_hour = hourly_mae['mean'].idxmax()

print('Finding 1')
print('Chart/Bukti: correlation table dan heatmap pada bagian EDA.')
print(f'Key Point: {top_corr_feature} memiliki korelasi absolut terbesar dengan target (r={corr_target[top_corr_feature]:.3f}).')
print('Why: hubungan linear univariat ini memberi alasan untuk mempertahankan fitur tersebut, tetapi performa akhir tetap dinilai dari semua fitur bersama-sama.')

print('\\nFinding 2')
print('Chart/Bukti: line chart rata-rata target menurut jam.')
print(f'Key Point: rata-rata target tertinggi terjadi pada {peak_hour:02d}:00 dan terendah pada {trough_hour:02d}:00.')
print('Why: pola intrahari yang terlihat mendukung encoding jam siklikal agar model linear dapat merepresentasikan pola berulang.')

print('\\nFinding 3')
print('Chart/Bukti: tabel model comparison pada validation period.')
print(f'Key Point: {selected_name} dipilih sebelum test dengan validation RMSE {comparison.loc[0, "rmse"]:.3f} °C dan parameter {selected_params}.')
print('Why: selection berbasis validation menjaga test set sebagai evaluasi akhir yang independen.')

print('\\nFinding 4')
print('Chart/Bukti: bar chart MAE menurut jam dan tabel 5% error terbesar di test set.')
print(f'Key Point: MAE rata-rata terbesar berada pada {highest_error_hour:02d}:00, sebesar {hourly_mae.loc[highest_error_hour, "mean"]:.3f} °C.')
print('Why: ini menunjukkan waktu yang perlu dipantau; grafik tidak dengan sendirinya membuktikan penyebab fisik error.')
""")

md("""
## 14. Limitations

- Dataset dinyatakan sintetis, sehingga pola, noise, dan missingness dapat berbeda dari operasi stasiun cuaca nyata.
- Hanya model linear yang digunakan sesuai aturan assessment; hubungan non-linear atau interaksi cuaca yang kompleks mungkin belum tertangkap.
- Fitur lag/rolling mengandalkan ketersediaan observasi historis. Pada deployment, keterlambatan sensor perlu ditangani dengan kebijakan missing value yang sama-sama kausal.
- Hasil test menggambarkan periode yang tersedia pada data ini; performa dapat berubah pada pola cuaca atau distribusi input yang belum muncul di dataset.
""")

md("""
## 15. Conclusion

Eksperimen memisahkan pemilihan model (training/CV/validation) dari evaluasi final (test). Cell berikut merangkum hasil aktual. Audit leakage di bawah mendokumentasikan kontrol yang digunakan.
""")
code("""
print(f'Final model: {selected_name} with {selected_params}')
print(f"Final test RMSE: {final_metrics['RMSE (°C)']:.6f} °C")
print(f"Final test MAE:  {final_metrics['MAE (°C)']:.6f} °C")
print(f"Final test R²:   {final_metrics['R²']:.6f}")

print('\\nLeakage audit')
print('- Target excluded from X:', TARGET not in X_feature.columns)
print('- Lag and rolling use past only: shift(1) is applied before every rolling feature.')
print('- Split is chronological:', X_train.index.max() < X_valid.index.min() < X_test.index.min())
print('- Imputer/scaler fit in training pipelines only; final pipeline fits only train+validation before test.')
print('- Hyperparameters selected from CV/training and validation; test is not used for selection.')
print('- No random split and no model outside the four permitted linear estimators are used.')
""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
}
nbf.write(nb, Path('nama-nim-ml.ipynb'))
print('Rebuilt audited notebook:', len(cells), 'cells')

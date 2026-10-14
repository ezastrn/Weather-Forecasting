# Weather Forecasting

A time-series forecasting project that predicts the temperature six hours ahead (`target_temperature_6h_c`) from historical weather observations. The analysis is designed around chronological evaluation to avoid data leakage and uses only the linear models permitted by the project brief.

## Highlights

- Exploratory analysis of timestamp coverage, data quality, physical bounds, missing values, duplicates, and temporal patterns.
- Causal feature engineering with time-based variables, lag features, and rolling statistics.
- Chronological 70/15/15 train-validation-test split and five-fold `TimeSeriesSplit` for tuning.
- Baseline comparison against Naive Persistence, Linear Regression, Ridge, Lasso, and Elastic Net.
- Pipeline-based preprocessing so imputation and scaling are fitted only on training data.
- Final error analysis with residual diagnostics and actual-versus-predicted visualizations.

## Repository layout

```text
.
├── data/
│   └── weather.csv                     # Source dataset
├── notebooks/
│   ├── weather_forecasting.ipynb       # Main analysis and forecasting workflow
│   └── assessment_template.ipynb       # Original assessment template
├── tools/                              # Utilities used while preparing the notebooks
├── requirements.txt                    # Python dependencies
└── README.md
```

## Getting started

1. Clone the repository and create a Python environment.

   ```bash
   git clone https://github.com/ezastrn/Weather-Forecasting.git
   cd Weather-Forecasting
   python -m venv .venv
   ```

2. Activate the environment and install dependencies.

   ```bash
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1

   pip install -r requirements.txt
   ```

3. Start Jupyter and open the main notebook.

   ```bash
   jupyter notebook notebooks/weather_forecasting.ipynb
   ```

The notebook resolves the tracked dataset at `data/weather.csv` when run from the repository root. In Google Colab, upload `weather.csv` when prompted or update the optional Google Drive path in the setup cell.

## Methodology

The project preserves temporal order throughout preparation and evaluation. Features are constructed only from information available before the forecast horizon. Model selection is performed on the validation period; the test period remains untouched until the final evaluation. This makes the reported result a more realistic measure of expected generalization.

## Tech stack

Python, Jupyter, pandas, NumPy, scikit-learn, Matplotlib, and Seaborn.

## Contributor

Created and maintained by [Faeyza Safa Izz Deyardi](https://github.com/ezastrn).

## License

This project is provided for educational and portfolio purposes.

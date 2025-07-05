import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from arch import arch_model

# Step 1: Download data
hul = yf.download("HINDUNILVR.NS", start="2024-01-01", end="2024-07-01")
nifty = yf.download("^NSEI", start="2024-01-01", end="2024-07-01")

hul.columns = [col[0] for col in hul.columns]
nifty.columns = [col[0] for col in nifty.columns]

hul['Return'] = hul['Close'].pct_change()
nifty['Market_Return'] = nifty['Close'].pct_change()

# Merge returns
data = hul[['Close', 'Return']].join(nifty['Market_Return'], how='inner').dropna()

# Step 2: CAPM Regression
X = sm.add_constant(data['Market_Return'])
y = data['Return']
capm_model = sm.OLS(y, X).fit()
data['CAPM_Pred'] = capm_model.predict(X)

# Step 3: Plot CAPM line
plt.figure(figsize=(8, 6))
sns.regplot(x='Market_Return', y='Return', data=data, line_kws={"color": "red"})
plt.title("CAPM Regression: HUL vs Nifty")
plt.xlabel("Market Return (Nifty)")
plt.ylabel("HUL Return")
plt.grid(True)
plt.tight_layout()
plt.savefig("capm_regression_plot.png")
plt.close()

# Step 4: Feature Engineering for ML
data['Lag1'] = data['Return'].shift(1)
data['Lag2'] = data['Return'].shift(2)
data['MA5'] = data['Return'].rolling(5).mean()
data['Vol5'] = data['Return'].rolling(5).std()
data['Target'] = (data['Return'].shift(-1) > 0).astype(int)
ml_data = data.dropna()

features = ['Market_Return', 'Lag1', 'Lag2', 'MA5', 'Vol5']
X_ml = ml_data[features]
y_ml = ml_data['Target']

X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.2, shuffle=False)

# Step 5: Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Step 6: Feature Importance Plot
importances = model.feature_importances_
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance_plot.png")
plt.close()

# Step 7: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_plot.png")
plt.close()

# Step 8: Value at Risk (Historical)
var_95 = np.percentile(data['Return'], 5)

# Step 9: VaR Distribution Plot
plt.figure(figsize=(8, 4))
sns.histplot(data['Return'], bins=50, kde=True)
plt.axvline(var_95, color='red', linestyle='--', label=f'95% VaR = {var_95:.2%}')
plt.title("HUL Daily Returns & 95% VaR")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("var_histogram_plot.png")
plt.close()

# Step 10: GARCH(1,1) Volatility Forecast
garch_model = arch_model(data['Return'] * 100, vol='Garch', p=1, q=1)
garch_result = garch_model.fit(disp='off')
forecast = garch_result.forecast(horizon=5)
vol_forecast = forecast.variance[-1:].values.flatten()**0.5

# Output Summary
capm_summary = capm_model.summary().as_text()
classification = classification_report(y_test, y_pred, output_dict=True)
roc_auc, var_95, vol_forecast[0], capm_summary[:500]  # Short preview of regression summary
# Summary Output 

print("CAPM Summary (first few lines):")
print(capm_summary[:500])  # Short preview

print(" Random Forest Classification Report:")
for label, metrics in classification.items():
    print(f"{label}: {metrics}")

print(f" 1-day 95% VaR: {var_95:.2%}")
print(f" GARCH(1,1) 5-day Volatility Forecast: {vol_forecast[0]:.2f}%")

print("Plots saved to current directory:")
print(" - capm_regression_plot.png")
print(" - feature_importance_plot.png")
print(" - roc_curve_plot.png")
print(" - var_histogram_plot.png")
print(capm_summary[:500])
print(classification_report(y_test, y_pred))
print(f"1-day 95% VaR: {var_95:.2%}")
print(f"GARCH(1,1) Volatility Forecast (5-day): {vol_forecast[0]:.2f}%")


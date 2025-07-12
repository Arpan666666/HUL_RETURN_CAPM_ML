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

# ---------------------- HUL ANALYSIS ----------------------
print("---------- HUL Analysis ----------")
hul = yf.download("HINDUNILVR.NS", start="2024-01-01", end="2024-07-01")
nifty = yf.download("^NSEI", start="2024-01-01", end="2024-07-01")

hul['Return'] = hul['Close'].pct_change()
nifty['Market_Return'] = nifty['Close'].pct_change()

data = pd.DataFrame({
    'Close': hul['Close'].squeeze(),
    'Return': hul['Return'].squeeze(),
    'Market_Return': nifty['Market_Return'].squeeze()
}).dropna()


# CAPM Regression
X = sm.add_constant(data['Market_Return'])
y = data['Return']
capm_model = sm.OLS(y, X).fit()
data['CAPM_Pred'] = capm_model.predict(X)

plt.figure(figsize=(8, 6))
sns.regplot(x='Market_Return', y='Return', data=data, line_kws={"color": "red"})
plt.title("CAPM Regression: HUL vs Nifty")
plt.xlabel("Market Return (Nifty)")
plt.ylabel("HUL Return")
plt.tight_layout()
plt.savefig("hul_capm_plot.png")
plt.close()

# ML Features
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

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

importances = model.feature_importances_
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features)
plt.title("HUL - Feature Importance")
plt.tight_layout()
plt.savefig("hul_feature_importance.png")
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("HUL - ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.tight_layout()
plt.savefig("hul_roc_curve.png")
plt.close()

var_95 = np.percentile(data['Return'], 5)
plt.figure(figsize=(8, 4))
sns.histplot(data['Return'], bins=50, kde=True)
plt.axvline(var_95, color='red', linestyle='--', label=f'95% VaR = {var_95:.2%}')
plt.title("HUL Daily Returns & 95% VaR")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("hul_var_hist.png")
plt.close()

garch_model = arch_model(data['Return'] * 100, vol='Garch', p=1, q=1)
garch_result = garch_model.fit(disp='off')
forecast = garch_result.forecast(horizon=5)
vol_forecast = forecast.variance[-1:].values.flatten() ** 0.5

print(capm_model.summary())
print("Random Forest Report:\n", classification_report(y_test, y_pred))
print(f"1-day 95% VaR: {var_95:.2%}")
print(f"GARCH(1,1) 5-day Volatility Forecast: {vol_forecast[0]:.2f}%")

# ---------------------- NESTLE ANALYSIS ----------------------
print("\n---------- Nestlé Analysis ----------")
nestle = yf.download("NESTLEIND.NS", start="2024-01-01", end="2024-07-01")
nifty = yf.download("^NSEI", start="2024-01-01", end="2024-07-01")

nestle['Return'] = nestle['Close'].pct_change()
nifty['Market_Return'] = nifty['Close'].pct_change()

data = pd.concat([
    nestle['Close'], 
    nestle['Return'], 
    nifty['Market_Return']
], axis=1)
data.columns = ['Close', 'Return', 'Market_Return']
data = data.dropna()


X = sm.add_constant(data['Market_Return'])
y = data['Return']
capm_model = sm.OLS(y, X).fit()
data['CAPM_Pred'] = capm_model.predict(X)

plt.figure(figsize=(8, 6))
sns.regplot(x='Market_Return', y='Return', data=data, line_kws={"color": "red"})
plt.title("CAPM Regression: Nestlé vs Nifty")
plt.xlabel("Market Return (Nifty)")
plt.ylabel("Nestlé Return")
plt.tight_layout()
plt.savefig("nestle_capm_plot.png")
plt.close()

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

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

importances = model.feature_importances_
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features)
plt.title("Nestlé - Feature Importance")
plt.tight_layout()
plt.savefig("nestle_feature_importance.png")
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Nestlé - ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.tight_layout()
plt.savefig("nestle_roc_curve.png")
plt.close()

var_95 = np.percentile(data['Return'], 5)
plt.figure(figsize=(8, 4))
sns.histplot(data['Return'], bins=50, kde=True)
plt.axvline(var_95, color='red', linestyle='--', label=f'95% VaR = {var_95:.2%}')
plt.title("Nestlé Daily Returns & 95% VaR")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("nestle_var_hist.png")
plt.close()

garch_model = arch_model(data['Return'] * 100, vol='Garch', p=1, q=1)
garch_result = garch_model.fit(disp='off')
forecast = garch_result.forecast(horizon=5)
vol_forecast = forecast.variance[-1:].values.flatten() ** 0.5

print(capm_model.summary())
print("Random Forest Report:\n", classification_report(y_test, y_pred))
print(f"1-day 95% VaR: {var_95:.2%}")
print(f"GARCH(1,1) 5-day Volatility Forecast: {vol_forecast[0]:.2f}%")
# GARCH(1,1) Volatility Forecast - HUL
hul_garch_model = arch_model(data['Return'] * 100, vol='Garch', p=1, q=1)
hul_garch_result = hul_garch_model.fit(disp='off')
hul_volatility = hul_garch_result.conditional_volatility

# Plotting GARCH volatility for HUL
plt.figure(figsize=(10, 4))
plt.plot(data.index[-len(hul_volatility):], hul_volatility, color='darkblue')
plt.title("HUL - GARCH(1,1) Conditional Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("hul_garch_volatility.png")
plt.show()
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Step 1: Download fresh Nestlé data
nestle = yf.download("NESTLEIND.NS", start="2024-01-01", end="2024-07-01", auto_adjust=True)

# Step 2: Check and compute returns
if 'Close' not in nestle.columns:
    print("Error: 'Close' column not found in Nestlé data.")
else:
    nestle['Return'] = nestle['Close'].pct_change() * 100  # Convert to % return
    nestle_garch_data = nestle[['Return']].dropna()

    if nestle_garch_data.empty:
        print("Error: Nestlé return data is empty after cleaning.")
    else:
        # Step 3: Fit GARCH(1,1)
        nestle_garch_model = arch_model(nestle_garch_data['Return'], vol='Garch', p=1, q=1)
        nestle_garch_result = nestle_garch_model.fit(disp='off')

        # Step 4: Forecast
        forecast = nestle_garch_result.forecast(horizon=5)
        vol_forecast = forecast.variance.iloc[-1].values**0.5
        print(f"GARCH(1,1) 5-day Volatility Forecast: {vol_forecast[0]:.2f}%")

        # Step 5: Plot
        plt.figure(figsize=(10, 4))
        plt.plot(nestle_garch_result.conditional_volatility, color='green')
        plt.title("Nestlé - GARCH(1,1) Conditional Volatility")
        plt.xlabel("Date")
        plt.ylabel("Volatility (%)")
        plt.tight_layout()
        plt.savefig("nestle_garch_volatility.png")
        plt.show()


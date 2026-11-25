import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


# ==========================================================
# 1. Load Wind Turbine Dataset
# ==========================================================

data = pd.read_csv("Turbine_Data.csv")

print(data.head())
print(data.shape)


# ==========================================================
# 2. Data Preprocessing
# ==========================================================

if "Unnamed: 0" in data.columns:
    data.drop("Unnamed: 0", axis=1, inplace=True)

data = data.dropna(axis=1, how="all")

# ---------- Plot: Missing values BEFORE cleaning ----------
plt.figure(figsize=(14, 5))
missing = data.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    missing.plot(kind="bar", color="salmon")
    plt.title("Missing Values per Column (Before Cleaning)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
else:
    print("No missing values detected.")

# Fill missing values
data = data.fillna(method="ffill")
data = data.fillna(method="bfill")

# Categorical handling
categorical = [col for col in data.columns if data[col].dtype == "object"]
print("Categorical:", categorical)

data = pd.get_dummies(data, columns=categorical)


# ==========================================================
# Target and Features
# ==========================================================

target = "ActivePower"

X = data.drop(target, axis=1)
y = data[target]

mask = y.notnull()
X = X[mask]
y = y[mask]


# ==========================================================
# ---------- Plot: Target Distribution ----------
# ==========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(y, bins=60, color="steelblue", edgecolor="black")
axes[0].set_title("ActivePower Distribution")
axes[0].set_xlabel("ActivePower")
axes[0].set_ylabel("Frequency")

axes[1].boxplot(y, vert=True)
axes[1].set_title("ActivePower Boxplot")
axes[1].set_ylabel("ActivePower")

plt.tight_layout()
plt.show()


# ==========================================================
# ---------- Plot: Correlation Heatmap ----------
# ==========================================================

plt.figure(figsize=(16, 12))
corr = data.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, cbar_kws={"shrink": 0.7})
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


# ==========================================================
# Normalization
# ==========================================================

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(np.array(y).reshape(-1, 1))


# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)


# ==========================================================
# 3. Kolmogorov-Arnold Network Layer
# ==========================================================

class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.phi = nn.Parameter(torch.randn(output_dim, input_dim))

    def forward(self, x):
        x1 = self.linear(x)
        adaptive = torch.sin(torch.matmul(x, self.phi.T))
        return x1 + adaptive


# ==========================================================
# 4. WindKAN-Net Architecture
# ==========================================================

class WindKANNet(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        self.feature_encoder = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.kan1 = KANLayer(128, 64)
        self.kan2 = KANLayer(64, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = self.feature_encoder(x)
        x = torch.relu(self.kan1(x))
        x = torch.relu(self.kan2(x))
        output = self.output_layer(x)
        return output


# ==========================================================
# 5. Model Training
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = WindKANNet(X_train.shape[1]).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

epochs = 100
loss_history = []
val_loss_history = []

# Prepare test tensors on device for validation
X_test_dev = X_test.to(device)
y_test_dev = y_test.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_dev)
        val_loss = criterion(val_pred, y_test_dev).item()
    val_loss_history.append(val_loss)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch: {epoch+1} | Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}")


# ==========================================================
# 6. Prediction
# ==========================================================

model.eval()
with torch.no_grad():
    prediction = model(X_test_dev)

prediction = prediction.cpu().numpy()

prediction = scaler_y.inverse_transform(prediction)
actual = scaler_y.inverse_transform(y_test.numpy())


# ==========================================================
# 7. Evaluation Metrics
# ==========================================================

rmse = np.sqrt(mean_squared_error(actual, prediction))
mae = mean_absolute_error(actual, prediction)
r2 = r2_score(actual, prediction)

mape = np.mean(np.abs((actual - prediction) / np.clip(np.abs(actual), 1e-6, None))) * 100

print("\nWindKAN-Net Performance")
print("-------------------------")
print("RMSE :", rmse)
print("MAE  :", mae)
print("R2   :", r2)
print("MAPE :", mape, "%")


# ==========================================================
# 8. PLOT 1: Training + Validation Loss
# ==========================================================

plt.figure(figsize=(9, 5))
plt.plot(loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("WindKAN-Net Training Convergence")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ==========================================================
# 9. PLOT 2: Actual vs Predicted (First 300 Samples)
# ==========================================================

plt.figure(figsize=(12, 5))
plt.plot(actual[:300], label="Actual", linewidth=1.5)
plt.plot(prediction[:300], label="Predicted", linewidth=1.5, alpha=0.8)
plt.xlabel("Samples")
plt.ylabel("Active Power")
plt.title("Wind Power Forecasting using WindKAN-Net")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ==========================================================
# 10. PLOT 3: Full Test Set Actual vs Predicted
# ==========================================================

plt.figure(figsize=(14, 5))
plt.plot(actual, label="Actual", linewidth=1.2)
plt.plot(prediction, label="Predicted", linewidth=1.2, alpha=0.8)
plt.xlabel("Samples")
plt.ylabel("Active Power")
plt.title("Full Test Set: Actual vs Predicted")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ==========================================================
# 11. PLOT 4: Scatter Plot Actual vs Predicted
# ==========================================================

plt.figure(figsize=(7, 7))
plt.scatter(actual, prediction, alpha=0.4, s=12, color="teal")
lims = [min(actual.min(), prediction.min()), max(actual.max(), prediction.max())]
plt.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
plt.xlabel("Actual Active Power")
plt.ylabel("Predicted Active Power")
plt.title(f"Actual vs Predicted (R² = {r2:.4f})")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ==========================================================
# 12. PLOT 5: Residual Plot
# ==========================================================

residuals = actual.flatten() - prediction.flatten()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].scatter(prediction.flatten(), residuals, alpha=0.4, s=12, color="purple")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Residual (Actual - Predicted)")
axes[0].set_title("Residuals vs Predicted")
axes[0].grid()

axes[1].hist(residuals, bins=60, color="coral", edgecolor="black")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution")
axes[1].grid()

plt.tight_layout()
plt.show()


# ==========================================================
# 13. PLOT 6: Error Distribution
# ==========================================================

abs_error = np.abs(residuals)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(abs_error, bins=60, color="mediumseagreen", edgecolor="black")
axes[0].set_xlabel("Absolute Error")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Absolute Error Distribution")
axes[0].grid()

axes[1].plot(abs_error[:300], color="darkorange")
axes[1].set_xlabel("Sample")
axes[1].set_ylabel("Absolute Error")
axes[1].set_title("Absolute Error Over Time (First 300)")
axes[1].grid()

plt.tight_layout()
plt.show()


# ==========================================================
# 14. PLOT 7: Zoomed Comparison (First 100 Samples)
# ==========================================================

plt.figure(figsize=(12, 5))
plt.plot(actual[:100], "o-", label="Actual", markersize=3)
plt.plot(prediction[:100], "s-", label="Predicted", markersize=3, alpha=0.8)
plt.xlabel("Samples")
plt.ylabel("Active Power")
plt.title("Zoomed View (First 100 Samples)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ==========================================================
# 15. PLOT 8: Metrics Bar Chart
# ==========================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics = {
    "RMSE": rmse,
    "MAE": mae,
    "MAPE (%)": mape
}

for ax, (name, value) in zip(axes, metrics.items()):
    ax.bar([name], [value], color="royalblue")
    ax.set_title(f"{name} = {value:.4f}")
    ax.grid(axis="y")

plt.suptitle("WindKAN-Net Evaluation Metrics", fontsize=14)
plt.tight_layout()
plt.show()


# ==========================================================
# 16. PLOT 9: Quantile-Quantile Style (Sorted Residuals)
# ==========================================================

sorted_residuals = np.sort(residuals)

plt.figure(figsize=(9, 5))
plt.plot(sorted_residuals, color="darkred")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Sorted Sample Index")
plt.ylabel("Residual")
plt.title("Sorted Residuals")
plt.grid()
plt.tight_layout()
plt.show()


# ==========================================================
# 17. PLOT 10: KDE of Actual vs Predicted
# ==========================================================

plt.figure(figsize=(10, 5))
sns.kdeplot(actual.flatten(), label="Actual", fill=True, color="blue", alpha=0.4)
sns.kdeplot(prediction.flatten(), label="Predicted", fill=True, color="red", alpha=0.4)
plt.xlabel("Active Power")
plt.ylabel("Density")
plt.title("KDE: Actual vs Predicted Distribution")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ==========================================================
# 18. PLOT 11: Rolling Mean Error (Window=50)
# ==========================================================

window = 50
rolling_mae = pd.Series(abs_error).rolling(window).mean()

plt.figure(figsize=(12, 5))
plt.plot(rolling_mae, color="darkviolet")
plt.xlabel("Sample")
plt.ylabel(f"Rolling MAE (window={window})")
plt.title("Rolling Mean Absolute Error")
plt.grid()
plt.tight_layout()
plt.show()


# ==========================================================
# 19. PLOT 12: Learning Rate Sensitivity (Optional Retrain)
# ==========================================================

lrs = [1e-4, 1e-3, 1e-2]
lr_histories = {}

for lr in lrs:
    m = WindKANNet(X_train.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    hist = []

    for epoch in range(20):
        m.train()
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = m(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        hist.append(total / len(train_loader))

    lr_histories[lr] = hist

plt.figure(figsize=(10, 5))
for lr, hist in lr_histories.items():
    plt.plot(hist, label=f"lr = {lr}")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Learning Rate Sensitivity (20 Epochs)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# ==========================================================
# 20. Summary Table
# ==========================================================

summary = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R²", "MAPE (%)"],
    "Value": [rmse, mae, r2, mape]
})
print("\n===== Final Summary =====")
print(summary.to_string(index=False))

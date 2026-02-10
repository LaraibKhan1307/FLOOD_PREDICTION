import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# =============================
# LOAD DATA
# =============================
df = pd.read_csv("train.csv")
target_col = df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]
X = X.iloc[:, 1:]  

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# SCALE
# =============================
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)

dump(scaler, "scaler.joblib")

# =============================
# TRAIN SKLEARN MODELS
# =============================
models = {
    "linear_model.joblib": LinearRegression(),
    "ridge_model.joblib": Ridge(alpha=1.0),
    "lasso_model.joblib": Lasso(alpha=0.001, max_iter=5000),
    "elasticnet_model.joblib": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "dt_model.joblib": DecisionTreeRegressor(max_depth=8, min_samples_leaf=50),
    "rf_model.joblib": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=30,
        n_jobs=-1,
        random_state=42
    )
}

for name, model in models.items():
    model.fit(X_train_std, y_train)
    dump(model, name)

# =============================
# NEURAL NETWORK
# =============================
class ProbabilisticMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(32, 1)
        self.log_var_head = nn.Linear(32, 1)

    def forward(self, x):
        h = self.net(x)
        mean = torch.sigmoid(self.mean_head(h))
        log_var = self.log_var_head(h)
        return mean, log_var

model = ProbabilisticMLP(X_train_std.shape[1])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

Xtr = torch.tensor(X_train_std, dtype=torch.float32)
ytr = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

for _ in range(50):
    optimizer.zero_grad()
    mean, log_var = model(Xtr)
    loss = torch.mean(0.5 * (log_var + (ytr - mean)**2 / torch.exp(log_var)))
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "mlp_model.pt")

print("✅ All models trained and saved.")

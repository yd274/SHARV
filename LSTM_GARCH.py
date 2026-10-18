import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# Force random seed for exact reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ───────────────────────────────────────────────────────────────────────
# 1. THE NEURAL NETWORK MODEL (Academic Architecture)
# ───────────────────────────────────────────────────────────────────────
class LSTMGARCH(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Recurrent layer to capture non-linear temporal volatility clusters
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Linear layer mapping hidden states to a single adjustment multiplier
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Shape out: [1, Sequence_Length, hidden_dim]
        lstm_out, _ = self.lstm(x)

        # Softplus ensures the multiplier factor is strictly greater than 0
        multipliers = F.softplus(self.linear(lstm_out))
        return multipliers.squeeze()


# ───────────────────────────────────────────────────────────────────────
# 2. THE PIPELINE WRAPPER (Tailored for 100 * log returns)
# ───────────────────────────────────────────────────────────────────────
class VolatilityModel:
    def __init__(self, hidden_dim=16, lr=0.005):
        self.model = LSTMGARCH(input_dim=3, hidden_dim=hidden_dim)
        self.lr = lr
        self.garch_fit = None
        self._fitted_mean = 0.0  # Explicitly stores the estimated mean (mu)

    @property
    def mean(self):
        """Extracts the fitted constant conditional mean (mu) from the GARCH engine"""
        if self.garch_fit is None:
            raise ValueError("Model must be fitted first.")
        return self._fitted_mean

    def _get_garch_features(self, returns_np):
        """Stage 1: Generate statistical baseline data using standard GARCH(1,1)"""
        # Fits directly on percentage returns without internal rescaling
        am = arch_model(returns_np, vol='Garch', p=1, q=1, rescale=False)
        self.garch_fit = am.fit(disp='off')

        # Save the fitted constant mean (mu)
        self._fitted_mean = self.garch_fit.params['mu']

        garch_var = self.garch_fit.conditional_volatility ** 2
        garch_res = self.garch_fit.resid  # This is already (returns - mu)

        return returns_np, garch_var, garch_res

    def fit(self, returns, epochs=150, verbose=False):
        """Stage 2: Align arrays and optimize network weights with explicit mean correction"""
        # Ensure input array is strictly 1D to prevent dimensionality concatenation crashes
        returns_np = np.asarray(returns, dtype=np.float32).flatten()
        r, g_var, g_res = self._get_garch_features(returns_np)

        # Create lagging historical sequences (t-1 inputs)
        feat_returns = np.concatenate([[0.0], r[:-1]])
        feat_garch_var = g_var
        feat_garch_res = np.concatenate([[0.0], g_res[:-1]])

        # Structure into a 3D matrix shape PyTorch expects: [Batch=1, Time_Steps, Features=3]
        features = np.stack([feat_returns, feat_garch_var, feat_garch_res], axis=1)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        target_variance = torch.tensor(g_var, dtype=torch.float32)

        # Subtract the fitted mean explicitly to compute true residual shocks (eps_sq)
        eps_sq = torch.tensor((r - self._fitted_mean) ** 2, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()

        if verbose:
            print(f"Training LSTM-GARCH with mean mu = {self._fitted_mean:.6f}...")

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Predict dynamic adjustments and apply them to the GARCH baseline
            multipliers = self.model(features_tensor)
            final_variance = torch.clamp(target_variance * multipliers, min=1e-6)

            # True Gaussian Negative Log-Likelihood Loss using centered residuals
            loss = 0.5 * torch.mean(torch.log(final_variance) + eps_sq / final_variance)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 25 == 0 and verbose:
                print(f"Epoch {epoch + 1}/{epochs} | Negative Log-Likelihood Loss: {loss.item():.6f}")

        self.model.eval()
        return self

    def predict(self, returns):
        """Output the final calibrated percentage volatility array (Standard Deviation)"""
        # Flatten inputs to maintain consistency
        returns_np = np.asarray(returns, dtype=np.float32).flatten()
        r, g_var, g_res = self._get_garch_features(returns_np)

        feat_returns = np.concatenate([[0.0], r[:-1]])
        feat_garch_var = g_var
        feat_garch_res = np.concatenate([[0.0], g_res[:-1]])

        features = np.stack([feat_returns, feat_garch_var, feat_garch_res], axis=1)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            multipliers = self.model(features_tensor).numpy()

        # Computes conditional variance and returns standard deviation natively in percentage space
        final_variance = g_var * multipliers
        return np.sqrt(final_variance)

    def standardised_residuals(self, returns):
        """Calculates true standardised residuals: (r_t - mu) / sigma_t"""
        returns_np = np.asarray(returns, dtype=np.float32).flatten()
        raw_residuals = returns_np - self.mean
        hybrid_volatility = self.predict(returns_np)
        return raw_residuals / hybrid_volatility

    def forecast(self, returns, horizon=5):
        """
        Generates out-of-sample multi-step recursive volatility forecasts.

        Returns:
            An array of length 'horizon' containing predicted percentage volatilities.
        """
        if self.garch_fit is None:
            raise ValueError("Model must be fitted before forecasting.")

        returns_np = np.asarray(returns, dtype=np.float32).flatten()

        # Pull the structural historical matrices up to time T
        r, g_var, g_res = self._get_garch_features(returns_np)

        feat_returns = np.concatenate([[0.0], r[:-1]])
        feat_garch_var = g_var
        feat_garch_res = np.concatenate([[0.0], g_res[:-1]])

        features = np.stack([feat_returns, feat_garch_var, feat_garch_res], axis=1)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Pull baseline parameters for out-of-sample extrapolation
        mu = self.mean
        omega = self.garch_fit.params['omega']
        alpha = self.garch_fit.params['alpha[1]']
        beta = self.garch_fit.params['beta[1]']

        # Propagate the LSTM through history to get the terminal hidden states at time T
        self.model.eval()
        with torch.no_grad():
            _, (h_t, c_t) = self.model.lstm(features_tensor)

        forecasted_variances = []

        # Seed out-of-sample generation trackers with terminal states at time T
        pred_var_prev = g_var[-1]
        pred_res_prev = g_res[-1]
        pred_ret_prev = r[-1]

        # Recursive Forecasting Loop
        for k in range(horizon):
            step_features = np.array([[pred_ret_prev, pred_var_prev, pred_res_prev]], dtype=np.float32)
            step_tensor = torch.tensor(step_features, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                lstm_step_out, (h_t, c_t) = self.model.lstm(step_tensor, (h_t, c_t))
                multiplier = F.softplus(self.model.linear(lstm_step_out)).item()

            # Extrapolate baseline variance via GARCH expectations (E[epsilon^2] = sigma^2)
            garch_baseline_var = omega + alpha * (pred_res_prev ** 2) + beta * pred_var_prev

            # Combine to get the out-of-sample hybrid point
            hybrid_variance = garch_baseline_var * multiplier
            forecasted_variances.append(hybrid_variance)

            # Update trackers for step T + k + 1
            pred_var_prev = garch_baseline_var
            pred_ret_prev = mu  # E[r_{t+k}] = mu
            pred_res_prev = np.sqrt(garch_baseline_var)  # E[epsilon_{t+k}^2] = baseline_var

        return np.sqrt(np.array(forecasted_variances))


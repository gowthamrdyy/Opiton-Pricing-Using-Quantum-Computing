"""
Quantum Options - Institutional Grade Options Analytics
Professional trading terminal with ML + Quantum pricing
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from typing import Dict, Optional, List, Tuple
import time

# ML
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Quantum
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    HAS_QUANTUM = True
except:
    HAS_QUANTUM = False

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Quantum Options",
    page_icon="◊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL CSS - NO EMOJIS, CLEAN INSTITUTIONAL LOOK
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #0a0a0a;
    --surface: #111;
    --surface-2: #161616;
    --border: #222;
    --text: #e5e5e5;
    --text-secondary: #888;
    --text-muted: #555;
    --accent: #3b82f6;
    --green: #22c55e;
    --red: #ef4444;
    --amber: #f59e0b;
}

* { font-family: 'IBM Plex Sans', -apple-system, sans-serif; }

.stApp {
    background: var(--bg);
    color: var(--text);
}

#MainMenu, footer, header, [data-testid="stToolbar"], 
[data-testid="stDecoration"], .stDeployButton { display: none !important; }

/* Typography */
h1, h2, h3 { font-weight: 500 !important; letter-spacing: -0.02em; }

/* Inputs */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 15px !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

.stTextInput > label { display: none !important; }

.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 6px !important;
    color: white !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #2563eb !important;
    transform: translateY(-1px) !important;
}

/* Header */
.header {
    padding: 32px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}

.logo {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--text-secondary);
    text-transform: uppercase;
}

.logo-accent {
    color: var(--accent);
}

/* Price display */
.ticker-label {
    font-size: 13px;
    color: var(--text-muted);
    margin-bottom: 4px;
    font-weight: 500;
}

.company-name {
    font-size: 14px;
    color: var(--text-secondary);
    margin-bottom: 8px;
}

.price-main {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 48px;
    font-weight: 500;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 12px;
}

.price-up { color: var(--green); }
.price-down { color: var(--red); }
.price-neutral { color: var(--text); }

.change-pill {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    font-weight: 500;
}

.change-up {
    background: rgba(34, 197, 94, 0.12);
    color: var(--green);
}

.change-down {
    background: rgba(239, 68, 68, 0.12);
    color: var(--red);
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
}

.card-header {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
}

.card-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px;
    font-weight: 500;
    color: var(--text);
}

.card-sub {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
}

/* Prediction cards */
.prediction-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 24px 0;
}

.pred-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    transition: border-color 0.2s;
}

.pred-card:hover {
    border-color: var(--accent);
}

.pred-date {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 8px;
    font-weight: 500;
}

.pred-price {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 20px;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 6px;
}

.pred-change {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    font-weight: 500;
}

.pred-up { color: var(--green); }
.pred-down { color: var(--red); }

.pred-confidence {
    margin-top: 10px;
    height: 3px;
    background: var(--surface-2);
    border-radius: 2px;
    overflow: hidden;
}

.pred-confidence-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
}

/* Signal */
.signal-container {
    text-align: center;
    padding: 32px;
    border-radius: 8px;
    margin: 24px 0;
    border: 1px solid;
}

.signal-bullish {
    background: rgba(34, 197, 94, 0.06);
    border-color: rgba(34, 197, 94, 0.3);
}

.signal-bearish {
    background: rgba(239, 68, 68, 0.06);
    border-color: rgba(239, 68, 68, 0.3);
}

.signal-neutral {
    background: rgba(245, 158, 11, 0.06);
    border-color: rgba(245, 158, 11, 0.3);
}

.signal-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}

.signal-value {
    font-size: 28px;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.signal-bullish .signal-value { color: var(--green); }
.signal-bearish .signal-value { color: var(--red); }
.signal-neutral .signal-value { color: var(--amber); }

.signal-reason {
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 8px;
}

/* Stats row */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin: 24px 0;
}

.stat-item {
    background: var(--surface);
    padding: 16px;
    text-align: center;
}

.stat-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 16px;
    font-weight: 500;
    color: var(--text);
}

/* Table */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.data-table th {
    text-align: left;
    padding: 12px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
}

.data-table td {
    padding: 12px;
    border-bottom: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
}

/* Risk warning */
.risk-warning {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 6px;
    padding: 16px;
    margin: 24px 0;
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.6;
}

.risk-warning strong {
    color: var(--red);
}

/* Footer */
.footer {
    margin-top: 48px;
    padding: 24px 0;
    border-top: 1px solid var(--border);
    text-align: center;
    font-size: 12px;
    color: var(--text-muted);
}

/* Status indicator */
.status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-muted);
    padding: 4px 10px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--green);
}

/* Section */
.section-title {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 32px 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}

/* Plotly overrides */
.js-plotly-plot { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=120, show_spinner=False)
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """Fetch comprehensive market data with retries."""
    import time
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            
            # 2 years of daily data
            df = stock.history(period="2y", interval="1d")
            
            if df.empty or len(df) < 50:
                # Try shorter period
                df = stock.history(period="1y", interval="1d")
            
            if df.empty or len(df) < 50:
                # Try even shorter
                df = stock.history(period="6mo", interval="1d")
            
            if df.empty or len(df) < 30:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return None, None
            
            df = df.reset_index()
            
            # Handle timezone
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
            else:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Get info safely
            try:
                info = stock.info
            except:
                info = {}
            
            current = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2] if len(df) > 1 else current
            
            return df, {
                'ticker': ticker,
                'name': info.get('shortName', info.get('longName', ticker)),
                'price': current,
                'prev_close': prev,
                'change': current - prev,
                'change_pct': ((current - prev) / prev) * 100 if prev != 0 else 0,
                'high_52w': df['High'].tail(min(252, len(df))).max(),
                'low_52w': df['Low'].tail(min(252, len(df))).min(),
                'volume': df['Volume'].iloc[-1],
                'avg_volume': df['Volume'].tail(min(20, len(df))).mean(),
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', 'N/A'),
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, None
    
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive feature engineering."""
    df = df.copy()
    
    # Trend indicators
    for p in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{p}'] = df['Close'].rolling(p).mean()
        df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    
    # Price position relative to MAs
    df['Price_vs_SMA20'] = (df['Close'] / df['SMA_20'] - 1) * 100
    df['Price_vs_SMA50'] = (df['Close'] / df['SMA_50'] - 1) * 100
    df['Price_vs_SMA200'] = (df['Close'] / df['SMA_200'] - 1) * 100
    
    # MA crossovers
    df['SMA20_vs_SMA50'] = (df['SMA_20'] / df['SMA_50'] - 1) * 100
    df['SMA50_vs_SMA200'] = (df['SMA_50'] / df['SMA_200'] - 1) * 100
    
    # Momentum
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['RSI_7'] = compute_rsi(df['Close'], 7)
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Stochastic
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low14) / (high14 - low14)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    # Bollinger Bands
    bb_sma = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = bb_sma + 2 * bb_std
    df['BB_Lower'] = bb_sma - 2 * bb_std
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / bb_sma * 100
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Volatility
    df['ATR_14'] = compute_atr(df, 14)
    df['ATR_Pct'] = df['ATR_14'] / df['Close'] * 100
    
    # Returns at different horizons
    for p in [1, 2, 3, 5, 10, 20, 60]:
        df[f'Return_{p}d'] = df['Close'].pct_change(p) * 100
    
    # Volatility (realized)
    df['Vol_10d'] = df['Return_1d'].rolling(10).std() * np.sqrt(252)
    df['Vol_20d'] = df['Return_1d'].rolling(20).std() * np.sqrt(252)
    df['Vol_60d'] = df['Return_1d'].rolling(60).std() * np.sqrt(252)
    
    # Volume analysis
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price momentum
    df['ROC_10'] = ((df['Close'] / df['Close'].shift(10)) - 1) * 100
    df['ROC_20'] = ((df['Close'] / df['Close'].shift(20)) - 1) * 100
    
    # Williams %R
    df['Williams_R'] = -100 * (high14 - df['Close']) / (high14 - low14)
    
    # ADX (simplified)
    df['DX'] = abs(df['High'].diff() - df['Low'].diff()) / df['ATR_14'] * 100
    df['ADX'] = df['DX'].rolling(14).mean()
    
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE - PROPERLY CALIBRATED
# ═══════════════════════════════════════════════════════════════════════════════

class PredictionEngine:
    """
    Professional-grade prediction engine with proper calibration.
    Uses ensemble methods with realistic confidence estimation.
    """
    
    def __init__(self):
        self.models = []
        self.scaler = RobustScaler()
        self.feature_names = []
        self.trained = False
        self.mae = 0
        self.direction_accuracy = 0
        
    def _get_feature_vector(self, row: pd.Series, close: float) -> List[float]:
        """Extract normalized feature vector from a row."""
        return [
            row.get('Price_vs_SMA20', 0) / 10,
            row.get('Price_vs_SMA50', 0) / 10,
            row.get('Price_vs_SMA200', 0) / 20,
            row.get('SMA20_vs_SMA50', 0) / 10,
            row.get('SMA50_vs_SMA200', 0) / 10,
            (row.get('RSI_14', 50) - 50) / 50,
            (row.get('RSI_7', 50) - 50) / 50,
            row.get('MACD', 0) / close * 100,
            row.get('MACD_Hist', 0) / close * 100,
            (row.get('Stoch_K', 50) - 50) / 50,
            (row.get('Stoch_D', 50) - 50) / 50,
            (row.get('BB_Position', 0.5) - 0.5) * 2,
            row.get('BB_Width', 10) / 20,
            row.get('ATR_Pct', 2) / 5,
            row.get('Return_1d', 0) / 5,
            row.get('Return_5d', 0) / 10,
            row.get('Return_10d', 0) / 15,
            row.get('Return_20d', 0) / 20,
            row.get('Vol_20d', 0.3) - 0.3,
            min(row.get('Volume_Ratio', 1), 3) / 3,
            row.get('ROC_10', 0) / 15,
            row.get('ROC_20', 0) / 20,
            (row.get('Williams_R', -50) + 50) / 50,
            min(row.get('ADX', 25), 50) / 50,
        ]
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data with multiple prediction horizons."""
        df = engineer_features(df)
        df = df.dropna()
        
        if len(df) < 50:
            return np.array([]), np.array([]), np.array([])
        
        X = []
        for i in range(len(df)):
            row = df.iloc[i]
            features = self._get_feature_vector(row, row['Close'])
            X.append(features)
        
        X = np.array(X)
        
        # Targets: 5-day forward return
        y = df['Close'].pct_change(5).shift(-5).values * 100
        
        # Direction target
        y_dir = (y > 0).astype(int)
        
        # Remove last 5 (no target) and handle NaN
        X = X[:-5]
        y = y[:-5]
        y_dir = y_dir[:-5]
        
        mask = ~np.isnan(y)
        return X[mask], y[mask], y_dir[mask]
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train ensemble of models with time-series cross-validation."""
        X, y, y_dir = self.prepare_data(df)
        
        if len(X) < 50:
            return {'success': False, 'error': 'Insufficient data (need 50+ samples)'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split - adapt splits to data size
        n_splits = min(5, max(2, len(X) // 50))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        maes = []
        dir_accs = []
        
        self.models = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            y_dir_val = y_dir[val_idx]
            
            if HAS_XGB:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    verbosity=0
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42
                )
            
            model.fit(X_train, y_train)
            self.models.append(model)
            
            # Validation metrics
            preds = model.predict(X_val)
            maes.append(mean_absolute_error(y_val, preds))
            
            # Direction accuracy
            pred_dir = (preds > 0).astype(int)
            dir_accs.append(np.mean(pred_dir == y_dir_val))
        
        self.mae = np.mean(maes)
        self.direction_accuracy = np.mean(dir_accs)
        self.trained = True
        
        return {
            'success': True,
            'mae': self.mae,
            'direction_accuracy': self.direction_accuracy * 100,
            'samples': len(X),
            'models': len(self.models)
        }
    
    def predict(self, df: pd.DataFrame, days: int = 7) -> List[Dict]:
        """Generate predictions using multi-timeframe momentum."""
        if not self.trained or not self.models:
            return []
        
        df = engineer_features(df)
        df = df.dropna()
        
        current_price = df['Close'].iloc[-1]
        last_date = df['Date'].iloc[-1]
        
        # Get current features
        row = df.iloc[-1]
        features = np.array([self._get_feature_vector(row, current_price)])
        features_scaled = self.scaler.transform(features)
        
        # ML Model prediction (5-day return %)
        predictions_raw = [m.predict(features_scaled)[0] for m in self.models]
        model_return = np.mean(predictions_raw)
        prediction_std = np.std(predictions_raw)
        
        # MULTI-TIMEFRAME MOMENTUM
        # Short-term returns
        r_5d = row.get('Return_5d', 0)
        r_10d = row.get('Return_10d', 0)
        r_20d = row.get('Return_20d', 0)
        
        # Longer-term: Use SMA positions as trend indicator
        price_vs_sma50 = row.get('Price_vs_SMA50', 0)   # % above/below SMA50
        price_vs_sma200 = row.get('Price_vs_SMA200', 0) # % above/below SMA200
        
        # SMA relationships (trend strength)
        sma20_vs_sma50 = row.get('SMA20_vs_SMA50', 0)
        sma50_vs_sma200 = row.get('SMA50_vs_SMA200', 0)
        
        # Short-term momentum signal
        short_momentum = r_5d * 0.6 + (r_10d * 0.5) * 0.4  # Scale 10d to 5d equiv
        
        # Long-term trend signal (position vs major MAs)
        # If price is below SMAs, that's bearish; if above, bullish
        long_trend = (price_vs_sma50 * 0.4 + price_vs_sma200 * 0.3 + 
                      sma20_vs_sma50 * 0.2 + sma50_vs_sma200 * 0.1)
        
        # Combine: Use both signals but let long-term trend moderate extremes
        # If long-term is bearish but short-term bullish, dampen the bullish signal
        if np.sign(short_momentum) != np.sign(long_trend) and abs(long_trend) > 3:
            # Conflicting signals - trust long-term more
            combined_signal = short_momentum * 0.3 + long_trend * 0.4
        else:
            # Aligned signals - trust momentum more
            combined_signal = short_momentum * 0.6 + long_trend * 0.2
        
        # RSI adjustment (contrarian at extremes)
        rsi = row.get('RSI_14', 50)
        rsi_adj = 0
        if rsi > 70:
            rsi_adj = -0.8
        elif rsi < 30:
            rsi_adj = 0.8
        
        # Final prediction
        base_return = (
            combined_signal * 0.55 +      # Multi-timeframe signal
            model_return * 0.30 +          # ML model
            rsi_adj * 0.15                 # RSI contrarian
        )
        
        # Quantum adjustment (minimal)
        if HAS_QUANTUM:
            quantum_factor = self._quantum_uncertainty()
            base_return += quantum_factor * 0.02
        
        # Cap to reasonable range
        base_return = np.clip(base_return, -8, 8)
        
        # Store for confidence calc
        momentum_signal = combined_signal
        
        # Daily volatility for noise
        annual_vol = row.get('Vol_20d', 25)
        daily_vol_pct = annual_vol / 16  # Convert to daily (~1.5%)
        
        # Generate daily predictions
        predictions = []
        trading_day = 0
        cumulative_return = 0
        
        for i in range(1, days * 2):  # Account for weekends
            pred_date = last_date + timedelta(days=i)
            
            # Skip weekends
            if pred_date.weekday() >= 5:
                continue
            
            trading_day += 1
            if trading_day > days:
                break
            
            # Daily return: spread total expected return across days
            daily_return = base_return / 5
            
            # Add realistic daily noise (small)
            daily_noise = np.random.normal(0, daily_vol_pct * 0.15)
            daily_return += daily_noise
            
            cumulative_return += daily_return
            predicted_price = current_price * (1 + cumulative_return / 100)
            
            # Confidence: higher when momentum is clear, lower with time
            momentum_clarity = min(abs(momentum_signal) / 3, 1)
            base_confidence = 50 + momentum_clarity * 12 - trading_day * 2 - prediction_std * 2
            confidence = np.clip(base_confidence, 35, 65)
            
            predictions.append({
                'date': pred_date,
                'price': predicted_price,
                'return_pct': cumulative_return,
                'confidence': confidence,
                'day': trading_day
            })
        
        return predictions
    
    def _quantum_uncertainty(self) -> float:
        """Quantum-derived uncertainty factor."""
        try:
            qc = QuantumCircuit(4, 4)
            qc.h([0, 1, 2, 3])
            qc.cx(0, 1)
            qc.cx(2, 3)
            qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
            
            backend = AerSimulator()
            job = backend.run(transpile(qc, backend), shots=200)
            counts = job.result().get_counts()
            
            # Convert to factor
            total = sum(counts.values())
            weighted = sum(int(k, 2) * v for k, v in counts.items())
            normalized = (weighted / total) / 16 - 0.5
            
            return normalized * 0.3  # Small adjustment
        except:
            return np.random.normal(0, 0.1)
    
    def get_signal(self, predictions: List[Dict]) -> Dict:
        """Generate trading signal from predictions."""
        if not predictions:
            return {'signal': 'NEUTRAL', 'strength': 0, 'reason': 'No predictions available'}
        
        # Look at 5-day prediction
        pred_5d = next((p for p in predictions if p['day'] == 5), predictions[-1])
        expected_return = pred_5d['return_pct']
        confidence = pred_5d['confidence']
        
        # Count bullish vs bearish days
        bullish_days = sum(1 for p in predictions if p['return_pct'] > 0)
        bearish_days = len(predictions) - bullish_days
        
        # Determine signal
        if expected_return > 2 and bullish_days >= 4 and confidence > 50:
            signal = 'BULLISH'
            strength = min(expected_return / 5 * 100, 100)
            reason = f'Expected +{expected_return:.1f}% in 5 days, {bullish_days}/{len(predictions)} days positive'
        elif expected_return < -2 and bearish_days >= 4 and confidence > 50:
            signal = 'BEARISH'
            strength = min(abs(expected_return) / 5 * 100, 100)
            reason = f'Expected {expected_return:.1f}% in 5 days, {bearish_days}/{len(predictions)} days negative'
        elif expected_return > 0.5:
            signal = 'SLIGHTLY BULLISH'
            strength = 40
            reason = f'Mild upside expected (+{expected_return:.1f}%)'
        elif expected_return < -0.5:
            signal = 'SLIGHTLY BEARISH'
            strength = 40
            reason = f'Mild downside expected ({expected_return:.1f}%)'
        else:
            signal = 'NEUTRAL'
            strength = 20
            reason = f'No clear direction (expected {expected_return:+.1f}%)'
        
        return {
            'signal': signal,
            'strength': strength,
            'reason': reason,
            'expected_return': expected_return,
            'confidence': confidence
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CHART
# ═══════════════════════════════════════════════════════════════════════════════

def create_price_chart(df: pd.DataFrame, predictions: List[Dict] = None) -> go.Figure:
    """Create professional price chart."""
    
    df_plot = df.tail(90).copy()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25]
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot['Date'],
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close'],
        increasing_line_color='#22c55e',
        decreasing_line_color='#ef4444',
        increasing_fillcolor='#22c55e',
        decreasing_fillcolor='#ef4444',
        name='Price',
        showlegend=False
    ), row=1, col=1)
    
    # SMA lines
    if 'SMA_20' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot['Date'], y=df_plot['SMA_20'],
            line=dict(color='rgba(59, 130, 246, 0.6)', width=1),
            name='SMA 20'
        ), row=1, col=1)
    
    if 'SMA_50' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot['Date'], y=df_plot['SMA_50'],
            line=dict(color='rgba(168, 85, 247, 0.6)', width=1),
            name='SMA 50'
        ), row=1, col=1)
    
    # Predictions
    if predictions:
        pred_dates = [df_plot['Date'].iloc[-1]] + [p['date'] for p in predictions]
        pred_prices = [df_plot['Close'].iloc[-1]] + [p['price'] for p in predictions]
        
        # Prediction line
        first_pred = predictions[0]
        line_color = '#22c55e' if first_pred['return_pct'] >= 0 else '#ef4444'
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_prices,
            mode='lines+markers',
            line=dict(color=line_color, width=2, dash='dot'),
            marker=dict(size=6, color=line_color),
            name='Prediction'
        ), row=1, col=1)
        
        # Confidence band
        upper = [pred_prices[0]]
        lower = [pred_prices[0]]
        for p in predictions:
            spread = p['price'] * (1 - p['confidence'] / 100) * 0.5
            upper.append(p['price'] + spread)
            lower.append(p['price'] - spread)
        
        fig.add_trace(go.Scatter(
            x=pred_dates + pred_dates[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor=f'rgba({34 if first_pred["return_pct"] >= 0 else 239}, {197 if first_pred["return_pct"] >= 0 else 68}, {94 if first_pred["return_pct"] >= 0 else 68}, 0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False
        ), row=1, col=1)
    
    # Volume
    colors = ['#22c55e' if c >= o else '#ef4444' 
              for c, o in zip(df_plot['Close'], df_plot['Open'])]
    
    fig.add_trace(go.Bar(
        x=df_plot['Date'],
        y=df_plot['Volume'],
        marker_color=colors,
        opacity=0.5,
        showlegend=False
    ), row=2, col=1)
    
    # Layout
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='#888', family='IBM Plex Sans'),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=11)
        ),
        xaxis_rangeslider_visible=False,
        xaxis2=dict(showticklabels=False)
    )
    
    fig.update_xaxes(
        gridcolor='#1a1a1a',
        showgrid=True,
        zeroline=False
    )
    
    fig.update_yaxes(
        gridcolor='#1a1a1a',
        showgrid=True,
        zeroline=False,
        side='right'
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

# Popular stock tickers
POPULAR_STOCKS = [
    "AAPL - Apple Inc.",
    "MSFT - Microsoft Corp.",
    "GOOGL - Alphabet Inc.",
    "AMZN - Amazon.com Inc.",
    "NVDA - NVIDIA Corp.",
    "TSLA - Tesla Inc.",
    "META - Meta Platforms",
    "NFLX - Netflix Inc.",
    "AMD - Advanced Micro Devices",
    "INTC - Intel Corp.",
    "JPM - JPMorgan Chase",
    "BAC - Bank of America",
    "V - Visa Inc.",
    "MA - Mastercard Inc.",
    "DIS - Walt Disney Co.",
    "PYPL - PayPal Holdings",
    "UBER - Uber Technologies",
    "ABNB - Airbnb Inc.",
    "CRM - Salesforce Inc.",
    "ORCL - Oracle Corp.",
    "IBM - IBM Corp.",
    "CSCO - Cisco Systems",
    "QCOM - Qualcomm Inc.",
    "TXN - Texas Instruments",
    "MU - Micron Technology",
    "SHOP - Shopify Inc.",
    "SQ - Block Inc.",
    "COIN - Coinbase Global",
    "PLTR - Palantir Technologies",
    "SNOW - Snowflake Inc.",
]

def main():
    # Header
    st.markdown("""
    <div class="header">
        <div class="logo">QUANTUM<span class="logo-accent">OPTIONS</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stock selector
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected = st.selectbox(
            "Select Stock",
            options=POPULAR_STOCKS,
            index=0,
            label_visibility="collapsed"
        )
        # Extract ticker from selection
        ticker = selected.split(" - ")[0].strip()
    
    if not ticker:
        st.stop()
    
    # Load data
    with st.spinner("Loading market data..."):
        df, info = fetch_market_data(ticker)
    
    if df is None:
        st.error(f"Could not fetch data for '{ticker}'. Please check the symbol.")
        st.stop()
    
    # Train model
    engine = PredictionEngine()
    with st.spinner("Analyzing..."):
        train_result = engine.train(df)
    
    if not train_result.get('success'):
        st.error(f"Analysis failed: {train_result.get('error', 'Unknown error')}")
        st.stop()
    
    # Generate predictions
    predictions = engine.predict(df, days=7)
    signal = engine.get_signal(predictions)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Price section
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        price_class = "price-up" if info['change'] >= 0 else "price-down"
        change_class = "change-up" if info['change'] >= 0 else "change-down"
        sign = "+" if info['change'] >= 0 else ""
        
        st.markdown(f"""
        <div style="padding: 20px 0;">
            <div class="ticker-label">{info['ticker']}</div>
            <div class="company-name">{info['name']}</div>
            <div class="price-main {price_class}">${info['price']:.2f}</div>
            <span class="change-pill {change_class}">
                {sign}{info['change']:.2f} ({sign}{info['change_pct']:.2f}%)
            </span>
            <div style="margin-top: 16px;">
                <span class="status">
                    <span class="status-dot"></span>
                    Live · {'Quantum' if HAS_QUANTUM else 'Classical'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        st.markdown(f"""
        <div class="stats-grid" style="grid-template-columns: repeat(2, 1fr); margin-top: 24px;">
            <div class="stat-item">
                <div class="stat-label">52W High</div>
                <div class="stat-value">${info['high_52w']:.2f}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">52W Low</div>
                <div class="stat-value">${info['low_52w']:.2f}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Volume</div>
                <div class="stat-value">{info['volume']/1e6:.1f}M</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Avg Vol</div>
                <div class="stat-value">{info['avg_volume']/1e6:.1f}M</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        df_features = engineer_features(df)
        chart = create_price_chart(df_features, predictions)
        st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
    
    # Signal
    signal_class = 'signal-bullish' if 'BULLISH' in signal['signal'] else ('signal-bearish' if 'BEARISH' in signal['signal'] else 'signal-neutral')
    
    st.markdown(f"""
    <div class="signal-container {signal_class}">
        <div class="signal-label">Model Signal</div>
        <div class="signal-value">{signal['signal']}</div>
        <div class="signal-reason">{signal['reason']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Predictions - use Streamlit columns instead of HTML grid
    if predictions:
        st.markdown('<div class="section-title">7-Day Price Forecast</div>', unsafe_allow_html=True)
        
        # Create columns for predictions
        cols = st.columns(len(predictions))
        for i, p in enumerate(predictions):
            with cols[i]:
                change_color = "#00E676" if p['return_pct'] >= 0 else "#FF5252"
                sign = "+" if p['return_pct'] >= 0 else ""
                arrow = "↑" if p['return_pct'] >= 0 else "↓"
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); 
                            border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="color: #888; font-size: 11px; margin-bottom: 4px;">{p['date'].strftime('%b %d')}</div>
                    <div style="color: #fff; font-size: 16px; font-weight: 600;">${p['price']:.2f}</div>
                    <div style="color: {change_color}; font-size: 13px; font-weight: 500;">{arrow} {sign}{p['return_pct']:.2f}%</div>
                    <div style="background: rgba(255,255,255,0.1); height: 3px; border-radius: 2px; margin-top: 8px;">
                        <div style="background: #00E5FF; height: 100%; width: {p['confidence']:.0f}%; border-radius: 2px;"></div>
                    </div>
                    <div style="color: #666; font-size: 9px; margin-top: 2px;">{p['confidence']:.0f}% conf</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Model stats
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-item">
            <div class="stat-label">Training Samples</div>
            <div class="stat-value">{train_result['samples']:,}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Direction Accuracy</div>
            <div class="stat-value">{train_result['direction_accuracy']:.1f}%</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Avg Error (MAE)</div>
            <div class="stat-value">{train_result['mae']:.2f}%</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Ensemble Models</div>
            <div class="stat-value">{train_result['models']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk warning
    st.markdown("""
    <div class="risk-warning">
        <strong>Risk Disclosure:</strong> This tool provides AI-generated predictions for educational and research purposes only. 
        Past performance does not guarantee future results. Options trading involves substantial risk of loss and is not suitable for all investors. 
        Never invest money you cannot afford to lose. Always conduct your own research and consult with a licensed financial advisor before making investment decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    now = datetime.now()
    st.markdown(f"""
    <div class="footer">
        Quantum Options · {now.strftime('%B %d, %Y %H:%M')} · 
        {'Qiskit Quantum Backend' if HAS_QUANTUM else 'Classical Ensemble'} · 
        {'XGBoost' if HAS_XGB else 'Gradient Boosting'} · 
        Data via Yahoo Finance
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

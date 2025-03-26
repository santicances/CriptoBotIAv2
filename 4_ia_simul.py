import ccxt
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = r"C:\Users\formacionIA\Desktop\arbitraje 2\models\generacion_2_accuracy_77.78.h5"
DATASET_PATH = r"C:\Users\formacionIA\Desktop\arbitraje 2\operaciones_ganadoras.csv"
START        = "2025-03-23 09:00:00"

# ─── 1️⃣ CARGA MODELO ─────────────────────────────────────────────────────────────
print("▶️ 1. Cargando modelo TensorFlow…")
model = load_model(MODEL_PATH, compile=False)
print("✅ Modelo cargado")

# ─── 2️⃣ PREPROCESS DATASET ───────────────────────────────────────────────────────
print("▶️ 2. Cargando dataset y preparando scalers…")
df_full = pd.read_csv(DATASET_PATH)
le = LabelEncoder().fit(df_full["direccion"])
numeric_cols = ['precio_entrada','sl_price','tp_price'] + [f"close_t-{i}" for i in range(1,11)]
scaler_input = StandardScaler().fit(df_full[numeric_cols])
scaler_price = StandardScaler().fit(df_full[['sl_price','tp_price']])
print(f"✅ Dataset listo — filas: {len(df_full)}")

# ─── 3️⃣ FUNCIÓN DE DESCARGA ──────────────────────────────────────────────────────
def get_btc_data():
    print(f"▶️ 3. Descargando datos BTC desde Binance desde {START}…")
    ex = ccxt.binance()
    since = ex.parse8601(f"{START}Z")
    all_ = []
    while True:
        batch = ex.fetch_ohlcv("BTC/USDT", timeframe="1m", since=since, limit=1000)
        if not batch:
            break
        all_.extend(batch)
        last_dt = datetime.utcfromtimestamp(batch[-1][0]//1000)
        print(f"   ➡️ Fetched {len(batch)} rows up to {last_dt}")
        since = batch[-1][0] + 60000
        if last_dt >= datetime.utcnow():
            break

    df = pd.DataFrame(all_, columns=['timestamp','open','high','low','close','volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    print(f"✅ Datos descargados — total filas: {len(df)}")
    return df

df = get_btc_data()

# ─── 4️⃣ BACKTRADER STRATEGY ───────────────────────────────────────────────────────
class MLStrategy(bt.Strategy):
    params = dict(model=model, scaler_input=scaler_input, scaler_price=scaler_price, le=le)

    def __init__(self):
        from collections import deque
        self.closes = deque(maxlen=10)
        self.current = {}
        self.trades = []

    def next(self):
        self.closes.append(self.data.close[0])
        if len(self.closes) < 10 or self.position:
            return

        entry = self.data.close[0]
        feat = {
            'precio_entrada': entry, 'sl_price': entry, 'tp_price': entry,
            **{f"close_t-{i+1}": self.closes[-(i+1)] for i in range(10)},
            'direccion': 0
        }
        df_feat = pd.DataFrame([feat])
        x = self.p.scaler_input.transform(df_feat[numeric_cols])
        inp = np.concatenate([x, [[0]]], axis=1)
        _, pred_cat, _ = self.p.model.predict(inp, verbose=0)
        direction = self.p.le.inverse_transform([pred_cat.argmax()])[0]

        df_feat['direccion'] = self.p.le.transform([direction])
        x2 = self.p.scaler_input.transform(df_feat[numeric_cols])
        inp2 = np.concatenate([x2, [[df_feat['direccion'][0]]]], axis=1)
        _, _, pred_prices = self.p.model.predict(inp2, verbose=0)
        sl, tp = self.p.scaler_price.inverse_transform(pred_prices)[0]

        if direction == "Larga":
            self.buy_bracket(size=0.001, price=entry, stopprice=sl, limitprice=tp)
        else:
            self.sell_bracket(size=0.001, price=entry, stopprice=sl, limitprice=tp)

    def notify_order(self, order):
        if order.status != order.Completed:
            return
        dt = self.data.datetime.datetime(0)
        if order.parent is None:
            self.current = {"entry_dt": dt, "direction": "Larga" if order.isbuy() else "Corta", "entry_price": order.executed.price}
        else:
            self.current.update({"exit_dt": dt, "exit_price": order.executed.price, "pnl": order.executed.pnl})
            self.trades.append(self.current.copy())
            self.current = {}

# ─── 5️⃣ EJECUCIÓN BACKTEST ────────────────────────────────────────────────────────
print("▶️ 4. Preparando Backtrader y ejecutando backtest…")
data_feed = bt.feeds.PandasData(dataname=df)
cerebro = bt.Cerebro()
cerebro.adddata(data_feed)
cerebro.addstrategy(MLStrategy)
cerebro.broker.setcash(1000)
cerebro.broker.setcommission(0.001)
results = cerebro.run()
print("✅ 5. Backtest completado")

# ─── 6️⃣ RESULTADOS ───────────────────────────────────────────────────────────────
trades_df = pd.DataFrame(results[0].trades)
print("▶️ 6. Trades DataFrame generado:")
print(trades_df)

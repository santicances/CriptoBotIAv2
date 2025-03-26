import pandas as pd
import numpy as np
import random
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

# Paths
MODEL_PATH = r"C:\Users\formacionIA\Desktop\arbitraje 2\models\generacion_2_accuracy_85.33.h5"
DATA_PATH  = r"C:\Users\formacionIA\Desktop\arbitraje 2\operaciones_ganadoras.csv"

# Cargar y preprocesar los datos
df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.duplicated()]

# Generar la lista de 200 columnas OHLC (open, high, low, close) de t-1 a t-50
ohlc_cols = []
for i in range(1, 51):
    ohlc_cols.extend([f'open_t-{i}', f'high_t-{i}', f'low_t-{i}', f'close_t-{i}'])

# Definir las columnas de entrada numéricas (3 básicas + 200 OHLC = 203 columnas)
numeric_input_cols = ['precio_entrada', 'sl_price', 'tp_price'] + ohlc_cols
output_cols = ['sl_price', 'tp_price']

# Si el CSV no tiene todas las columnas esperadas, se añaden con un valor por defecto (0.0)
for col in numeric_input_cols:
    if col not in df.columns:
        df[col] = 0.0

# Seleccionar únicamente las columnas requeridas: 'direccion' + numeric_input_cols
df = df[['direccion'] + numeric_input_cols]

# Codificar la columna "direccion"
le = LabelEncoder()
df['direccion'] = le.fit_transform(df['direccion'])

# Calcular SL/TP como porcentaje de la entrada
df["sl_pct"] = (df["sl_price"] - df["precio_entrada"]).abs() / df["precio_entrada"] * 100
df["tp_pct"] = (df["tp_price"] - df["precio_entrada"]).abs() / df["precio_entrada"] * 100
mean_sl_pct = df["sl_pct"].mean()
mean_tp_pct = df["tp_pct"].mean()

# Ajustar los escaladores
scaler_input = StandardScaler().fit(df[numeric_input_cols])
scaler_prices = StandardScaler().fit(df[output_cols])

# Cargar el modelo
model = load_model(MODEL_PATH, compile=False)

# Bucle de evaluación
total = 100
correct_dir = sum_sl_err = sum_tp_err = sum_pnl = 0.0
epsilon = 1e-8

# Imprimir encabezado, incluyendo la columna "Result"
print(f"{'Idx':>3} | TRUE Dir | SL      | TP      || PRED Dir | SL      | TP      || Result | SL err   | TP err   | PnL")
print("-" * 130)

for _ in range(total):
    idx = random.randrange(len(df))
    true_dir = le.inverse_transform([df.at[idx, "direccion"]])[0]
    entry = df.at[idx, "precio_entrada"]
    true_sl, true_tp = df.at[idx, "sl_price"], df.at[idx, "tp_price"]

    # Transformar las características numéricas
    x = scaler_input.transform(df.loc[[idx], numeric_input_cols])  # Forma: (1, 203)
    # Concatenar la codificación de "direccion" (ya numérica)
    input_features = np.concatenate([x, [[df.at[idx, "direccion"]]]], axis=1)  # Forma: (1, 204)
    # Reestructurar para que tenga forma (batch_size, 204, 1)
    inp = input_features.reshape(1, -1, 1)
    
    # Realizar la predicción
    _, pred_cat, pred_prices = model.predict(inp, verbose=0)
    pred_dir = le.inverse_transform([np.argmax(pred_cat[0])])[0]
    pred_sl, pred_tp = scaler_prices.inverse_transform(pred_prices)[0]

    # Definir el resultado de la comparación
    result = "ok" if pred_dir == true_dir else "Wrong"

    sign = 1 if true_dir == "Larga" else -1
    sl_err = sign * (pred_sl - true_sl) / (abs(true_sl) + epsilon) * 100
    tp_err = sign * (pred_tp - true_tp) / (abs(true_tp) + epsilon) * 100
    pnl = ((true_tp - entry) / entry * 100) if true_dir == "Larga" else ((entry - true_tp) / entry * 100)

    sum_sl_err += sl_err
    sum_tp_err += tp_err
    sum_pnl += pnl
    if pred_dir == true_dir:
        correct_dir += 1

    print(f"{idx:3d} | {true_dir:>5} | {true_sl:8.2f} | {true_tp:8.2f} || {pred_dir:>5} | {pred_sl:8.2f} | {pred_tp:8.2f} || {result:>6} | {sl_err:+7.2f}% | {tp_err:+7.2f}% | {pnl:+6.2f}%")

# Resumen de la evaluación
accuracy = correct_dir / total * 100
mean_sl_err, mean_tp_err, mean_pnl = sum_sl_err / total, sum_tp_err / total, sum_pnl / total
abs_mean_err = (abs(mean_sl_err) + abs(mean_tp_err)) / 2
real_pnl = mean_pnl - abs_mean_err

print("\n" + "-" * 130)
print(f"Accuracy          = {accuracy:.2f}%")
print(f"Mean SL Error     = {mean_sl_err:+.2f}%")
print(f"Mean TP Error     = {mean_tp_err:+.2f}%")
print(f"Mean PnL          = {mean_pnl:+.2f}%")
print(f"Abs Mean Error    = {abs_mean_err:+.2f}%")
print(f"Real PnL          = {real_pnl:+.2f}%")
print(f"Mean SL placement = {mean_sl_pct:.2f}%")
print(f"Mean TP placement = {mean_tp_pct:.2f}%")

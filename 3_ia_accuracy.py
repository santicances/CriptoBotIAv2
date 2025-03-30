import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

# Paths de los tres modelos
MODEL_PATHS = {
    "Model_013_gen_3": r"C:\Users\formacionIA\Desktop\arbitraje 2\models\model_Ind013_gen_3.keras",
    "Model_015_gen_4": r"C:\Users\formacionIA\Desktop\arbitraje 2\models\model_Ind015_gen_4.keras",
    "Model_037_gen_5": r"C:\Users\formacionIA\Desktop\arbitraje 2\models\model_Ind037_gen_5.keras"
}
DATA_PATH  = r"C:\Users\formacionIA\Desktop\arbitraje 2\operaciones_ganadoras.csv"
TOTAL = 100
epsilon = 1e-8

# Cargar y preparar datos
df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.duplicated()]
ohlc_cols = [f'{x}_t-{i}' for i in range(1, 51) for x in ['open', 'high', 'low', 'close']]
numeric_input_cols = ['precio_entrada', 'sl_price', 'tp_price'] + ohlc_cols
output_cols = ['direccion', 'precio_entrada', 'sl_price', 'tp_price']

for col in numeric_input_cols:
    if col not in df.columns:
        df[col] = 0.0
df = df[['direccion'] + numeric_input_cols]
le = LabelEncoder()
df['direccion'] = le.fit_transform(df['direccion'])

df["sl_pct"] = (df["sl_price"] - df["precio_entrada"]).abs() / df["precio_entrada"] * 100
df["tp_pct"] = (df["tp_price"] - df["precio_entrada"]).abs() / df["precio_entrada"] * 100
mean_sl_pct = df["sl_pct"].mean()
mean_tp_pct = df["tp_pct"].mean()

scaler_input = StandardScaler().fit(df[numeric_input_cols])
scaler_prices = StandardScaler().fit(df[output_cols])

# Almacenar métricas para tabla resumen
resultados_finales = []

# Iterar sobre cada modelo
for model_name, model_path in MODEL_PATHS.items():
    print(f"\n\n=== Evaluando {model_name} ===\n")

    model = load_model(model_path, compile=False)
    pred_bin, pred_cat, pred_prices, pred_entry = None, None, None, None

    print(f"{'Idx':>3} | TRUE Dir | SL      | TP      || PRED Dir | SL      | TP      || Result | SL err   | TP err   | PnL")
    print("-" * 130)

    correct_dir = sum_sl_err = sum_tp_err = sum_pnl = 0.0

    for _ in range(TOTAL):
        idx = random.randrange(len(df))
        true_dir = le.inverse_transform([df.at[idx, "direccion"]])[0]
        entry = df.at[idx, "precio_entrada"]
        true_sl, true_tp = df.at[idx, "sl_price"], df.at[idx, "tp_price"]

        x = scaler_input.transform(df.loc[[idx], numeric_input_cols])
        input_features = np.concatenate([x, [[df.at[idx, "direccion"]]]], axis=1)
        inp = input_features.reshape(1, -1, 1)

        prediction = model.predict(inp, verbose=0)
        if isinstance(prediction, list) and len(prediction) == 4:
            pred_bin, pred_cat, pred_prices, pred_entry = prediction
        else:
            print(f"Error: modelo {model_name} no devuelve 4 salidas.")
            break

        pred_dir = le.inverse_transform([np.argmax(pred_cat[0])])[0]
        dummy_row = np.array([[0, 0, pred_prices[0][0], pred_prices[0][1]]])
        pred_sl, pred_tp = scaler_prices.inverse_transform(dummy_row)[0][2:]

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

    # Métricas agregadas para este modelo
    accuracy = correct_dir / TOTAL * 100
    mean_sl_err = sum_sl_err / TOTAL
    mean_tp_err = sum_tp_err / TOTAL
    mean_pnl = sum_pnl / TOTAL
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

    # Guardar resultados
    resultados_finales.append({
        "Modelo": model_name,
        "Accuracy (%)": round(accuracy, 2),
        "SL Error (%)": round(mean_sl_err, 2),
        "TP Error (%)": round(mean_tp_err, 2),
        "PnL (%)": round(mean_pnl, 2),
        "Real PnL (%)": round(real_pnl, 2)
    })

# Tabla resumen final
print("\n\n=== RESUMEN COMPARATIVO ===\n")
df_resultados = pd.DataFrame(resultados_finales)
print(df_resultados.to_string(index=False))

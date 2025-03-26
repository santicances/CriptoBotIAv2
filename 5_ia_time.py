import pandas as pd

# 📁 Ruta al CSV
file_path = r"C:\Users\formacionIA\Desktop\arbitraje 2\BTC_data.csv"

# 📥 Leer CSV y parsear timestamp
df = pd.read_csv(file_path)
try:
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
except:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# 📊 Calcular retorno % basado en open–close
df['return_pct'] = (df['close'] - df['open']) / df['open'] * 100

# ⏱ Crear franja de 15 minutos (formato HH:MM)
df['time_slot'] = df['timestamp'].dt.floor('15T').dt.strftime('%H:%M')

# 📈 Agrupar por franja y calcular retorno medio
slot_stats = (
    df.groupby('time_slot')['return_pct']
      .mean()
      .reset_index(name='avg_return_pct')
)

# 🔢 Generar las 96 franjas posibles
all_slots = pd.DataFrame({
    'time_slot': pd.date_range('00:00', '23:45', freq='15min').strftime('%H:%M')
})

# 🔗 Unir para asegurar todas las franjas (llenando NaN donde no haya datos)
all_stats = all_slots.merge(slot_stats, on='time_slot', how='left')

# 🔢 Ordenar cronológicamente
all_stats = all_stats.sort_values('time_slot').reset_index(drop=True)

# 🏆 Identificar mejor y peor franja (ignorando NaN)
best = all_stats.loc[all_stats['avg_return_pct'].idxmax()]
worst = all_stats.loc[all_stats['avg_return_pct'].idxmin()]

# 📋 Mostrar resultados
print("\n📊 Retorno medio (%) por cada una de las 96 franjas de 15 minutos:\n")
print(all_stats.to_string(index=False))
print(f"\n→ Franja con mayor subida media: {best['time_slot']} ({best['avg_return_pct']:.2f}%)")
print(f"→ Franja con mayor bajada media: {worst['time_slot']} ({worst['avg_return_pct']:.2f}%)")

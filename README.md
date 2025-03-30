# Análisis de Modelos para Trading de Criptomonedas

Este proyecto tiene como objetivo evaluar y comparar diferentes modelos de machine learning aplicados al trading de criptomonedas, utilizando datos históricos y métricas clave de rendimiento.

## 📊 Objetivos

- Entrenar modelos sobre datos de mercado (velas OHLC, volumen, indicadores técnicos, etc.).
- Evaluar la precisión de las predicciones.
- Comparar métricas como el retorno simulado, drawdown y ratio Sharpe.
- Visualizar resultados y tomar decisiones informadas.

## 🧰 Requisitos

- Python 3.11.9
- Ver dependencias en `requirements.txt`

## 📁 Estructura del Proyecto
:

🧩 Archivos principales
1_data_DW_GEN.py
🔽 Descarga de datos de mercado

Este script se encarga de recopilar los datos históricos necesarios para entrenar y evaluar los modelos. Puede conectarse a APIs de exchanges  y guardar los datos en formato .csv u otro formato estructurado una vez descargados los datos o si ya existen genera el archivo operaciones_ganadoras.csv que es el que va a ser utilizado paras entrenar el modelo.


Responsabilidades:
Descargar velas OHLC (Open, High, Low, Close)
Almacenar volumen, timestamps y posiblemente 
Guardar los datos en csv

2_ia_train.py
🧠 Entrenamiento del modelo de inteligencia artificial

Este script entrena modelos de machine learning o deep learning sobre los datos descargados. Puede admitir diferentes arquitecturas (LSTM, CNN,GPT,LTSM etc.) y guardar el modelo entrenado para uso posterior.

Responsabilidades:
Cargar y preprocesar datos
Entrenar el modelo (definido o parametrizable)
Guardar el modelo resultante en la carpeta models/

3_ia_accuracy.py
📊 Evaluación de la precisión del modelo

Este script evalúa el modelo entrenado utilizando datos de validación o test. Mide métricas clásicas de clasificación o regresión, y ayuda a entender si el modelo tiene capacidad predictiva.

Responsabilidades:

Cargar modelo entrenado
Comparar predicciones con resultados reales
Calcular métricas como accuracy de 
Generar visualizaciones de rendimiento

4_ia_backtest.py
💹 Simulación de trading (Backtesting)

Este script aplica las predicciones del modelo a datos históricos para simular qué habría pasado si se hubiera tomado una decisión de trading real. Evalúa la rentabilidad potencial de usar el modelo en producción.

Responsabilidades:

Simular operaciones (comprar, vender, mantener)
Calcular retorno, drawdown,, etc.
Guardar gráficos y métricas del backtest


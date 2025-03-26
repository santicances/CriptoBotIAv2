import os
import shutil
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.utils import to_categorical
import threading
import time
import datetime  # para fechas y horas

# ===============================
# VARIABLES GLOBALES Y FLAGS
# ===============================
DEFAULT_POPULATION_SIZE = 10
DEFAULT_GENERATIONS = 5

pause_flag = threading.Event()   # Para pausar el algoritmo
finalize_flag = threading.Event()# Para finalizar la ejecución
ga_thread = None                 # Hilo del algoritmo genético
session_start_time = None        # Momento de inicio de la sesión
session_epoch_times = []         # Tiempos de cada epoch (acumulados durante la sesión)

# *** MEJORA 1 ***: set de emojis para IDs
EMOJIS_ID = [
    "🤖","👾","🧠","🚀","💡","🔬","🧬","📱","💻","🤔","🤝","🧑‍💻","👀","🔍","🔎","🏆",
    "⚙️","⏳","💾","🏗️","🌐","☁️","🚦","🕹️","🖥️","📡","🛰️","📈","📉","📝","⚡",
    "💥","🌟","💬","🗣️","📢","🪄","🏅","🏭","🪛","🧩","🤯","🏷️","🪫","🪪","🎛️",
    "🎚️","🔗","🔐","🔓"
]

# *** MEJORA 2 ***: diccionario para links a CSV
CONFIG_FILE = "configurations.csv"
SAVED_MODELS_FILE = "saved_models.csv"
SESSION_STATS_FILE = "session_stats.csv"
LOSS_HISTORY_FOLDER = "loss_history"  # carpeta donde guardaremos historiales de pérdida

# Columnas para configurations.csv
CONFIG_COLS = ["ID", "Generation", "Arquitectura", "Optimizador", 
               "Epochs", "Layer Sizes", "LR", "Batch"]

# ===============================
# FUNCIONES PARA CSV DE CONFIGURACIÓN
# ===============================
def load_configurations_csv():
    if os.path.exists(CONFIG_FILE):
        df = pd.read_csv(CONFIG_FILE)
    else:
        df = pd.DataFrame(columns=CONFIG_COLS)
    for col in CONFIG_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[CONFIG_COLS]

def save_configurations_csv(df):
    df.to_csv(CONFIG_FILE, index=False)

def add_or_update_configuration(config_dict):
    df = load_configurations_csv()
    row_df = pd.DataFrame([[config_dict.get(c, np.nan) for c in CONFIG_COLS]], 
                          columns=CONFIG_COLS)
    if config_dict["ID"] in df["ID"].values:
        mask = df["ID"] == config_dict["ID"]
        for col in CONFIG_COLS:
            df.loc[mask, col] = row_df.loc[0, col]
    else:
        df = pd.concat([df, row_df], ignore_index=True)
    save_configurations_csv(df)

def reassign_generations(pop_size=DEFAULT_POPULATION_SIZE):
    """
    Reordena 'Generation' según el orden de ID,
    de modo que cada pop_size individuos se incrementa la generación.
    """
    df = load_configurations_csv()
    df = df.sort_values(by=["Generation", "ID"]).reset_index(drop=True)
    new_generation = 1
    count = 0
    new_gens = []
    for i, row in df.iterrows():
        new_gens.append(new_generation)
        count += 1
        if count >= pop_size:
            new_generation += 1
            count = 0
    df["Generation"] = new_gens
    save_configurations_csv(df)

# ===============================
# FUNCIONES PARA CSV DE ESTADÍSTICAS DE SESIÓN
# ===============================
def load_session_stats_csv():
    if os.path.exists(SESSION_STATS_FILE):
        return pd.read_csv(SESSION_STATS_FILE)
    else:
        return pd.DataFrame(columns=["Start", "End", "Duration", "AvgEpochTime"])

def add_session_stat(start, end, duration, avg_epoch):
    df = load_session_stats_csv()
    new_entry = {"Start": start, 
                 "End": end, 
                 "Duration": duration, 
                 "AvgEpochTime": avg_epoch}
    new_df = pd.DataFrame([new_entry])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(SESSION_STATS_FILE, index=False)

# ===============================
# FUNCIONES PARA CSV DE MODELOS GUARDADOS
# ===============================
def load_saved_models_csv():
    if os.path.exists(SAVED_MODELS_FILE):
        return pd.read_csv(SAVED_MODELS_FILE)
    else:
        return pd.DataFrame(columns=["Modelo", "Score", "Accuracy", "Ruta"])

def save_saved_models_csv(df):
    df.to_csv(SAVED_MODELS_FILE, index=False)

def add_saved_model(model_name, score, accuracy, path):
    df = load_saved_models_csv()
    new_entry = {
        "Modelo": model_name,
        "Score": score,
        "Accuracy": accuracy,
        "Ruta": path
    }
    new_df = pd.DataFrame([new_entry])
    df = pd.concat([df, new_df], ignore_index=True)
    save_saved_models_csv(df)

# ===============================
# CARPETA MODELS Y LOSS_HISTORY
# ===============================
if not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists(LOSS_HISTORY_FOLDER):
    os.makedirs(LOSS_HISTORY_FOLDER)

# ===============================
# CARGA Y PREPROCESADO DE DATOS
# ===============================
data_path = r"C:\Users\formacionIA\Desktop\arbitraje 2\operaciones_ganadoras.csv"
df = pd.read_csv(data_path)
df = df.loc[:, ~df.columns.duplicated()]  # Eliminar columnas duplicadas

# Definir 200 columnas OHLC
ohlc_cols = []
for i in range(1, 51):
    ohlc_cols.extend([f'open_t-{i}', f'high_t-{i}', f'low_t-{i}', f'close_t-{i}'])

numeric_input_cols = ['precio_entrada', 'sl_price', 'tp_price'] + ohlc_cols
output_cols = ['precio_entrada', 'sl_price', 'tp_price']

df = df[['direccion'] + numeric_input_cols]

# Codificar "direccion"
label_encoder = LabelEncoder()
df['direccion'] = label_encoder.fit_transform(df['direccion'])

scaler_input = StandardScaler()
X_numeric = scaler_input.fit_transform(df[numeric_input_cols])
X_numeric = pd.DataFrame(X_numeric, columns=numeric_input_cols)
X_final = pd.concat([X_numeric, df[['direccion']].reset_index(drop=True)], axis=1)

scaler_prices = StandardScaler()
y_prices_scaled = scaler_prices.fit_transform(df[['sl_price', 'tp_price']])
y_prices_scaled = pd.DataFrame(y_prices_scaled, columns=['sl_price', 'tp_price'])

scaler_entrada = StandardScaler()
y_entrada_scaled = scaler_entrada.fit_transform(df[['precio_entrada']])

y_direccion = df['direccion']
y_direccion_cat = to_categorical(y_direccion, num_classes=2)

X_train, X_test, y_train_direccion, y_test_direccion, y_train_precios, y_test_precios, y_train_entrada, y_test_entrada = train_test_split(
    X_final, y_direccion, y_prices_scaled, y_entrada_scaled, test_size=0.3, random_state=42
)
y_train_direccion_cat = to_categorical(y_train_direccion, num_classes=2)
y_test_direccion_cat = to_categorical(y_test_direccion, num_classes=2)

print("\n🔧 Arquitectura del modelo:")
print(f"Entradas: {X_train.shape[1]} neuronas -> {list(X_train.columns)}")
print("Salidas: [dirección, precio_entrada, sl_price, tp_price]")

# ===============================
# HIPERPARÁMETROS
# ===============================
architectures = ["Dense", "LSTM", "GRU", "TCN", "Transformer"]
activation_functions = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'swish']
optimizers_dict = {'adam': optimizers.Adam, 
                   'sgd': optimizers.SGD, 
                   'rmsprop': optimizers.RMSprop}
batch_sizes = [8, 16,32]
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
epoch_range = (60, 120)

# ===============================
# CALLBACK TIEMPO MÁXIMO POR EPOCH
# ===============================
class TimeLimitCallback(keras.callbacks.Callback):
    def __init__(self, max_time):
        super().__init__()
        self.max_time = max_time
        self.start_time = None
        self.exceeded_time = False

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        session_epoch_times.append(elapsed)
        if elapsed > self.max_time:
            print(f"    Epoch {epoch+1} excedió el tiempo máximo de {self.max_time} s ({elapsed:.2f} s). Se descartará.")
            self.exceeded_time = True
            self.model.stop_training = True

# ===============================
# CALLBACK PARA GUARDAR HISTORIAL DE PÉRDIDA
# ===============================
class LossHistoryCallback(keras.callbacks.Callback):
    """
    Guarda el historial de la pérdida (por epoch) en un CSV,
    para luego poder re-dibujarlo.
    """
    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        mae_val = logs.get('precios_mae')
        if mae_val is None:
            mae_val = 0.0
        self.losses.append(mae_val)

    def on_train_end(self, logs=None):
        # Al terminar, guardamos en un CSV
        out_file = os.path.join(LOSS_HISTORY_FOLDER, f"{self.model_id}_loss.csv")
        df_losses = pd.DataFrame({"Epoch": range(1, len(self.losses)+1), "MAE": self.losses})
        df_losses.to_csv(out_file, index=False)

# ===============================
# CREACIÓN DEL MODELO MULTISALIDA
# ===============================
def create_model(layer_sizes, activation, optimizer_name, learning_rate, architecture):
    if architecture == "Dense":
        inputs = keras.Input(shape=(X_train.shape[1],))
        x = inputs
        for size in layer_sizes:
            x = layers.Dense(size, activation=activation)(x)

    elif architecture in ["LSTM", "GRU", "TCN", "Transformer"]:
        inputs = keras.Input(shape=(X_train.shape[1], 1))
        x = inputs
        if architecture == "LSTM":
            for size in layer_sizes:
                x = layers.LSTM(size, activation=activation, return_sequences=True)(x)
            x = layers.Attention()([x, x])
        elif architecture == "GRU":
            for size in layer_sizes:
                x = layers.GRU(size, activation=activation, return_sequences=True)(x)
            x = layers.Attention()([x, x])
        elif architecture == "TCN":
            for size in layer_sizes:
                x = layers.Conv1D(filters=size, kernel_size=2, activation=activation, padding='causal')(x)
            x = layers.Attention()([x, x])
        elif architecture == "Transformer":
            x = layers.Dense(64)(x)
            attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization()(x)
            for size in layer_sizes:
                x = layers.Dense(size, activation=activation)(x)
        x = layers.Flatten()(x)
    else:
        inputs = keras.Input(shape=(X_train.shape[1],))
        x = inputs
        for size in layer_sizes:
            x = layers.Dense(size, activation=activation)(x)
    
    # Salidas
    direccion_bin = layers.Dense(1, activation='sigmoid', name='direccion_bin')(x)
    direccion_cat = layers.Dense(2, activation='softmax', name='direccion_cat')(x)
    precios = layers.Dense(2, activation='linear', name='precios')(x)
    precio_entrada = layers.Dense(1, activation='linear', name='precio_entrada')(x)
    
    model = keras.Model(inputs=inputs, outputs=[direccion_bin, direccion_cat, precios, precio_entrada])
    optimizer = optimizers_dict[optimizer_name](learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss={
            'direccion_bin': 'binary_crossentropy',
            'direccion_cat': 'categorical_crossentropy',
            'precios': 'mse',
            'precio_entrada': 'mse'
        },
        metrics={
            'direccion_bin': 'accuracy',
            'precios': ['mse', 'mae'],
            'precio_entrada': 'mae'
        }
    )
    return model

# ===============================
# MOSTRAR HISTORIAL
# ===============================
def update_history_treeview(gen, legend_entry, app_data):
    tree_history = app_data["tree_history"]
    bullet = "●"
    tree_history.insert("", "end", values=(
        bullet,
        legend_entry["id"],
        legend_entry["metric"],
        f"{legend_entry['fitness']:.4f}",
        f"{legend_entry['accuracy']*100:.2f}%",
        legend_entry["epochs"],
        gen
    ), tags=(legend_entry["id"],))
    tree_history.tag_configure(legend_entry["id"], foreground=legend_entry["color"])

# ===============================
# ANIMACIÓN DE FLUJO DE DATOS CON BOLITAS
# ===============================
def animate_data_flow_with_balls(canvas, 
                                 input_nodes_numeric,
                                 input_node_categorical,
                                 hidden_nodes,
                                 tp_node,
                                 config,
                                 epoch_time=2.0):
    lr = config['lr']
    batch = config['batch_size']
    max_lr = 5e-3
    min_lr = 1e-4
    norm_lr = min(max((lr - min_lr) / (max_lr - min_lr), 0), 1)

    base_size = 4 + int(norm_lr * 10) + (batch // 8) * 2

    def get_ball_color(is_categorical=False):
        if is_categorical:
            return "#00c853"  
        r = norm_lr
        g = 0
        b = 1.0 - norm_lr
        return mcolors.to_hex((r, g, b))

    def launch_ball(path_coords, ball_color, ball_size):
        frames = 40
        step_time = epoch_time / frames
        x0, y0 = path_coords[0]
        ball_id = canvas.create_oval(x0-ball_size, y0-ball_size,
                                     x0+ball_size, y0+ball_size,
                                     fill=ball_color, outline="")

        total_segments = len(path_coords) - 1
        for seg_idx in range(total_segments):
            x_start, y_start = path_coords[seg_idx]
            x_end, y_end = path_coords[seg_idx+1]
            for f in range(frames // total_segments):
                if pause_flag.is_set() or finalize_flag.is_set():
                    canvas.delete(ball_id)
                    return
                t = f / float(frames // total_segments)
                cur_x = x_start + (x_end - x_start) * t
                cur_y = y_start + (y_end - y_start) * t
                dynamic_size = ball_size + int(2 * np.sin(np.pi * t))
                canvas.coords(ball_id,
                              cur_x - dynamic_size,
                              cur_y - dynamic_size,
                              cur_x + dynamic_size,
                              cur_y + dynamic_size)
                canvas.update()
                time.sleep(step_time)
        canvas.delete(ball_id)

    def center_of_node(nid):
        x1, y1, x2, y2 = canvas.coords(nid)
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    all_paths = []
    for in_nid in input_nodes_numeric:
        in_center = center_of_node(in_nid)
        color_in = get_ball_color(False)
        size_in = base_size
        for hid_nid in hidden_nodes:
            hid_center = center_of_node(hid_nid)
            path_coords = [in_center, hid_center, center_of_node(tp_node)]
            all_paths.append((path_coords, color_in, size_in))

    cat_center = center_of_node(input_node_categorical)
    color_cat = get_ball_color(True)
    size_cat = base_size + 2
    for hid_nid in hidden_nodes:
        hid_center = center_of_node(hid_nid)
        path_coords = [cat_center, hid_center, center_of_node(tp_node)]
        all_paths.append((path_coords, color_cat, size_cat))

    for idx, (coords, col, sz) in enumerate(all_paths):
        threading.Thread(target=launch_ball, args=(coords, col, sz), daemon=True).start()
        time.sleep(0.02)

def update_canvas_architecture_text(app_data, text):
    canvas = app_data["canvas"]
    if "arch_text_id" in canvas.data:
        canvas.delete(canvas.data["arch_text_id"])
    arch_text_id = canvas.create_text(
        500, 40, text=text, font=("Helvetica", 16, "bold"), fill="#ff5722"
    )
    canvas.data["arch_text_id"] = arch_text_id

# ===============================
# FUNCIÓN DE EVALUACIÓN DE INDIVIDUOS
# ===============================
def evaluate_individual(config, individual_id, axes, individual_index, total_individuales, max_epoch_time, app_data, generation):
    """
    Crea y entrena el modelo, calcula fitness, 
    y ejecuta la animación con bolitas en el diagrama.
    """
    arch_text = f"{config['architecture']} (LR={config['lr']}, Batch={config['batch_size']}, Epochs={config['epochs']})"
    update_canvas_architecture_text(app_data, arch_text)
    
    last_epoch_time = session_epoch_times[-1] if session_epoch_times else 2.0
    
    animate_data_flow_with_balls(
        canvas=app_data["canvas"],
        input_nodes_numeric=app_data["input_node_ids_numeric"],
        input_node_categorical=app_data["input_node_id_categorical"],
        hidden_nodes=app_data["hidden_node_ids"],
        tp_node=app_data["tp_node_id"],
        config=config,
        epoch_time=last_epoch_time
    )
    
    model = create_model(
        config['layer_sizes'],
        config['activation'],
        config['optimizer'],
        config['lr'],
        config['architecture']
    )
    
    if config['architecture'] in ["LSTM", "GRU", "TCN", "Transformer"]:
        X_train_mod = X_train.values.reshape(-1, X_train.shape[1], 1)
    else:
        X_train_mod = X_train
    
    cmap = plt.get_cmap("tab20")
    color_rgba = cmap(individual_index / total_individuales)
    color = mcolors.to_hex(color_rgba)
    epochs_ind = config['epochs']
    
    time_callback = TimeLimitCallback(max_epoch_time)

    model_id = f"Ind{individual_id:03}"
    loss_callback = LossHistoryCallback(model_id)

    history = model.fit(
        X_train_mod,
        {
            'direccion_bin': y_train_direccion,
            'direccion_cat': y_train_direccion_cat,
            'precios': y_train_precios,
            'precio_entrada': y_train_entrada
        },
        epochs=epochs_ind,
        batch_size=config['batch_size'],
        verbose=0,
        callbacks=[time_callback, loss_callback]
    )
    
    if time_callback.exceeded_time:
        print("    Modelo descartado por exceder tiempo máximo epoch.")
        return 1e6, 0, color, ""
    
    preds_train = model.predict(X_train_mod, verbose=0)
    precios_pred = scaler_prices.inverse_transform(preds_train[2])
    y_train_prices_orig = scaler_prices.inverse_transform(y_train_precios.values)

    sl_diff, tp_diff = [], []
    for i in range(len(y_train_prices_orig)):
        sl_diff.append(abs(precios_pred[i, 0] - y_train_prices_orig[i, 0]))
        tp_diff.append(abs(precios_pred[i, 1] - y_train_prices_orig[i, 1]))
    sl_avg = np.mean(sl_diff)
    tp_avg = np.mean(tp_diff)
    
    precio_entrada_pred = scaler_entrada.inverse_transform(preds_train[3])
    y_train_entrada_orig = scaler_entrada.inverse_transform(y_train_entrada)
    pe_diff = [abs(precio_entrada_pred[i, 0] - y_train_entrada_orig[i, 0]) 
               for i in range(len(y_train_entrada_orig))]
    pe_avg = np.mean(pe_diff)
    
    total_acc = sl_avg + tp_avg + pe_avg
    combined_fitness = total_acc

    final_accuracy = history.history.get('direccion_bin_accuracy', [0])[-1]

    output_accuracy_string = f"SL_AVG:{sl_avg:.0f} TP_AVG:{tp_avg:.0f} PE_AVG:{pe_avg:.0f} TOTAL:{total_acc:.0f}"
    print(f"📉 [{model_id}] Config: {config} - Score(Fitness): {combined_fitness:.4f}, Accuracy (DIR): {final_accuracy*100:.2f}%")

    filename = f"models/model_{model_id}_gen_{generation}.keras"
    model.save(filename)
    
    add_saved_model(
        model_name=os.path.basename(filename),
        score=f"{combined_fitness:.4f}",
        accuracy=f"{final_accuracy*100:.2f}%",
        path=filename
    )
    app_data["tree_saved_models"].insert("", "end", values=(
        os.path.basename(filename),
        f"{combined_fitness:.4f}",
        f"{final_accuracy*100:.2f}%",
        filename
    ))
    
    return combined_fitness, final_accuracy, color, output_accuracy_string

# ===============================
# FUNCIÓN ALEATORIA PARA CONFIG
# ===============================
def random_config(index=None):
    num_layers = random.randint(1, 5)
    layer_sizes = [random.randint(10, 128) for _ in range(num_layers)]
    arch = random.choice(architectures) if index is None else architectures[index % len(architectures)]
    return {
        'layer_sizes': layer_sizes,
        'activation': random.choice(activation_functions),
        'optimizer': random.choice(list(optimizers_dict.keys())),
        'lr': random.choice(learning_rates),
        'batch_size': random.choice(batch_sizes),
        'architecture': arch,
        'epochs': random.randint(epoch_range[0], epoch_range[1])
    }

# ===============================
# LOGICA DE ALGORITMO GENÉTICO DINÁMICO
# ===============================
def genetic_algorithm(population_size, generations, app_data):
    df_config = load_configurations_csv()
    max_generation = 0 if df_config.empty else df_config["Generation"].max()
    current_generation = max_generation + 1

    total_individuales = population_size * generations

    # Creamos population inicial (cada elemento es un dict con config + fitness dummy)
    # por ej: config + 'my_fitness': 999999
    population = []
    for i in range(population_size):
        cfg = random_config(i)
        cfg["my_fitness"] = 999999  # fitness inicial
        population.append(cfg)
    
    best_config = None
    best_fitness = float('inf')
    individual_id = 0 if df_config.empty else len(df_config)
    legend_data = []
    
    axes = {'mae': app_data["graf_ax"]}
    app_data["graf_ax"].clear()
    app_data["graf_ax"].set_title("Pérdida", fontsize=14, fontweight="bold")
    app_data["graf_ax"].set_xlabel("Epoch", fontsize=12)
    app_data["graf_ax"].set_ylabel("MAE", fontsize=12)

    start_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== Iniciando entrenamiento genético (fase) en {start_dt} ===")

    for gen in range(current_generation, current_generation + generations):
        if finalize_flag.is_set():
            print("Algoritmo finalizado por el usuario.")
            break
        
        app_data["progress_bar"]["value"] = int(((gen - current_generation + 1) / generations) * 100)
        app_data["lbl_status"].config(text=f"Ejecutando... Generación {gen}/{current_generation + generations - 1}")
        app_data["lbl_generation"].config(text=f"Generación: {gen}")
        
        app_data["root"].after(0, lambda: app_data["lbl_phase"].config(text="Fase: Entrenando Individuos"))

        print(f"\n=== 🌱 Generación {gen} ===")
        max_epoch_time = float(app_data["entry_epoch_time"].get())
        
        # Revisamos si CSV ya tiene pop_size individuos de gen => los parseamos
        df_config_gen = df_config[df_config["Generation"] == gen]
        if len(df_config_gen) < population_size:
            missing = population_size - len(df_config_gen)
            # generamos new_individuals
            new_individuals = []
            for _ in range(missing):
                cfg = random_config()
                cfg["my_fitness"] = 999999
                new_individuals.append(cfg)
            # unimos los que había en CSV + new
            # parsearlos a la forma population
            # En un ejemplo, simplemente:
            old_configs = df_config_gen.to_dict("records")
            # each record en CSV es { "ID":..., "Arquitectura":..., ...}
            # conv conv . parse... O si no, lo obviamos
            # Para demo:
            population = new_individuals
        else:
            # si ya hay pop_size en CSV, parseamos a population
            old_configs = df_config_gen.to_dict("records")
            population = []
            for rec in old_configs:
                pc = {
                    'layer_sizes': eval(rec['Layer Sizes']),
                    'activation': rec['Arquitectura'],   # OJO: adaptalo
                    'optimizer': rec['Optimizador'],
                    'lr': float(rec['LR']),
                    'batch_size': int(rec['Batch']),
                    'architecture': rec['Arquitectura'],
                    'epochs': int(rec['Epochs']),
                    'my_fitness': 999999
                }
                population.append(pc)

        # 1) Entrenamos cada uno de population
        for ind_cfg in population:
            if finalize_flag.is_set():
                break
            while pause_flag.is_set():
                time.sleep(0.5)
                if finalize_flag.is_set():
                    break
            if finalize_flag.is_set():
                break
            
            # Creamos un ID/emoji
            emoji = random.choice(EMOJIS_ID)
            current_id_str = f"{emoji} Ind {individual_id:03}"
            print(f"🧪 Entrenando {current_id_str} en Gen {gen}")
            
            # Entregamos ind_cfg a evaluate_individual
            # ajusta la firma para que reciba un dict
            fitness, acc, col, output_acc_str = evaluate_individual(
                ind_cfg,  # pasa su config
                individual_id,
                axes,
                individual_id,
                total_individuales,
                max_epoch_time,
                app_data,
                gen
            )

            # guardamos el fitness en 'my_fitness'
            ind_cfg["my_fitness"] = fitness

            # Creamos legend_entry
            legend_entry = {
                "id": current_id_str,
                "fitness": fitness,
                "accuracy": acc,
                "metric": output_acc_str,
                "epochs": ind_cfg['epochs'],
                "color": col,
                "generation": gen
            }
            legend_data.append(legend_entry)
            update_history_treeview(gen, legend_entry, app_data)
            
            # registramos en CSV
            config_entry = {
                "ID": current_id_str,
                "Generation": gen,
                "Arquitectura": ind_cfg['architecture'],
                "Optimizador": ind_cfg['optimizer'],
                "Epochs": ind_cfg['epochs'],
                "Layer Sizes": str(ind_cfg['layer_sizes']),
                "LR": ind_cfg['lr'],
                "Batch": ind_cfg['batch_size']
            }
            add_or_update_configuration(config_entry)
            reassign_generations(population_size)
            
            app_data["tree_configs"].insert("", "end", values=(
                current_id_str,
                ind_cfg['architecture'],
                ind_cfg['optimizer'],
                ind_cfg['epochs'],
                str(ind_cfg['layer_sizes']),
                ind_cfg['lr'],
                ind_cfg['batch_size']
            ))
            individual_id += 1
        
        if finalize_flag.is_set():
            break

        # Tomamos todos los que se entrenaron en esta gen
        # Sacamos el mejor
        gen_legend = [x for x in legend_data if x['generation'] == gen]
        if gen_legend:
            best_gen = min(gen_legend, key=lambda x: x['fitness'])
            if best_gen['fitness'] < best_fitness:
                best_fitness = best_gen['fitness']
                best_config = best_gen
            print(f"\n🔝 Mejor individuo de la Generación {gen}:")
            print(f"{best_gen['id']} - Score: {best_gen['fitness']:.4f}")
        
        # === AHORA HACEMOS LA FASE DE CRUCE Y MUTACIÓN REAL
        app_data["lbl_phase"].config(text="Fase: Cruce y Mutación")

        # 2) Cruzamos y mutamos => produce la poblacion para la siguiente gen
        population = crossover_and_mutation(population, population_size, mutation_rate=0.2)
        
        # *** Fin cruce/mutación
        app_data["lbl_phase"].config(text="Fase: Finalización de Generación")
        time.sleep(0.5)
    
    app_data["progress_bar"]["value"] = 100
    app_data["lbl_status"].config(text="Finalizado")

    if best_config is not None:
        # entrenar best_model
        df_confs = load_configurations_csv()
        row_best = df_confs[df_confs["ID"] == best_config["id"]]
        if not row_best.empty:
            arch = row_best["Arquitectura"].values[0]
            opt = row_best["Optimizador"].values[0]
            ep = int(row_best["Epochs"].values[0])
            lrs = float(row_best["LR"].values[0])
            bs = int(row_best["Batch"].values[0])
            layers_s = eval(row_best["Layer Sizes"].values[0])
            model_best = create_model(layers_s, 'relu', opt, lrs, arch)
            if arch in ["LSTM", "GRU", "TCN", "Transformer"]:
                X_train_mod = X_train.values.reshape(-1, X_train.shape[1], 1)
            else:
                X_train_mod = X_train
            model_best.fit(
                X_train_mod,
                {
                    'direccion_bin': y_train_direccion,
                    'direccion_cat': y_train_direccion_cat,
                    'precios': y_train_precios,
                    'precio_entrada': y_train_entrada
                },
                epochs=ep,
                batch_size=bs,
                verbose=0
            )

    end_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = time.time() - session_start_time if session_start_time else 0
    avg_epoch = sum(session_epoch_times)/len(session_epoch_times) if session_epoch_times else 0
    add_session_stat(start_dt, end_dt, duration, avg_epoch)


# ===============================
# EVALUAR EN TEST
# ===============================
def evaluate_test_set(model, X_test, y_test_direccion, y_test_precios, scaler_prices, label_encoder, architecture):
    if architecture in ["LSTM", "GRU", "TCN", "Transformer"]:
        X_test_mod = X_test.values.reshape(-1, X_test.shape[1], 1)
    else:
        X_test_mod = X_test.values
    
    preds = model.predict(X_test_mod, verbose=0)
    direccion_pred = (preds[0] > 0.5).astype(int).flatten()
    direction_accuracy = np.mean(direccion_pred == y_test_direccion.values) * 100
    
    precios_pred = scaler_prices.inverse_transform(preds[2])
    y_test_prices_orig = scaler_prices.inverse_transform(y_test_precios.values)
    sl_diff, tp_diff = [], []
    for i in range(len(y_test_prices_orig)):
        sl_diff.append(abs(precios_pred[i, 0] - y_test_prices_orig[i, 0]))
        tp_diff.append(abs(precios_pred[i, 1] - y_test_prices_orig[i, 1]))
    return direction_accuracy, np.mean(sl_diff), np.mean(tp_diff)

# ===============================
# FUNCIÓN DE PREDICCIÓN
# ===============================
def predecir(input_vals, best_model, best_config):
    if len(input_vals) != X_train.shape[1]:
        print(f"Error: se requieren {X_train.shape[1]} valores de entrada.")
        return
    input_scaled = scaler_input.transform([input_vals])[0]
    if best_config['architecture'] in ["LSTM", "GRU", "TCN", "Transformer"]:
        entrada_np = np.array(input_scaled).reshape(1, X_train.shape[1], 1)
    else:
        entrada_np = np.array(input_scaled).reshape(1, -1)
    
    preds = best_model.predict(entrada_np, verbose=0)
    direccion_bin_pred, direccion_cat_pred, precios_pred, precio_entrada_pred = preds
    direccion_pred = label_encoder.inverse_transform([int(round(direccion_bin_pred[0][0]))])[0]
    precios_orig = scaler_prices.inverse_transform(precios_pred)
    precio_entrada_orig = scaler_entrada.inverse_transform(precio_entrada_pred)
    
    print(f"\n🧠 Entrada: {input_vals}")
    print(f"📈 Predicción → Dirección: {direccion_pred}, "
          f"precio_entrada: {precio_entrada_orig[0][0]:.2f}, "
          f"sl_price: {precios_orig[0][0]:.2f}, "
          f"tp_price: {precios_orig[0][1]:.2f}")
    return direccion_pred, precio_entrada_orig[0][0], precios_orig[0][0], precios_orig[0][1]

# ===============================
# CREACIÓN DE LA APP TKINTER
# ===============================
def create_app():
    root = tk.Tk()
    root.title("Algoritmo Genético - Resumen y Control")
    root.geometry("1200x900")
    root.configure(bg="#eceff1")
    
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"), foreground="navy")
    style.configure("Treeview", font=("Helvetica", 9), rowheight=25)
    style.configure("TButton", font=("Helvetica", 10, "bold"))
    style.configure("TLabel", font=("Helvetica", 10))
    
    notebook = ttk.Notebook(root)

    # -------------------------------------------------
    # PESTAÑA: ARQUITECTURA
    # -------------------------------------------------
    frame_arch = ttk.Frame(notebook, padding=15)
    notebook.add(frame_arch, text="Arquitectura del Modelo")
    
    lbl_title_arch = ttk.Label(frame_arch, text="ESQUEMA DE LA RED NEURONAL (Diseño Superior + Rotado 90°)", 
                               font=("Helvetica", 16, "bold"), foreground="#1a237e")
    lbl_title_arch.pack(pady=10)

    border_frame = ttk.Frame(frame_arch, padding=10, borderwidth=2, relief="ridge")
    border_frame.pack(fill='both', expand=True)

    separator1 = ttk.Separator(border_frame, orient='horizontal')
    separator1.pack(fill='x', pady=5)
    
    info_frame = ttk.Frame(border_frame, padding=10, relief="sunken")
    info_frame.pack(fill='x', padx=10, pady=5)
    
    numeric_list = list(X_train.columns[:-1])
    direccion_str = X_train.columns[-1]
    lbl_input = ttk.Label(info_frame, text="Entradas:\n  - Numéricas: " + ", ".join(numeric_list) +
                                   f"\n  - Categórica: {direccion_str}", font=("Helvetica", 12))
    lbl_input.grid(row=0, column=0, sticky='w', padx=5, pady=5)
    
    lbl_output = ttk.Label(info_frame, text="Salidas:\n  [dirección, precio_entrada, sl_price, tp_price]", 
                           font=("Helvetica", 12))
    lbl_output.grid(row=0, column=1, sticky='w', padx=5, pady=5)
    
    separator2 = ttk.Separator(border_frame, orient='horizontal')
    separator2.pack(fill='x', pady=5)
    
    # Canvas
    canvas = tk.Canvas(border_frame, width=1150, height=600, bg="#fafafa", bd=4, relief="groove")
    canvas.pack(pady=10)
    canvas.data = {}
    
    # Ejemplo: dibujo de rectángulos / conexiones
    canvas.create_rectangle(50, 50, 1100, 250, outline="#b0bec5", width=3, dash=(4,2))
    canvas.create_text(80, 30, text="Diagrama Superior", anchor="w", font=("Helvetica", 12, "bold"), fill="#424242")

    canvas.create_rectangle(80, 80, 480, 140, fill="#bbdefb", outline="#1e88e5", width=3)
    canvas.create_text(280, 110, text="Input Numérico\n(203 columnas)", font=("Helvetica", 12, "bold"), fill="#0d47a1")
    canvas.create_rectangle(80, 160, 480, 220, fill="#fff9c4", outline="#fbc02d", width=3)
    canvas.create_text(280, 190, text="Input Categórico\n(dirección)", font=("Helvetica", 12), fill="#f57f17")
    
    canvas.create_line(480, 110, 540, 110, arrow=tk.LAST, width=3, fill="#424242")
    canvas.create_line(480, 190, 540, 190, arrow=tk.LAST, width=3, fill="#424242")
    canvas.create_line(820, 150, 880, 150, arrow=tk.LAST, width=3, fill="#424242")
    
    canvas.create_rectangle(540, 80, 940, 220, fill="#c8e6c9", outline="#43a047", width=3)
    canvas.create_text(740, 150, text="Output\n[dirección, precio_entrada, sl_price, tp_price]", 
                       font=("Helvetica", 12, "bold"), fill="#1b5e20")

    # Ejemplo de nodos "rotados"
    num_input_numeric = 5
    num_input_categorical = 1  
    hidden_nodes_count = 5
    output_nodes_count = 4

    x_start_input = 100
    y_start_input = 330
    spacing_input = 50
    
    input_node_ids_numeric = []
    for i in range(num_input_numeric):
        x = x_start_input
        y = y_start_input + i*spacing_input
        node_id = canvas.create_oval(x-15, y-15, x+15, y+15, fill="#bbdefb", outline="blue", width=2)
        input_node_ids_numeric.append(node_id)
        canvas.create_text(x, y, text=f"In Num {i+1}", font=("Helvetica", 8, "bold"))
    
    x_cat = x_start_input
    y_cat = y_start_input + num_input_numeric*spacing_input + 60
    input_node_id_categorical = canvas.create_oval(x_cat-15, y_cat-15, x_cat+15, y_cat+15, 
                                                   fill="#fff9c4", outline="#fbc02d", width=2)
    canvas.create_text(x_cat, y_cat, text="In Cat", font=("Helvetica", 8, "bold"))

    x_start_hidden = 500
    y_start_hidden = 330
    spacing_hidden = 60
    
    hidden_node_ids = []
    for i in range(hidden_nodes_count):
        x = x_start_hidden
        y = y_start_hidden + i*spacing_hidden
        node_id = canvas.create_oval(x-15, y-15, x+15, y+15, fill="#d1c4e9", outline="#5e35b1", width=2)
        hidden_node_ids.append(node_id)
        canvas.create_text(x, y, text=f"H {i+1}", font=("Helvetica", 8, "bold"))
    
    x_start_output = 900
    y_start_output = 330
    spacing_output = 70
    
    output_node_ids = []
    output_labels = ["Dirección", "P.Entrada", "SL", "TP"]
    for i in range(output_nodes_count):
        x = x_start_output
        y = y_start_output + i*spacing_output
        node_id = canvas.create_oval(x-20, y-20, x+20, y+20, fill="#c8e6c9", outline="#43a047", width=2)
        output_node_ids.append(node_id)
        out_lbl = output_labels[i]
        canvas.create_text(x, y, text=out_lbl, font=("Helvetica", 8, "bold"))
    
    for in_node in (input_node_ids_numeric + [input_node_id_categorical]):
        in_coords = canvas.coords(in_node)
        in_center = ((in_coords[0]+in_coords[2])/2, (in_coords[1]+in_coords[3])/2)
        for h_node in hidden_node_ids:
            h_coords = canvas.coords(h_node)
            h_center = ((h_coords[0]+h_coords[2])/2, (h_coords[1]+h_coords[3])/2)
            canvas.create_line(in_center[0]+15, in_center[1],
                               h_center[0]-15, h_center[1],
                               fill="#ccc", width=1)

    for h_node in hidden_node_ids:
        h_coords = canvas.coords(h_node)
        h_center = ((h_coords[0]+h_coords[2])/2, (h_coords[1]+h_coords[3])/2)
        for out_node in output_node_ids:
            o_coords = canvas.coords(out_node)
            o_center = ((o_coords[0]+o_coords[2])/2, (o_coords[1]+o_coords[3])/2)
            canvas.create_line(h_center[0]+15, h_center[1],
                               o_center[0]-20, o_center[1],
                               fill="#ccc", width=1)
    
    tp_node_id = output_node_ids[3]

    def export_diagram():
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if file_path:
            ps_file = file_path.replace(".png", ".ps")
            canvas.postscript(file=ps_file)
            try:
                from PIL import Image
                img = Image.open(ps_file)
                img.save(file_path, 'png')
                os.remove(ps_file)
                messagebox.showinfo("Exportar Diagrama", f"Diagrama exportado a {file_path}")
            except ImportError:
                messagebox.showerror("Error", "No se pudo convertir a PNG (falta Pillow). Se generó un .ps igualmente.")
    btn_export = ttk.Button(border_frame, text="Exportar Diagrama", command=export_diagram)
    btn_export.pack(pady=5)

    # -------------------------------------------------
    # PESTAÑA: HISTORIAL
    # -------------------------------------------------
    frame_summary = ttk.Frame(notebook, padding=10)
    notebook.add(frame_summary, text="Historial")

    lbl_title_sum = ttk.Label(frame_summary, text="Historial del Algoritmo", font=("Helvetica", 16, "bold"), 
                              foreground="darkgreen")
    lbl_title_sum.pack(pady=8)

    lbl_generation = ttk.Label(frame_summary, text="Generación: 0", font=("Helvetica", 14))
    lbl_generation.pack(pady=5)

    tree_history = ttk.Treeview(frame_summary, columns=("Color", "ID", "Accuracy outputs", 
                                                        "Fitness", "Accuracy", "Epochs", "Gen"), 
                                show="headings", height=8)
    tree_history.heading("Color", text="Color")
    tree_history.column("Color", width=60, anchor="center")
    tree_history.heading("ID", text="ID")
    tree_history.column("ID", width=80, anchor="center")
    tree_history.heading("Accuracy outputs", text="Accuracy outputs")
    tree_history.column("Accuracy outputs", width=220, anchor="center")
    tree_history.heading("Fitness", text="Fitness")
    tree_history.column("Fitness", width=80, anchor="center")
    tree_history.heading("Accuracy", text="Accuracy")
    tree_history.column("Accuracy", width=80, anchor="center")
    tree_history.heading("Epochs", text="Epochs")
    tree_history.column("Epochs", width=80, anchor="center")
    tree_history.heading("Gen", text="Gen")
    tree_history.column("Gen", width=80, anchor="center")
    tree_history.pack(fill="both", expand=True, padx=10, pady=10)

    # -------------------------------------------------
    # PESTAÑA: GRÁFICOS
    # -------------------------------------------------
    frame_graf = ttk.Frame(notebook, padding=10)
    notebook.add(frame_graf, text="Gráficos")
    
    fig, ax = plt.subplots(figsize=(6,4))
    canvas_fig = FigureCanvasTkAgg(fig, master=frame_graf)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack(fill="both", expand=True)

    # -------------------------------------------------
    # PESTAÑA: MODELOS GUARDADOS
    # -------------------------------------------------
    frame_modelos = ttk.Frame(notebook, padding=10)
    notebook.add(frame_modelos, text="Modelos Guardados")
    
    lbl_modelos = ttk.Label(frame_modelos, text="Modelos Guardados (Carpeta: models)", 
                            font=("Helvetica", 16, "bold"), foreground="purple")
    lbl_modelos.pack(pady=8)
    
    tree_saved_models = ttk.Treeview(frame_modelos, columns=("Modelo", "Score", "Accuracy", "Ruta"), 
                                     show="headings", height=8)
    for c in ("Modelo","Score","Accuracy","Ruta"):
        tree_saved_models.heading(c, text=c)
        tree_saved_models.column(c, width=100, anchor="center")
    tree_saved_models.pack(fill="both", expand=True, padx=10, pady=10)
    
    saved_models_df = load_saved_models_csv()
    for idx, row in saved_models_df.iterrows():
        tree_saved_models.insert("", "end", values=(
            row["Modelo"],
            row["Score"],
            row["Accuracy"],
            row["Ruta"]
        ))
    
    def descargar_modelo():
        selected = tree_saved_models.selection()
        if not selected:
            messagebox.showwarning("Atención", "Seleccione un modelo para descargar.")
            return
        item = tree_saved_models.item(selected[0])
        ruta = item["values"][3]
        modelo = item["values"][0]
        destino = filedialog.asksaveasfilename(
            initialfile=modelo, 
            defaultextension=".keras", 
            filetypes=[("Modelo Keras", "*.keras"), ("Todos", "*.*")]
        )
        if destino:
            shutil.copyfile(ruta, destino)
            messagebox.showinfo("Descarga", f"Modelo guardado en: {destino}")
    
    def borrar_modelo():
        selected = tree_saved_models.selection()
        if not selected:
            messagebox.showwarning("Atención", "Seleccione un modelo para borrar.")
            return
        item = tree_saved_models.item(selected[0])
        ruta = item["values"][3]
        modelo = item["values"][0]
        if messagebox.askyesno("Confirmar", f"¿Borrar el modelo {modelo}?"):
            try:
                if os.path.exists(ruta):
                    os.remove(ruta)
                tree_saved_models.delete(selected[0])
                df_sm = load_saved_models_csv()
                df_sm = df_sm[df_sm["Ruta"] != ruta].copy()
                save_saved_models_csv(df_sm)
                messagebox.showinfo("Borrar", f"Modelo {modelo} borrado correctamente.")
            except Exception as e:
                messagebox.showerror("Error", f"Error al borrar el modelo: {e}")

    frame_model_buttons = ttk.Frame(frame_modelos)
    frame_model_buttons.pack(pady=5)
    btn_descargar = ttk.Button(frame_model_buttons, text="Descargar Modelo", command=descargar_modelo)
    btn_descargar.grid(row=0, column=0, padx=5)
    btn_borrar = ttk.Button(frame_model_buttons, text="Borrar Modelo", command=borrar_modelo)
    btn_borrar.grid(row=0, column=1, padx=5)

    # -------------------------------------------------
    # PESTAÑA: CONFIGURACIONES
    # -------------------------------------------------
    frame_config = ttk.Frame(notebook, padding=10)
    notebook.add(frame_config, text="Configuraciones")
    
    lbl_config = ttk.Label(frame_config, text="Configuraciones de Modelos Evaluados", 
                           font=("Helvetica", 16, "bold"), foreground="brown")
    lbl_config.pack(pady=8)
    
    tree_configs = ttk.Treeview(frame_config, columns=("ID", "Arquitectura", "Optimizador", 
                                                       "Epochs", "Layer Sizes", "LR", "Batch"),
                                show="headings", height=8)
    for col in ("ID", "Arquitectura", "Optimizador", "Epochs", "Layer Sizes", "LR", "Batch"):
        tree_configs.heading(col, text=col)
        tree_configs.column(col, width=110, anchor="center")
    tree_configs.pack(fill="both", expand=True, padx=10, pady=10)
    
    df_conf_load = load_configurations_csv()
    for idx, row in df_conf_load.iterrows():
        tree_configs.insert("", "end", values=(
            row["ID"],
            row["Arquitectura"],
            row["Optimizador"],
            row["Epochs"],
            row["Layer Sizes"],
            row["LR"],
            row["Batch"]
        ))

    # -------------------------------------------------
    # PESTAÑA: CONTROL
    # -------------------------------------------------
    frame_control = ttk.Frame(notebook, padding=10)
    notebook.add(frame_control, text="Control")
    
    lbl_status = ttk.Label(frame_control, text="Estado: Esperando inicio", font=("Helvetica", 16))
    lbl_status.pack(pady=5)
    
    progress_bar = ttk.Progressbar(frame_control, orient="horizontal", mode="determinate", maximum=100)
    progress_bar.pack(fill="x", padx=10, pady=5)
    
    lbl_phase = ttk.Label(frame_control, text="Fase: N/A", font=("Helvetica", 12, "italic"), foreground="darkred")
    lbl_phase.pack(pady=5)
    
    frame_time = ttk.Frame(frame_control)
    frame_time.pack(pady=5)
    lbl_time = ttk.Label(frame_time, text="Tiempo máximo por epoch (s):", font=("Helvetica", 10))
    lbl_time.pack(side="left")
    entry_epoch_time = ttk.Entry(frame_time, width=5)
    entry_epoch_time.insert(0, "3")
    entry_epoch_time.pack(side="left", padx=5)
    
    def toggle_pause():
        if pause_flag.is_set():
            pause_flag.clear()
            btn_pause.config(text="Pausar")
            lbl_status.config(text="Estado: Ejecutando")
        else:
            pause_flag.set()
            btn_pause.config(text="Continuar")
            lbl_status.config(text="Estado: Pausado")
    
    btn_pause = ttk.Button(frame_control, text="Pausar", command=toggle_pause)
    btn_pause.pack(pady=5)
    
    lbl_session_timer = ttk.Label(frame_control, text="Tiempo de sesión: 0 s", font=("Helvetica", 12))
    lbl_session_timer.pack(pady=5)
    
    session_stats_tree = ttk.Treeview(frame_control, columns=("Start", "End", "Duration", "AvgEpochTime"), 
                                      show="headings", height=5)
    for col in ("Start", "End", "Duration", "AvgEpochTime"):
        session_stats_tree.heading(col, text=col)
        session_stats_tree.column(col, width=120, anchor="center")
    session_stats_tree.pack(fill="x", padx=10, pady=5)
    
    df_stats_load = load_session_stats_csv()
    for idx, row in df_stats_load.iterrows():
        session_stats_tree.insert("", "end", values=(
            row["Start"],
            row["End"],
            f"{row['Duration']:.2f}",
            f"{row['AvgEpochTime']:.2f}"
        ))
    
    lbl_log = ttk.Label(frame_control, text="Logs de Ejecución:", font=("Helvetica", 12))
    lbl_log.pack(pady=5)
    txt_log = tk.Text(frame_control, height=5)
    txt_log.pack(fill="x", padx=10, pady=5)
    
    timer_id = None
    def update_timer():
        nonlocal timer_id
        if session_start_time:
            elapsed = time.time() - session_start_time
            lbl_session_timer.config(text=f"Tiempo de sesión: {int(elapsed)} s")
        timer_id = root.after(1000, update_timer)
    
    timer_id = root.after(1000, update_timer)
    
    def finalizar():
        finalize_flag.set()
        session_end = time.time()
        duration = session_end - session_start_time if session_start_time else 0
        avg_epoch = sum(session_epoch_times)/len(session_epoch_times) if session_epoch_times else 0
        dt_str_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dt_str_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") if not session_start_time else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session_start_time))
        
        add_session_stat(dt_str_start, dt_str_end, duration, avg_epoch)
        
        session_stats_tree.insert("", "end", values=(
            dt_str_start,
            dt_str_end,
            f"{duration:.2f}",
            f"{avg_epoch:.2f}"
        ))
        lbl_status.config(text="Estado: Finalizado")
    
    def start_algorithm():
        global ga_thread, session_start_time, session_epoch_times
        if ga_thread is None or not ga_thread.is_alive():
            session_start_time = time.time()
            session_epoch_times.clear()
            finalize_flag.clear()
            pause_flag.clear()
            lbl_status.config(text="Estado: Ejecutando")
            ga_thread = threading.Thread(
                target=genetic_algorithm, 
                args=(DEFAULT_POPULATION_SIZE, DEFAULT_GENERATIONS, app_data),
                daemon=True
            )
            ga_thread.start()
        else:
            messagebox.showinfo("Información", "El algoritmo ya está en ejecución.")
    
    btn_start = ttk.Button(frame_control, text="Iniciar", command=start_algorithm)
    btn_start.pack(pady=5)
    
    btn_finalizar = ttk.Button(frame_control, text="Finalizar", command=finalizar)
    btn_finalizar.pack(pady=5)
    
    def on_closing():
        finalizar()
        if timer_id is not None:
            root.after_cancel(timer_id)
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Creamos el diccionario app_data con todo lo necesario
    app_data = {
        "root": root,
        "lbl_generation": lbl_generation,   # <-- La etiqueta de generación
        "tree_history": tree_history,
        "canvas": canvas,
        "graf_fig": fig,
        "graf_ax": ax,
        "canvas_fig": canvas_fig,
        "tree_saved_models": tree_saved_models,
        "tree_configs": tree_configs,
        "progress_bar": progress_bar,
        "lbl_status": lbl_status,
        "lbl_phase": lbl_phase,
        "entry_epoch_time": entry_epoch_time,
        "timer_id": timer_id,

        # Nodos para la animación
        "input_node_ids_numeric": input_node_ids_numeric,
        "input_node_id_categorical": input_node_id_categorical,
        "hidden_node_ids": hidden_node_ids,
        "tp_node_id": tp_node_id
    }
    
    notebook.pack(fill="both", expand=True)
    return app_data

# ===============================
# INICIAR LA APP
# ===============================
app = create_app()
app["root"].after(100, lambda: print("La interfaz se ha mostrado. Pulsa 'Iniciar' en la pestaña Control para comenzar."))
app["root"].mainloop()

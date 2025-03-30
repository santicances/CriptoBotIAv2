import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Ejemplo de label encoder para la columna "dirección"
le = LabelEncoder()
le.fit(["Corta", "Larga"])  # Ajusta según tus datos

# ──────────────────────────────────────────────────────────────
# 1. Configuración y parámetros de la simulación
# ──────────────────────────────────────────────────────────────
class SimulationConfig:
    def __init__(self):
        self.initial_capital = 1000.0
        self.commission = 0.001
        self.wait_time = 0.5         # segundos entre operaciones
        self.target_trades = 100
        self.start_time = "2020-03-23 09:00:00"
        self.entry_volume = 0.001    # volumen de entrada
        # Ruta del CSV con datos reales de BTC
        self.data_path = r"C:\Users\formacionIA\Desktop\arbitraje 2\BTC_data.csv"
        # Diccionario de modelos
        self.model_paths = {
            "Model_013_gen_3": r"C:\Users\formacionIA\Desktop\arbitraje 2\models\model_Ind013_gen_3.keras",
            "Model_015_gen_4": r"C:\Users\formacionIA\Desktop\arbitraje 2\models\model_Ind015_gen_4.keras",
            "Model_037_gen_5": r"C:\Users\formacionIA\Desktop\arbitraje 2\models\model_Ind037_gen_5.keras"
        }

# ──────────────────────────────────────────────────────────────
# 2. Estrategia de Trading (MLStrategy)
# Se ha añadido un parámetro stop_flag y update_win_ratio para la actualización en vivo.
# ──────────────────────────────────────────────────────────────
class MLStrategy(bt.Strategy):
    params = dict(
        target_trades=100,
        wait_time=0.5,
        entry_volume=0.001,
        trade_max_bars=3,  # máximo número de barras permitidas en una operación
        model=None,        # modelo cargado
        model_name="",     # nombre del modelo para log
        stop_flag=None,    # flag para detener la simulación
        update_progress=None,
        update_log=None,
        update_plot=None,
        update_open=None,
        update_closed=None,
        update_win_ratio=None  # callback para actualizar win ratio en vivo
    )

    def __init__(self):
        self.open_trades = []
        self.closed_trades = []
        self.current_trade_record = None

        self.in_trade = False
        self.trade_entry_price = None
        self.trade_SL = None
        self.trade_TP = None
        self.trade_direction = None
        self.current_trade_count = 0
        self.start_cash = self.broker.getvalue()
        self.equity_curve = []
        self.trade_bar_count = 0

    def next(self):
        # Verificar si se ha solicitado detener la simulación
        if self.p.stop_flag and self.p.stop_flag["stop"]:
            self.env.runstop()
            return

        dt = self.data.datetime.datetime(0)
        current_equity = self.broker.getvalue()
        self.equity_curve.append((dt, current_equity))

        if self.p.update_log:
            self.p.update_log(f"Barra {len(self)} - Tiempo: {dt} - Equity: {current_equity:.2f}\n")

        if len(self) < 50:
            time.sleep(self.p.wait_time)
            return

        if not self.in_trade:
            # Construir vector de 204 características para el modelo:
            raw_features = []
            last_price = self.data.close[0]
            raw_features.extend([last_price, last_price, last_price])
            for i in range(1, 51):
                raw_features.extend([self.data.open[-i], self.data.high[-i], self.data.low[-i], self.data.close[-i]])
            raw_features.append(0)
            raw_features = np.array(raw_features).reshape(1, -1)
            inp = raw_features.reshape(1, -1, 1)

            prediction = self.p.model.predict(inp, verbose=0)
            if isinstance(prediction, list) and len(prediction) == 4:
                pred_bin, pred_cat, pred_prices, pred_entry = prediction
            else:
                if self.p.update_log:
                    self.p.update_log(f"Error: modelo {self.p.model_name} no devuelve 4 salidas.\n")
                raise ValueError(f"Error: modelo {self.p.model_name} no devuelve 4 salidas.")

            pred_dir = le.inverse_transform([np.argmax(pred_cat[0])])[0]
            pred_sl, pred_tp = pred_prices[0][0], pred_prices[0][1]
            entry_price = pred_entry[0][0]

            self.trade_entry_price = entry_price
            self.trade_SL = pred_sl
            self.trade_TP = pred_tp
            self.trade_direction = "Larga" if pred_dir == "Larga" else "Corta"

            self.in_trade = True
            self.trade_bar_count = 0

            self.current_trade_record = {
                "Entry Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "Direction": self.trade_direction,
                "Entry Price": f"{entry_price:.2f}"
            }
            self.open_trades.append(self.current_trade_record)
            if self.p.update_open:
                self.p.update_open(self.open_trades)
            if self.p.update_log:
                self.p.update_log(f"Iniciando operación ({self.p.model_name}): Dirección {self.trade_direction}, Entrada {entry_price:.2f}, SL {pred_sl:.2f}, TP {pred_tp:.2f}\n")
            if self.trade_direction == "Larga":
                self.buy(size=self.p.entry_volume)
            else:
                self.sell(size=self.p.entry_volume)
        else:
            self.trade_bar_count += 1
            if self.trade_direction == "Larga":
                if self.data.low[0] <= self.trade_SL:
                    self.close()
                    pnl = self.trade_SL - self.trade_entry_price
                    if self.current_trade_record:
                        self.current_trade_record["Exit Time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                        self.current_trade_record["Exit Price"] = f"{self.trade_SL:.2f}"
                        self.current_trade_record["PnL"] = f"{pnl:.2f}"
                        self.closed_trades.append(self.current_trade_record)
                        if self.current_trade_record in self.open_trades:
                            self.open_trades.remove(self.current_trade_record)
                        if self.p.update_closed:
                            self.p.update_closed(self.closed_trades)
                        if self.p.update_open:
                            self.p.update_open(self.open_trades)
                    if self.p.update_log:
                        self.p.update_log(f"SL alcanzado ({self.p.model_name}) a {self.trade_SL:.2f}\n")
                    self.in_trade = False
                    self.current_trade_count += 1
                    self.trade_bar_count = 0
                elif self.data.high[0] >= self.trade_TP:
                    self.close()
                    pnl = self.trade_TP - self.trade_entry_price
                    if self.current_trade_record:
                        self.current_trade_record["Exit Time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                        self.current_trade_record["Exit Price"] = f"{self.trade_TP:.2f}"
                        self.current_trade_record["PnL"] = f"{pnl:.2f}"
                        self.closed_trades.append(self.current_trade_record)
                        if self.current_trade_record in self.open_trades:
                            self.open_trades.remove(self.current_trade_record)
                        if self.p.update_closed:
                            self.p.update_closed(self.closed_trades)
                        if self.p.update_open:
                            self.p.update_open(self.open_trades)
                    if self.p.update_log:
                        self.p.update_log(f"TP alcanzado ({self.p.model_name}) a {self.trade_TP:.2f}\n")
                    self.in_trade = False
                    self.current_trade_count += 1
                    self.trade_bar_count = 0
                elif self.trade_bar_count > self.p.trade_max_bars:
                    self.close()
                    exit_price = self.data.close[0]
                    pnl = exit_price - self.trade_entry_price
                    if self.current_trade_record:
                        self.current_trade_record["Exit Time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                        self.current_trade_record["Exit Price"] = f"{exit_price:.2f}"
                        self.current_trade_record["PnL"] = f"{pnl:.2f} (Forzado)"
                        self.closed_trades.append(self.current_trade_record)
                        if self.current_trade_record in self.open_trades:
                            self.open_trades.remove(self.current_trade_record)
                        if self.p.update_closed:
                            self.p.update_closed(self.closed_trades)
                        if self.p.update_open:
                            self.p.update_open(self.open_trades)
                    if self.p.update_log:
                        self.p.update_log(f"Cierre forzado ({self.p.model_name}) a {exit_price:.2f} tras {self.trade_bar_count} barras\n")
                    self.in_trade = False
                    self.current_trade_count += 1
                    self.trade_bar_count = 0
            else:  # Caso para posición Corta
                if self.data.high[0] >= self.trade_SL:
                    self.close()
                    pnl = self.trade_entry_price - self.trade_SL
                    if self.current_trade_record:
                        self.current_trade_record["Exit Time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                        self.current_trade_record["Exit Price"] = f"{self.trade_SL:.2f}"
                        self.current_trade_record["PnL"] = f"{pnl:.2f}"
                        self.closed_trades.append(self.current_trade_record)
                        if self.current_trade_record in self.open_trades:
                            self.open_trades.remove(self.current_trade_record)
                        if self.p.update_closed:
                            self.p.update_closed(self.closed_trades)
                        if self.p.update_open:
                            self.p.update_open(self.open_trades)
                    if self.p.update_log:
                        self.p.update_log(f"SL alcanzado en Corta ({self.p.model_name}) a {self.trade_SL:.2f}\n")
                    self.in_trade = False
                    self.current_trade_count += 1
                    self.trade_bar_count = 0
                elif self.data.low[0] <= self.trade_TP:
                    self.close()
                    pnl = self.trade_entry_price - self.trade_TP
                    if self.current_trade_record:
                        self.current_trade_record["Exit Time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                        self.current_trade_record["Exit Price"] = f"{self.trade_TP:.2f}"
                        self.current_trade_record["PnL"] = f"{pnl:.2f}"
                        self.closed_trades.append(self.current_trade_record)
                        if self.current_trade_record in self.open_trades:
                            self.open_trades.remove(self.current_trade_record)
                        if self.p.update_closed:
                            self.p.update_closed(self.closed_trades)
                        if self.p.update_open:
                            self.p.update_open(self.open_trades)
                    if self.p.update_log:
                        self.p.update_log(f"TP alcanzado en Corta ({self.p.model_name}) a {self.trade_TP:.2f}\n")
                    self.in_trade = False
                    self.current_trade_count += 1
                    self.trade_bar_count = 0
                elif self.trade_bar_count > self.p.trade_max_bars:
                    self.close()
                    exit_price = self.data.close[0]
                    pnl = self.trade_entry_price - exit_price
                    if self.current_trade_record:
                        self.current_trade_record["Exit Time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                        self.current_trade_record["Exit Price"] = f"{exit_price:.2f}"
                        self.current_trade_record["PnL"] = f"{pnl:.2f} (Forzado)"
                        self.closed_trades.append(self.current_trade_record)
                        if self.current_trade_record in self.open_trades:
                            self.open_trades.remove(self.current_trade_record)
                        if self.p.update_closed:
                            self.p.update_closed(self.closed_trades)
                        if self.p.update_open:
                            self.p.update_open(self.open_trades)
                    if self.p.update_log:
                        self.p.update_log(f"Cierre forzado en Corta ({self.p.model_name}) a {exit_price:.2f} tras {self.trade_bar_count} barras\n")
                    self.in_trade = False
                    self.current_trade_count += 1
                    self.trade_bar_count = 0

        # Actualizar el progreso
        if self.p.update_progress:
            progress = (self.current_trade_count / self.p.target_trades) * 100
            self.p.update_progress(progress)
        # Actualizar la curva de equity
        if self.p.update_plot:
            self.p.update_plot(self.equity_curve)
        # Calcular y actualizar el ratio de acierto en vivo
        if self.p.update_win_ratio:
            total = len(self.closed_trades)
            wins = 0
            for trade in self.closed_trades:
                try:
                    pnl = float(trade.get("PnL", "0").split()[0])
                    if pnl > 0:
                        wins += 1
                except:
                    pass
            ratio = (wins / total * 100) if total > 0 else 0
            self.p.update_win_ratio(ratio)
        time.sleep(self.p.wait_time)

# ──────────────────────────────────────────────────────────────
# 3. SimulationRunner: ejecuta la estrategia y calcula estadísticas.
# Se utiliza un diccionario stop_flag para indicar al backtest que debe detenerse.
# ──────────────────────────────────────────────────────────────
class SimulationRunner(threading.Thread):
    def __init__(self, config, model, model_name,
                 update_progress_callback, update_log_callback, update_plot_callback,
                 update_open_callback, update_closed_callback, update_win_ratio_callback,
                 on_simulation_complete):
        super().__init__()
        self.config = config
        self.model = model
        self.model_name = model_name
        self.update_progress = update_progress_callback
        self.update_log = update_log_callback
        self.update_plot = update_plot_callback
        self.update_open = update_open_callback
        self.update_closed = update_closed_callback
        self.update_win_ratio = update_win_ratio_callback
        self.on_simulation_complete = on_simulation_complete
        self._stop_event = threading.Event()
        self.stop_flag = {"stop": False}
        self.equity_curve_result = []
        self.winrate = 0.0
        self.summary = {}

    def stop(self):
        self._stop_event.set()
        self.stop_flag["stop"] = True

    def run(self):
        try:
            self.start_time = datetime.now()  # Registro de inicio
            self.update_log(f"Iniciando simulación para {self.model_name}...\n")
            self.update_log("Cargando datos BTC desde CSV...\n")
            df = self.get_btc_data_from_csv()
            self.update_log(f"Datos cargados: {len(df)} filas\n")

            cerebro = bt.Cerebro()
            cerebro.broker.setcash(self.config.initial_capital)
            cerebro.broker.setcommission(commission=self.config.commission)
            data_feed = bt.feeds.PandasData(dataname=df)
            cerebro.adddata(data_feed)

            cerebro.addstrategy(
                MLStrategy,
                target_trades=self.config.target_trades,
                wait_time=self.config.wait_time,
                entry_volume=self.config.entry_volume,
                model=self.model,
                model_name=self.model_name,
                stop_flag=self.stop_flag,
                update_progress=self.update_progress,
                update_log=self.update_log,
                update_plot=self.update_plot,
                update_open=self.update_open,
                update_closed=self.update_closed,
                update_win_ratio=self.update_win_ratio
            )
            self.update_log("Ejecutando backtest...\n")
            results = cerebro.run()
            final_capital = cerebro.broker.getvalue()
            self.update_log(f"Capital Final para {self.model_name}: {final_capital:.2f}\n")

            strat = results[0][0]
            self.equity_curve_result = strat.equity_curve
            closed_trades = strat.closed_trades
            total = len(closed_trades)
            wins = 0
            for trade in closed_trades:
                try:
                    pnl = float(trade.get("PnL", "0").split()[0])
                    if pnl > 0:
                        wins += 1
                except:
                    pass
            self.winrate = (wins / total * 100) if total > 0 else 0
            self.end_time = datetime.now()  # Registro de fin

            # Cálculo de métricas adicionales
            initial_capital = self.config.initial_capital
            final_capital = cerebro.broker.getvalue()
            profit = final_capital - initial_capital
            equity_values = [v for (_, v) in strat.equity_curve]
            peak = -float("inf")
            dd = 0
            for val in equity_values:
                if val > peak:
                    peak = val
                current_dd = peak - val
                if current_dd > dd:
                    dd = current_dd
            drawdown = dd
            hit_ratio = self.winrate
            final_profitability = (profit / initial_capital) * 100 if initial_capital != 0 else 0
            num_operations = total

            self.summary = {
                "Modelo": self.model_name,
                "Capital Inicial": initial_capital,
                "Capital Final": final_capital,
                "Drawdown": drawdown,
                "Beneficio": profit,
                "Ratio Acierto": hit_ratio,
                "Rentabilidad Final": final_profitability,
                "Inicio": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Fin": self.end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Operaciones": num_operations
            }
            self.on_simulation_complete()
        except Exception as e:
            self.update_log(f"Error en simulación para {self.model_name}: {str(e)}\n")

    def get_btc_data_from_csv(self):
        df = pd.read_csv(self.config.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df

# ──────────────────────────────────────────────────────────────
# 4. Interfaz gráfica con Tkinter (ventana con pestañas y mejoras UI)
# ──────────────────────────────────────────────────────────────
class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulación de Trading - Arbitraje")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")

        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 12, 'bold'), padding=6)
        style.configure('TLabel', font=('Helvetica', 12), padding=4)
        style.configure('TEntry', font=('Helvetica', 12), padding=4)
        style.configure('Treeview.Heading', font=('Helvetica', 12, 'bold'))
        style.configure('Treeview', font=('Helvetica', 11), rowheight=25)

        # Barra de estado
        self.status_var = tk.StringVar(value="Listo")
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, font=('Helvetica', 10))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Temporizador
        self.timer_var = tk.StringVar(value="00:00:00")
        self.timer_label = tk.Label(root, textvariable=self.timer_var, font=('Helvetica', 10), bd=1, relief=tk.SUNKEN, anchor=tk.E)
        self.timer_label.pack(side=tk.BOTTOM, fill=tk.X)

        header = tk.Label(root, text="Simulación de Trading - Arbitraje", font=("Helvetica", 18, "bold"), fg="#003366", bg="#f0f0f0")
        header.pack(pady=10)

        # Frame para configuración
        config_frame = tk.Frame(root, bd=2, relief=tk.GROOVE, padx=10, pady=10, bg="#e6e6e6")
        config_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(config_frame, text="Capital Inicial:", bg="#e6e6e6").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.capital_entry = tk.Entry(config_frame, font=('Helvetica', 12))
        self.capital_entry.grid(row=0, column=1, padx=5, pady=5)
        self.capital_entry.insert(0, "10000")

        tk.Label(config_frame, text="Comisión:", bg="#e6e6e6").grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.commission_entry = tk.Entry(config_frame, font=('Helvetica', 12))
        self.commission_entry.grid(row=0, column=3, padx=5, pady=5)
        self.commission_entry.insert(0, "0.000")

        tk.Label(config_frame, text="Tiempo de espera (s):", bg="#e6e6e6").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.wait_entry = tk.Entry(config_frame, font=('Helvetica', 12))
        self.wait_entry.grid(row=1, column=1, padx=5, pady=5)
        self.wait_entry.insert(0, "0.01")

        tk.Label(config_frame, text="Target de Operaciones:", bg="#e6e6e6").grid(row=1, column=2, sticky="e", padx=5, pady=5)
        self.target_entry = tk.Entry(config_frame, font=('Helvetica', 12))
        self.target_entry.grid(row=1, column=3, padx=5, pady=5)
        self.target_entry.insert(0, "1000")

        tk.Label(config_frame, text="Volumen de Entrada:", bg="#e6e6e6").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.volume_entry = tk.Entry(config_frame, font=('Helvetica', 12))
        self.volume_entry.grid(row=2, column=1, padx=5, pady=5)
        self.volume_entry.insert(0, "0.001")

        tk.Label(config_frame, text="Modelo a simular:", bg="#e6e6e6").grid(row=2, column=2, sticky="e", padx=5, pady=5)
        self.model_selector = ttk.Combobox(config_frame, font=('Helvetica', 12), state="readonly")
        self.model_selector['values'] = list(SimulationConfig().model_paths.keys())
        self.model_selector.current(0)
        self.model_selector.grid(row=2, column=3, padx=5, pady=5)

        # Etiqueta para Win Ratio en vivo
        tk.Label(config_frame, text="Ratio Acierto:", bg="#e6e6e6").grid(row=3, column=4, sticky="e", padx=5, pady=5)
        self.win_ratio_var = tk.StringVar(value="0.00%")
        self.win_ratio_label = tk.Label(config_frame, textvariable=self.win_ratio_var, font=('Helvetica', 12), bg="#e6e6e6")
        self.win_ratio_label.grid(row=3, column=5, padx=5, pady=5)

        # Botones de control
        self.btn_start = ttk.Button(config_frame, text="Iniciar Simulación", command=self.start_simulation)
        self.btn_start.grid(row=4, column=0, padx=5, pady=10)
        self.btn_stop = ttk.Button(config_frame, text="Detener Simulación", command=self.stop_simulation, state="disabled")
        self.btn_stop.grid(row=4, column=1, padx=5, pady=10)
        self.btn_reset = ttk.Button(config_frame, text="Reiniciar", command=self.reset_simulation, state="disabled")
        self.btn_reset.grid(row=4, column=2, padx=5, pady=10)
        # Nota: Los tooltips se pueden agregar utilizando bibliotecas adicionales si se desea.

        self.progress = ttk.Progressbar(config_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=6, padx=5, pady=10)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Pestaña: Gráfico de Evolución
        self.tab_graph = tk.Frame(self.notebook)
        self.notebook.add(self.tab_graph, text="Gráfico")
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_graph)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Pestaña: Operaciones Abiertas
        self.tab_open = tk.Frame(self.notebook)
        self.notebook.add(self.tab_open, text="Operaciones Abiertas")
        self.open_trades_tree = ttk.Treeview(self.tab_open, columns=("Entry Time", "Direction", "Entry Price"), show="headings")
        self.open_trades_tree.heading("Entry Time", text="Entry Time")
        self.open_trades_tree.heading("Direction", text="Direction")
        self.open_trades_tree.heading("Entry Price", text="Entry Price")
        self.open_trades_tree.column("Entry Time", width=140, anchor="center")
        self.open_trades_tree.column("Direction", width=100, anchor="center")
        self.open_trades_tree.column("Entry Price", width=100, anchor="center")
        self.open_trades_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Pestaña: Operaciones Finalizadas
        self.tab_closed = tk.Frame(self.notebook)
        self.notebook.add(self.tab_closed, text="Operaciones Finalizadas")
        self.closed_trades_tree = ttk.Treeview(self.tab_closed, 
                                               columns=("Entry Time", "Direction", "Entry Price", "Exit Time", "Exit Price", "PnL"),
                                               show="headings")
        for col in ("Entry Time", "Direction", "Entry Price", "Exit Time", "Exit Price", "PnL"):
            self.closed_trades_tree.heading(col, text=col)
            self.closed_trades_tree.column(col, width=120, anchor="center")
        self.closed_trades_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Pestaña: Log de Eventos
        self.tab_log = tk.Frame(self.notebook)
        self.notebook.add(self.tab_log, text="Log de Eventos")
        self.log_text = tk.Text(self.tab_log, height=10, font=('Helvetica', 12))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Pestaña: Resumen de Simulaciones
        self.tab_summary = tk.Frame(self.notebook)
        self.notebook.add(self.tab_summary, text="Resumen de Simulaciones")
        self.summary_tree = ttk.Treeview(self.tab_summary, 
                                         columns=("Modelo", "Capital Inicial", "Capital Final", "Drawdown", "Beneficio", 
                                                  "Ratio Acierto", "Rentabilidad Final", "Inicio", "Fin", "Operaciones"),
                                         show="headings")
        columnas = ["Modelo", "Capital Inicial", "Capital Final", "Drawdown", "Beneficio", 
                    "Ratio Acierto", "Rentabilidad Final", "Inicio", "Fin", "Operaciones"]
        for col in columnas:
            self.summary_tree.heading(col, text=col)
            self.summary_tree.column(col, width=110, anchor="center")
        self.summary_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.simulation_config = SimulationConfig()
        self.simulation_runner = None
        self.simulation_start_time = None
        self.timer_running = False

    def update_progress(self, value):
        self.progress['value'] = value
        self.root.update_idletasks()

    def update_log(self, message):
        self.log_text.insert(tk.END, message)
        self.log_text.see(tk.END)

    def update_plot(self, equity_curve):
        self.ax.cla()
        if equity_curve:
            dates, equity = zip(*equity_curve)
            self.ax.plot(dates, equity, label="Equity Curve", linewidth=2)
            self.ax.set_xlabel("Fecha", fontsize=12)
            self.ax.set_ylabel("Valor del Portafolio", fontsize=12)
            self.ax.set_title("Evolución del Equity", fontsize=14, fontweight="bold")
            self.ax.legend()
            self.ax.grid(True)
        self.canvas.draw()

    def update_open_trades(self, open_trades):
        for item in self.open_trades_tree.get_children():
            self.open_trades_tree.delete(item)
        for trade in open_trades:
            self.open_trades_tree.insert("", "end", values=(
                trade.get("Entry Time", ""),
                trade.get("Direction", ""),
                trade.get("Entry Price", "")
            ))

    def update_closed_trades(self, closed_trades):
        for item in self.closed_trades_tree.get_children():
            self.closed_trades_tree.delete(item)
        for trade in closed_trades:
            self.closed_trades_tree.insert("", "end", values=(
                trade.get("Entry Time", ""),
                trade.get("Direction", ""),
                trade.get("Entry Price", ""),
                trade.get("Exit Time", ""),
                trade.get("Exit Price", ""),
                trade.get("PnL", "")
            ))

    def update_win_ratio(self, ratio):
        # Actualiza el label de win ratio en tiempo real
        self.win_ratio_var.set(f"{ratio:.2f}%")

    def disable_inputs(self):
        # Deshabilitar campos y botones durante la simulación
        self.capital_entry.config(state="disabled")
        self.commission_entry.config(state="disabled")
        self.wait_entry.config(state="disabled")
        self.target_entry.config(state="disabled")
        self.volume_entry.config(state="disabled")
        self.model_selector.config(state="disabled")
        self.btn_start.config(state="disabled")
        self.btn_reset.config(state="disabled")

    def enable_inputs(self):
        # Rehabilitar campos y botones al finalizar/detener la simulación
        self.capital_entry.config(state="normal")
        self.commission_entry.config(state="normal")
        self.wait_entry.config(state="normal")
        self.target_entry.config(state="normal")
        self.volume_entry.config(state="normal")
        self.model_selector.config(state="readonly")
        self.btn_start.config(state="normal")
        self.btn_reset.config(state="normal")

    def start_timer(self):
        self.simulation_start_time = datetime.now()
        self.timer_running = True
        self.update_timer()

    def update_timer(self):
        if self.timer_running:
            elapsed = datetime.now() - self.simulation_start_time
            self.timer_var.set(str(elapsed).split('.')[0])
            self.root.after(1000, self.update_timer)

    def stop_timer(self):
        self.timer_running = False

    def start_simulation(self):
        try:
            initial_capital = float(self.capital_entry.get())
            commission = float(self.commission_entry.get())
            wait_time = float(self.wait_entry.get())
            target_trades = int(self.target_entry.get())
            entry_volume = float(self.volume_entry.get())
        except ValueError:
            self.update_log("Error: Verifica que los valores ingresados sean numéricos.\n")
            return

        # Actualizar configuración y deshabilitar entradas
        self.simulation_config.initial_capital = initial_capital
        self.simulation_config.commission = commission
        self.simulation_config.wait_time = wait_time
        self.simulation_config.target_trades = target_trades
        self.simulation_config.entry_volume = entry_volume
        self.disable_inputs()
        self.btn_stop.config(state="normal")
        self.btn_reset.config(state="disabled")
        self.status_var.set("Simulación en ejecución...")
        self.start_timer()

        selected_model = self.model_selector.get()
        model_path = self.simulation_config.model_paths.get(selected_model, None)
        if model_path:
            model = load_model(model_path, compile=False)
        else:
            model = None

        self.simulation_runner = SimulationRunner(
            config=self.simulation_config,
            model=model,
            model_name=selected_model,
            update_progress_callback=self.update_progress,
            update_log_callback=self.update_log,
            update_plot_callback=self.update_plot,
            update_open_callback=self.update_open_trades,
            update_closed_callback=self.update_closed_trades,
            update_win_ratio_callback=self.update_win_ratio,
            on_simulation_complete=self.on_simulation_complete
        )
        self.simulation_runner.start()

    def check_simulation_runner(self):
        # Método de polling para verificar si el thread finalizó sin bloquear el UI
        if self.simulation_runner.is_alive():
            self.root.after(500, self.check_simulation_runner)
        else:
            self.on_simulation_complete()
            self.enable_inputs()

    def stop_simulation(self):
        if self.simulation_runner:
            if messagebox.askyesno("Confirmar", "¿Desea detener la simulación?"):
                self.simulation_runner.stop()
                self.update_log("Simulación detenida por el usuario.\n")
                self.btn_stop.config(state="disabled")
                self.status_var.set("Simulación detenida")
                self.stop_timer()
                # Usar polling para esperar que el thread finalice sin bloquear
                self.root.after(500, self.check_simulation_runner)

    def reset_simulation(self):
        self.progress['value'] = 0
        self.log_text.delete(1.0, tk.END)
        self.ax.cla()
        self.canvas.draw()
        self.open_trades_tree.delete(*self.open_trades_tree.get_children())
        self.closed_trades_tree.delete(*self.closed_trades_tree.get_children())
        self.update_log("Simulación reiniciada.\n")
        self.btn_start.config(state="normal")
        self.btn_reset.config(state="disabled")
        self.status_var.set("Listo")
        self.timer_var.set("00:00:00")
        self.win_ratio_var.set("0.00%")

    def on_simulation_complete(self):
        self.update_log("Simulación completada.\n")
        self.btn_stop.config(state="disabled")
        self.btn_reset.config(state="normal")
        self.status_var.set("Simulación finalizada")
        self.stop_timer()
        # Guardar resumen en la pestaña de Resumen de Simulaciones
        if hasattr(self.simulation_runner, "summary"):
            summary = self.simulation_runner.summary
            self.summary_tree.insert("", "end", values=(
                summary["Modelo"],
                f"{summary['Capital Inicial']:.2f}",
                f"{summary['Capital Final']:.2f}",
                f"{summary['Drawdown']:.2f}",
                f"{summary['Beneficio']:.2f}",
                f"{summary['Ratio Acierto']:.2f}%",
                f"{summary['Rentabilidad Final']:.2f}%",
                summary["Inicio"],
                summary["Fin"],
                summary["Operaciones"]
            ))
        self.enable_inputs()

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationApp(root)
    root.mainloop()

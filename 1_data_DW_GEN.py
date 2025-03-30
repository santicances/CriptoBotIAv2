import backtrader as bt
import pandas as pd
import numpy as np
import random
import os
import ccxt
from tqdm import tqdm

# Configuración global
random.seed(42)
np.random.seed(42)
NUM_SIMULACIONES = 20
NUM_VELAS = 8760
CAPITAL_INICIAL = 100000
CSV_NAME = 'BTC_data.csv'

def fetch_btc_data_ccxt():
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=NUM_VELAS)
    df = pd.DataFrame(ohlcv, columns=['datetime','open','high','low','close','volume'])
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df

def get_btc_data():
    if os.path.exists(CSV_NAME) and os.path.getsize(CSV_NAME) > 1024:
        df = pd.read_csv(CSV_NAME, index_col=0, parse_dates=True)
        return df.iloc[-NUM_VELAS:]
    df = fetch_btc_data_ccxt()
    df.to_csv(CSV_NAME)
    return df

class RandomStrategy(bt.Strategy):
    params = (('simulacion_id', 1),)

    def __init__(self):
        self.trade_count = 0
        self.winning_trades = 0
        self.results = []
        self.simulacion_id = self.p.simulacion_id
        self.current_params = {}

    def next(self):
        if not self.position:
            price = self.data.close[0]
            sl_pct = random.uniform(0.0075, 0.035)
            tp_pct = random.uniform(0.0075, 0.035)
            if random.random() < 0.5:
                self.buy_bracket(exectype=bt.Order.Market,
                                 limitprice=price*(1+tp_pct),
                                 stopprice=price*(1-sl_pct))
                direction = 'long'
            else:
                self.sell_bracket(exectype=bt.Order.Market,
                                  limitprice=price*(1-tp_pct),
                                  stopprice=price*(1+sl_pct))
                direction = 'short'
            self.current_params = {'sl': sl_pct, 'tp': tp_pct, 'direction': direction}

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_count += 1
            pnl = trade.pnlcomm
            idx = len(self.data) - 1
            available = idx if idx > 0 else 0

            if available > 0:
                mean_open  = np.mean(self.data.open.get(size=available))
                mean_high  = np.mean(self.data.high.get(size=available))
                mean_low   = np.mean(self.data.low.get(size=available))
                mean_close = np.mean(self.data.close.get(size=available))
            else:
                mean_open = mean_high = mean_low = mean_close = self.data.close[0]

            record = {
                'simulacion': self.simulacion_id,
                'operacion': self.trade_count,
                'direccion': 'Larga' if trade.long else 'Corta',
                'precio_entrada': round(trade.price,2),
                'precio_salida': round(trade.value,2),
                'sl_pct': round(self.current_params['sl']*100,2),
                'tp_pct': round(self.current_params['tp']*100,2),
                'sl_price': round(trade.price*(1-self.current_params['sl']) if trade.long else trade.price*(1+self.current_params['sl']),2),
                'tp_price': round(trade.price*(1+self.current_params['tp']) if trade.long else trade.price*(1-self.current_params['tp']),2),
                'pnl': round(pnl,2),
                'duracion': (trade.close_datetime() - trade.open_datetime()).total_seconds()//60,
                'resultado': 'Ganadora' if pnl>0 else 'Perdedora'
            }

            for i in range(1, 51):
                if idx-i >= 0:
                    record[f'open_t-{i}']  = round(self.data.open[-i],2)
                    record[f'high_t-{i}']  = round(self.data.high[-i],2)
                    record[f'low_t-{i}']   = round(self.data.low[-i],2)
                    record[f'close_t-{i}'] = round(self.data.close[-i],2)
                else:
                    record[f'open_t-{i}']  = round(mean_open,2)
                    record[f'high_t-{i}']  = round(mean_high,2)
                    record[f'low_t-{i}']   = round(mean_low,2)
                    record[f'close_t-{i}'] = round(mean_close,2)

            if pnl > 0:
                self.winning_trades += 1
            self.results.append(record)

    def stop(self):
        self.final_value = self.broker.getvalue()

def run_simulations():
    data = get_btc_data()
    resumen_list = []
    operaciones = []

    for sim in tqdm(range(NUM_SIMULACIONES), desc="Simulaciones"):
        cerebro = bt.Cerebro()
        cerebro.addstrategy(RandomStrategy, simulacion_id=sim+1)
        cerebro.adddata(bt.feeds.PandasData(dataname=data))
        cerebro.broker.setcash(CAPITAL_INICIAL)
        cerebro.broker.setcommission(commission=0.001)
        strat = cerebro.run()[0]

        resumen_list.append({
            'simulacion': sim+1,
            'operaciones_totales': strat.trade_count,
            'operaciones_ganadoras': strat.winning_trades,
            'ratio_acierto': round(strat.winning_trades/strat.trade_count*100,2) if strat.trade_count else 0,
            'pnl_total': round(strat.final_value - CAPITAL_INICIAL,2),
            'capital_final': round(strat.final_value,2)
        })

        operaciones.extend(strat.results)

    df_operaciones = pd.DataFrame(operaciones)
    df_ganadoras = df_operaciones[df_operaciones['resultado']=='Ganadora']
    return pd.DataFrame(resumen_list), df_ganadoras

if __name__ == '__main__':
    resumen, ganadoras = run_simulations()
    #resumen.to_csv('resumen_simulaciones.csv', index=False)
    ganadoras.to_csv('operaciones_ganadoras.csv', index=False)

    print(resumen)
    print("\nEjemplo de ganadoras:")
    print(ganadoras.sample(5))

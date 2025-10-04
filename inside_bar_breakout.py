import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from collections import deque

# --- MOCK/MIGRATED INDICATOR FUNCTIONS ---

def calculate_atr(df, period=14):
    """Calculates ATR (True Range used for stop trailing)."""
    df['TR'] = np.maximum.reduce([
        df['High'] - df['Low'],
        np.abs(df['High'] - df['Close'].shift(1)),
        np.abs(df['Low'] - df['Close'].shift(1))
    ])
    df['ATR'] = df['TR'].rolling(period, min_periods=period).mean()
    return df['ATR']

def calculate_ema(df, period=20):
    """Calculates EMA (Used for MTF trend check)."""
    return df['Close'].ewm(span=period, adjust=False).mean()

# --- TRADE MANAGEMENT CLASSES ---

class Trade:
    """Represents an active or closed trade position."""
    def __init__(self, symbol, entry_time, entry_price, direction, sl, tp, risk):
        self.symbol = symbol
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.sl = sl
        self.tp = tp
        self.initial_risk = risk
        self.pnl = 0.0
        self.status = 'OPEN'
        self.close_time = None
        self.close_price = None

    def close(self, price, time):
        self.close_price = price
        self.close_time = time
        if self.direction == 'LONG':
            self.pnl = price - self.entry_price
        else: # SHORT
            self.pnl = self.entry_price - price
        self.status = 'CLOSED'

# --- CORE STRATEGY CLASS ---

class InsideBarBreakoutStrategy:
    """
    MTF Backtesting Engine for the Inside Bar Breakout Strategy.
    """
    def __init__(self, data_path, entry_timeframe='15T', max_ibr_lookback=5, risk_reward=1.5):
        self.data_path = data_path
        self.entry_tf = entry_timeframe
        self.max_ibr_lookback = max_ibr_lookback
        self.risk_reward = risk_reward
        self.tick_points = 2  # 2 Ticks / Points buffer for entry
        self.safety_buffer = 5  # 5 Ticks / Points buffer for SL beyond MB
        self.higher_timeframes = ['60T', '2H', '1D', '1W']
        self.ohlcv_data = {}  # Stores all resampled data: {symbol: {timeframe: df}}
        self.trades = []
        self.open_trade = None

    def _load_and_resample_data(self, file_path):
        """Loads 1m data and resamples it to all required timeframes."""
        
        # 1. Load 1-minute data
        df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.sort_index().replace([np.inf, -np.inf], np.nan).dropna()
        
        if df.empty: return {}
        
        all_timeframes = {self.entry_tf: None}
        for tf in self.higher_timeframes:
            all_timeframes[tf] = None

        resampled_data = {}
        
        # 2. Resample logic
        for tf in all_timeframes:
            resampled_df = df.resample(tf).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Calculate indicators (EMA(20) and ATR(14) for all relevant TFs)
            if not resampled_df.empty:
                resampled_df['EMA'] = calculate_ema(resampled_df, period=20)
                resampled_df['ATR'] = calculate_atr(resampled_df, period=14)
                resampled_data[tf] = resampled_df
                
        return resampled_data

    def load_all_market_data(self):
        """Finds all CSV files and loads/resamples data for all symbols."""
        print(f"Loading data from: {self.data_path}")
        
        csv_files = glob.glob(os.path.join(self.data_path, '*_minute.csv'))
        
        for file_path in csv_files:
            symbol = os.path.basename(file_path).split('_minute')[0]
            print(f"Processing symbol: {symbol}")
            
            data = self._load_and_resample_data(file_path)
            if data:
                self.ohlcv_data[symbol] = data
            
        print(f"Loaded data for {len(self.ohlcv_data)} symbols.")

    def is_inside_bar(self, current_bar, mother_bar):
        """Checks if the current bar is fully contained within the mother bar."""
        return (current_bar['High'] < mother_bar['High']) and \
               (current_bar['Low'] > mother_bar['Low'])

    def check_mtf_trend(self, symbol, current_time, direction):
        """
        Checks MTF trend alignment for a given time using the most recent HTF bar.
        Returns: True if aligned, False otherwise.
        """
        aligned_count = 0
        
        for tf in self.higher_timeframes:
            df_htf = self.ohlcv_data[symbol][tf]
            
            # Get the most recently closed HTF bar *before* the current_time
            htf_bar = df_htf.loc[df_htf.index < current_time].iloc[-1]
            prev_htf_bar = df_htf.loc[df_htf.index < current_time].iloc[-2]
            
            current_close = htf_bar['Close']
            current_ema = htf_bar['EMA']
            prev_ema = prev_htf_bar['EMA']
            
            # Trend Check: Close vs EMA AND EMA slope
            is_up = current_close > current_ema and current_ema > prev_ema
            is_down = current_close < current_ema and current_ema < prev_ema
            
            if (direction == 'LONG' and is_up) or (direction == 'SHORT' and is_down):
                aligned_count += 1
                
        # Must align with at least one higher timeframe
        return aligned_count > 0

    def find_inside_bar_breakout(self, df_bars, current_index):
        """
        Identifies the Mother Bar and the Inside Bar Range (IBR) cluster ending at current_index.
        Returns: (is_valid, MB_Bar, IBR_High, IBR_Low, IBR_cluster_start_index)
        """
        if current_index < 1: return False, None, None, None, None

        # Look back up to max_ibr_lookback + 1 (for MB)
        lookback_limit = max(0, current_index - self.max_ibr_lookback - 1)
        
        # Current bar is the first potential IB
        # Find the start of the IB cluster (i.e., the first bar not inside the *previous* bar)
        ib_start_index = current_index
        
        # IBR search loop
        while ib_start_index > lookback_limit:
            current_bar = df_bars.iloc[ib_start_index]
            prev_bar = df_bars.iloc[ib_start_index - 1]
            
            if self.is_inside_bar(current_bar, prev_bar):
                ib_start_index -= 1
            else:
                # MB is the bar immediately preceding the first IB found
                MB_index = ib_start_index - 1
                MB_bar = df_bars.iloc[MB_index]
                
                # The IBR cluster consists of bars from ib_start_index to current_index
                IBR_cluster = df_bars.iloc[ib_start_index : current_index + 1]
                
                # Check if all bars in the IBR cluster are inside the MB
                if not all(self.is_inside_bar(bar, MB_bar) for _, bar in IBR_cluster.iterrows()):
                    # This should ideally not happen if logic is correct, but handles edge cases
                    return False, None, None, None, None
                
                # 1. Volume Validation (Contraction Filter)
                MB_volume = MB_bar['Volume']
                IBR_max_volume = IBR_cluster['Volume'].max()
                
                # Volume must be lower during consolidation
                if IBR_max_volume >= MB_volume:
                    return False, None, None, None, None
                
                # 2. Define IBR range
                IBR_High = IBR_cluster['High'].max()
                IBR_Low = IBR_cluster['Low'].min()
                
                return True, MB_bar, IBR_High, IBR_Low, ib_start_index

        return False, None, None, None, None

    def _manage_open_trade(self, current_bar, current_time):
        """Checks if the open trade hits SL or TP."""
        trade = self.open_trade
        if not trade: return
        
        # Check for Close
        
        # 1. Check for Stop Loss (SL) Hit
        if (trade.direction == 'LONG' and current_bar['Low'] <= trade.sl) or \
           (trade.direction == 'SHORT' and current_bar['High'] >= trade.sl):
            
            # SL hit logic: Assuming price hits SL before TP/Reversal within the bar
            close_price = trade.sl
            trade.close(close_price, current_time)
            self.trades.append(trade)
            self.open_trade = None
            return

        # 2. Check for Take Profit (TP) Hit
        if (trade.direction == 'LONG' and current_bar['High'] >= trade.tp) or \
           (trade.direction == 'SHORT' and current_bar['Low'] <= trade.tp):
           
            # TP hit logic: Assuming fill at TP
            close_price = trade.tp
            trade.close(close_price, current_time)
            self.trades.append(trade)
            self.open_trade = None
            return

        # 3. Trailing Stop Logic (Simplified: Move to BE at 1R, then trail 2*ATR)
        r_distance = trade.initial_risk
        
        if trade.status == 'OPEN':
            
            # Check 1R BE move
            if trade.direction == 'LONG' and current_bar['Close'] >= trade.entry_price + r_distance:
                if trade.sl < trade.entry_price: # Move to BE
                    trade.sl = trade.entry_price
            elif trade.direction == 'SHORT' and current_bar['Close'] <= trade.entry_price - r_distance:
                if trade.sl > trade.entry_price: # Move to BE
                    trade.sl = trade.entry_price
            
            # Check ATR trailing stop (only after enough bars for ATR calculation)
            if current_bar['ATR'] > 0:
                atr_stop_level = 2 * current_bar['ATR']
                
                if trade.direction == 'LONG':
                    new_sl = current_bar['Close'] - atr_stop_level
                    trade.sl = max(trade.sl, new_sl) # Only trail up
                else: # SHORT
                    new_sl = current_bar['Close'] + atr_stop_level
                    trade.sl = min(trade.sl, new_sl) # Only trail down

    def run_backtest(self):
        """
        Executes the backtest logic bar-by-bar for all loaded symbols.
        """
        if not self.ohlcv_data:
            print("Error: No market data loaded. Run load_all_market_data() first.")
            return

        for symbol, data in self.ohlcv_data.items():
            df_et = data.get(self.entry_tf)
            if df_et is None: continue
            
            print(f"\n--- Starting Backtest for {symbol} on {self.entry_tf} ---")
            
            # Iterate through the entry timeframe bars (starting after min data needed for indicators)
            min_bars = 20 # Need at least 20 bars for EMA
            
            for i in range(min_bars, len(df_et)):
                current_time = df_et.index[i]
                current_bar = df_et.iloc[i]
                
                # --- Part B: Trade Management ---
                if self.open_trade:
                    self._manage_open_trade(current_bar, current_time)
                    continue # Skip new entry search while a trade is open

                # --- Part A: Pattern Detection & Order Placement ---
                
                # Look back up to the current bar (index i) to find the IBR pattern
                is_valid, mb_bar, ibr_high, ibr_low, ib_start_index = \
                    self.find_inside_bar_breakout(df_et.iloc[:i], i - 1) # Check pattern on last closed bar
                
                if not is_valid: continue

                # 1. Calculate parameters
                risk_price_diff = mb_bar['High'] - mb_bar['Low']
                
                # --- LONG SETUP ---
                long_entry = ibr_high + self.tick_points
                long_sl = mb_bar['Low'] - self.safety_buffer
                long_risk = long_entry - long_sl 
                long_tp = long_entry + (long_risk * self.risk_reward)
                
                # --- SHORT SETUP ---
                short_entry = ibr_low - self.tick_points
                short_sl = mb_bar['High'] + self.safety_buffer
                short_risk = short_sl - short_entry 
                short_tp = short_entry - (short_risk * self.risk_reward)
                
                # 2. MTF and Entry Validation
                next_bar = df_et.iloc[i] # The bar where the breakout occurs
                
                # --- LONG BREAKOUT CHECK ---
                if next_bar['High'] >= long_entry:
                    if self.check_mtf_trend(symbol, current_time, 'LONG'):
                        # Simulating volume spike confirmation check is complex but assumed valid upon entry
                        self.open_trade = Trade(symbol, current_time, long_entry, 'LONG', long_sl, long_tp, long_risk)
                        # The next loop iteration will manage this trade
                        
                # --- SHORT BREAKOUT CHECK ---
                if next_bar['Low'] <= short_entry:
                    if self.check_mtf_trend(symbol, current_time, 'SHORT'):
                        # Simulating volume spike confirmation check is complex but assumed valid upon entry
                        self.open_trade = Trade(symbol, current_time, short_entry, 'SHORT', short_sl, short_tp, short_risk)
                        # The next loop iteration will manage this trade

    def generate_report(self):
        """Generates a consolidated performance report."""
        total_pnl = sum(t.pnl for t in self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades else 0
        
        print("\n" + "="*50)
        print("         INSIDE BAR BREAKOUT BACKTEST REPORT")
        print("="*50)
        print(f"Entry Timeframe: {self.entry_tf}")
        print(f"Symbols Tested: {len(self.ohlcv_data)}")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate * 100:.2f}%")
        print(f"Total PnL (Units): {total_pnl:.2f}")
        print("="*50)
        
        if self.open_trade:
            print(f"NOTE: 1 trade remains open on {self.open_trade.symbol}.")

# --- EXECUTION ---

if __name__ == '__main__':
    # Configuration - Change this path to your folder
    DATA_FOLDER = r"D:\py_code_workspace\NSE _STOCK _DATA" 
    
    # Initialize the strategy and backtester
    backtester = InsideBarBreakoutStrategy(
        data_path=DATA_FOLDER,
        entry_timeframe='15T',  # Example: 15-minute entry
        risk_reward=1.5
    )
    
    # Load all data files and create required timeframes
    backtester.load_all_market_data()
    
    # Run the bar-by-bar backtest
    backtester.run_backtest()
    
    # Generate the performance summary
    backtester.generate_report()

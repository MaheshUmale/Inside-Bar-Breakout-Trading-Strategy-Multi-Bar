import pandas as pd
import numpy as np
import glob
import os
import concurrent.futures
import threading
from typing import List
import math

# Assume calculate_ema and calculate_atr are defined elsewhere
# Example placeholder functions (replace with your actual implementations):
def calculate_ema(df, period):
    return df['Close'].ewm(span=period, adjust=False).mean()

def calculate_atr(df, period):
    # Simplified ATR calculation (replace with your preferred method)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


class Trade:
    """Represents an active or closed trade position."""
    def __init__(self, symbol, entry_time, entry_price, direction, sl, tp, initial_risk, size):
        self.symbol = symbol
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.sl = sl
        self.tp = tp
        self.initial_risk = initial_risk # Store initial risk in points
        self.size = size # Store the trade size
        self.pnl = 0.0
        self.status = 'OPEN'
        self.close_time = None
        self.close_price = None

    def close(self, price, time):
        self.close_price = price
        self.close_time = time
        # Calculate PnL in currency based on trade size
        if self.direction == 'LONG':
            self.pnl = (price - self.entry_price) * self.size
        else: # SHORT
            self.pnl = (self.entry_price - price) * self.size
        self.status = 'CLOSED'


class InsideBarBreakoutStrategy:
    """
    MTF Backtesting Engine for the Inside Bar Breakout Strategy.
    Dynamically adjusts Higher Timeframes (HTF) based on the Entry Timeframe (ET).
    """

    # Define a fixed hierarchy of supported timeframes using standard Pandas aliases
    SUPPORTED_TFS = ['3min', '5min', '15min', '30min', '1h', '2h', '1D', '1W'] # Updated aliases

    def __init__(self, data_path, entry_timeframe: str = '5min', max_ibr_lookback=5, risk_reward=1.5, max_workers=None, initial_capital=1000000, risk_per_trade_percent=0.02):
        self.data_path = data_path

        if entry_timeframe not in self.SUPPORTED_TFS:
             raise ValueError(f"Entry timeframe '{entry_timeframe}' is not supported. Choose from: {self.SUPPORTED_TFS}")

        self.entry_tf = entry_timeframe
        self.max_ibr_lookback = max_ibr_lookback
        self.risk_reward = risk_reward
        self.tick_points = 2  # 2 Ticks / Points buffer for entry
        self.safety_buffer = 5  # 5 Ticks / Points buffer for SL beyond MB
        self.debug_mode = False # <-- NEW: Set to True to enable rejection logging
        self.rvol_lookback_bars = 20 # Look back 20 bars for calculating average volume for Rvol
        self.lower_tf_for_volume = '5min' # <-- NEW: Define the lower timeframe for volume check
        self.max_workers = max_workers # Number of worker threads for multi-threading
        self.initial_capital = initial_capital # Add initial capital
        self.risk_per_trade_percent = risk_per_trade_percent # Add risk per trade percentage


        # Dynamically determine HTFs based on entry_tf
        self.higher_timeframes = self._get_higher_timeframes(entry_timeframe)

        # Ensure resampling covers all TFs we might check, including lower_tf_for_volume
        self.all_required_tfs = list(self.SUPPORTED_TFS) # Create a mutable copy
        if self.lower_tf_for_volume not in self.all_required_tfs:
             self.all_required_tfs.insert(0, self.lower_tf_for_volume) # Add lower TF at the beginning if not present


        self.ohlcv_data = {}  # Stores all resampled data: {symbol: {timeframe: df}}
        self.trades = []
        self.open_trade_per_symbol = {} # Track open trade per symbol in multi-threaded backtest
        self._trades_lock = threading.Lock() # Lock for appending to the shared trades list


    def _get_higher_timeframes(self, entry_tf) -> List[str]:
        """Returns all supported timeframes higher than the entry_tf."""
        try:
            start_index = self.SUPPORTED_TFS.index(entry_tf)
            return self.SUPPORTED_TFS[start_index + 1:]
        except ValueError:
            return [] # Should not happen due to validation in __init__

    def _load_and_resample_data(self, file_path):
        """Loads 1m data and resamples it to all required timeframes."""

        # 1. Load 1-minute data
        # Explicitly convert 'date' column to datetime, coercing errors
        df = pd.read_csv(file_path, parse_dates=['date'])
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']) # Drop rows where date parsing failed
        df = df.set_index('date') # Set the datetime column as index

        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.sort_index().replace([np.inf, -np.inf], np.nan).dropna()


        if df.empty: return {}

        resampled_data = {}

        # 2. Resample logic across ALL required timeframes
        for tf in self.all_required_tfs:
            try:
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
                elif self.debug_mode:
                     print(f"  DEBUG: Resampling {os.path.basename(file_path)} to {tf} resulted in empty DataFrame.")

            except Exception as e:
                 if self.debug_mode:
                      print(f"  DEBUG: Error resampling {os.path.basename(file_path)} to {tf}: {e}")
                 continue # Continue to next timeframe if resampling fails


        return resampled_data


    def load_all_market_data(self):
        """Finds all CSV files and loads/resamples data for all symbols."""
        print(f"Loading data from: {self.data_path}")

        csv_files = glob.glob(os.path.join(self.data_path, '3*_minute.csv'))

        # Use ThreadPoolExecutor for faster data loading and resampling
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit a future for each CSV file
            future_to_symbol = {executor.submit(self._load_and_resample_data, file_path): os.path.basename(file_path).split('_minute')[0] for file_path in csv_files}

            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data:
                        self.ohlcv_data[symbol] = data
                        if self.debug_mode:
                             print(f"Processed symbol: {symbol}")
                    elif self.debug_mode:
                        print(f"  DEBUG: No valid data loaded for symbol {symbol}.")

                except Exception as exc:
                    print(f' {symbol} generated an exception during data processing: {exc}')

        print(f"Loaded data for {len(self.ohlcv_data)} symbols.")


    def is_inside_bar(self, current_bar, mother_bar):
        """Checks if the current bar is fully contained within the mother bar."""
        # Using a strict definition: High < MB High AND Low > MB Low
        return (current_bar['High'] < mother_bar['High']) and \
               (current_bar['Low'] > mother_bar['Low'])

    def check_mtf_trend(self, symbol, current_time, direction):
        """
        Checks MTF trend alignment against the dynamically defined higher timeframes.
        Returns: True if aligned with at least one HTF, False otherwise.
        """
        aligned_count = 0
        checked_tfs = 0

        # Only check against the HTFs relative to the entry_tf
        for tf in self.higher_timeframes:
            # Skip if data for this HTF wasn't loaded (e.g., too short a file)
            if tf not in self.ohlcv_data[symbol]:
                 if self.debug_mode:
                     print(f"  DEBUG: Skipping MTF check for {symbol} on {tf} at {current_time} - data not loaded.")
                 continue

            df_htf = self.ohlcv_data[symbol][tf]

            try:
                # Find the index of the most recently closed HTF bar *at or before* the current_time
                # Use boolean indexing
                preceding_bars = df_htf.index[df_htf.index <= current_time]

                if preceding_bars.empty:
                    raise IndexError("No preceding bars found") # No HTF bar at or before current_time

                # Get the index of the last bar at or before current_time
                # Use standard integer-based indexing on the result of the boolean filter
                htf_bar_index_in_filtered = len(preceding_bars) - 1
                htf_bar_index = df_htf.index.get_loc(preceding_bars[htf_bar_index_in_filtered])


                if htf_bar_index < 1:
                    raise IndexError("Not enough preceding bars for trend calculation") # Need at least 2 bars for trend


                htf_bar = df_htf.iloc[htf_bar_index]
                prev_htf_bar = df_htf.iloc[htf_bar_index - 1]

                # Double check that the selected htf_bar is indeed at or before current_time
                if htf_bar.name > current_time:
                     if self.debug_mode:
                          print(f"  DEBUG: Found HTF bar {htf_bar.name} which is after current_time {current_time} for {symbol} on {tf}. Skipping MTF check.")
                     continue # Skip this HTF if the bar is after the current time


            except IndexError:
                # Not enough data for this HTF to calculate trend up to current_time
                if self.debug_mode:
                     print(f"  DEBUG: Not enough data for MTF check for {symbol} on {tf} at {current_time}. Skipping.")
                continue
            except KeyError:
                 if self.debug_mode:
                     print(f"  DEBUG: Could not find matching HTF bar for {symbol} on {tf} at {current_time}. Skipping.")
                 continue
            except Exception as e:
                 if self.debug_mode:
                      print(f"  DEBUG: Unexpected error during MTF bar lookup for {symbol} on {tf} at {current_time}: {e}. Skipping.")
                 continue


            checked_tfs += 1

            current_close = htf_bar['Close']
            current_ema = htf_bar['EMA']
            prev_ema = prev_htf_bar['EMA']

            # Trend Check: Close vs EMA AND EMA slope
            # Adjusted EMA slope check to be strictly greater/less than
            is_up = current_close > current_ema and current_ema > prev_ema
            is_down = current_close < current_ema and current_ema < prev_ema

            if self.debug_mode:
                 trend_status = "UP" if is_up else ("DOWN" if is_down else "SIDEWAYS")
                 alignment_status = "ALIGNED" if (direction == 'LONG' and is_up) or (direction == 'SHORT' and is_down) else "NOT ALIGNED"
                 print(f"  DEBUG: MTF Check {tf} for {symbol} at {current_time}: HTF Bar Time={htf_bar.name}, Close={current_close:.2f}, EMA={current_ema:.2f}, PrevEMA={prev_ema:.2f}. Trend: {trend_status}. Alignment: {alignment_status}")


            if (direction == 'LONG' and is_up) or (direction == 'SHORT' and is_down):
                aligned_count += 1

        # Must align with at least one higher timeframe, *if* there were any HTFs checked.
        # If no HTFs were checked (e.g., entry TF is 1W), then MTF check is considered passed.
        is_aligned = aligned_count > 0 or checked_tfs == 0

        if self.debug_mode:
            if not is_aligned and checked_tfs > 0:
                 print(f"REJECTED: MTF alignment failed for {symbol} at {current_time} ({direction}). Aligned with {aligned_count}/{checked_tfs} TFs.")
            elif is_aligned and checked_tfs > 0:
                 print(f"DEBUG: MTF alignment PASSED for {symbol} at {current_time} ({direction}). Aligned with {aligned_count}/{checked_tfs} TFs.")
            elif checked_tfs == 0:
                 print(f"DEBUG: No higher timeframes to check for {symbol} at {current_time}. MTF check PASSED by default.")


        return is_aligned


    # Simplifying find_inside_bar_breakout to look for a single Inside Bar

    def find_inside_bar_breakout(self, df_bars, current_index):
        """
        Identifies a Mother Bar followed by a single Inside Bar ending at current_index - 1.
        Returns: (is_valid, MB_Bar, IBR_High, IBR_Low, IBR_cluster_start_index)
        """
        # We need at least 2 bars before the current_index to have a potential MB and a single IB
        if current_index < 2:
            if self.debug_mode:
                current_time = df_bars.index[current_index] if current_index < len(df_bars) else "Unknown Time"
                symbol = df_bars.iloc[:current_index+1].index.name if df_bars.iloc[:current_index+1].index.name else "Unknown Symbol"
                print(f"REJECTED: Not enough bars ({current_index+1}) to check for single IB pattern ending at index {current_index} ({current_time}) on {symbol}. Need at least 3 total bars (MB, IB, Breakout).")
            return False, None, None, None, None

        current_time = df_bars.index[current_index]
        symbol = df_bars.iloc[:current_index+1].index.name if df_bars.iloc[:current_index+1].index.name else "Unknown Symbol"


        # The potential Inside Bar is the bar immediately preceding the potential breakout bar.
        potential_ib_index = current_index - 1

        # The potential Mother Bar is the bar immediately preceding the potential Inside Bar.
        potential_mb_index = potential_ib_index - 1

        # Check if we have enough bars for MB and IB
        if potential_mb_index < 0:
             # This check is technically redundant due to the initial current_index < 2 check, but kept for clarity.
             if self.debug_mode:
                  print(f"REJECTED: Not enough bars to check for MB before potential IB at index {potential_ib_index} ({df_bars.index[potential_ib_index]}) for {symbol} at {current_time}.")
             return False, None, None, None, None


        MB_bar = df_bars.iloc[potential_mb_index]
        IB_bar = df_bars.iloc[potential_ib_index]

        if self.debug_mode:
            print(f"\nDEBUG: find_inside_bar_breakout (single IB) called for {symbol} at index {current_index} ({current_time}). Potential MB index: {potential_mb_index} ({MB_bar.name}), Potential IB index: {potential_ib_index} ({IB_bar.name}).")
            print(f"DEBUG: Checking if bar at index {potential_ib_index} is inside bar at index {potential_mb_index}...")


        # 1. Check if the potential IB is an actual Inside Bar relative to the potential MB
        if not self.is_inside_bar(IB_bar, MB_bar):
            if self.debug_mode:
                 print(f"REJECTED: Bar at index {potential_ib_index} ({IB_bar.name}) is NOT an Inside Bar relative to MB at index {potential_mb_index} ({MB_bar.name}) for {symbol} at {current_time}. IB H/L: {IB_bar['High']:.2f}/{IB_bar['Low']:.2f}, MB H/L: {MB_bar['High']:.2f}/{MB_bar['Low']:.2f}")
            return False, None, None, None, None

        if self.debug_mode:
             print(f"DEBUG: Bar at index {potential_ib_index} IS an Inside Bar.")


        # 2. Volume Validation (Contraction Filter) - Check IB volume vs MB volume
        MB_volume = MB_bar['Volume']
        IB_volume = IB_bar['Volume']

        # IB Volume must be strictly lower than MB volume
        if IB_volume >= MB_volume:
            if self.debug_mode:
                print(f"REJECTED: Volume contraction filter failed (IB volume >= MB volume) for single IB at {current_time} on {symbol}. MB Vol: {MB_volume:.0f}, IB Vol: {IB_volume:.0f}")

            return False, None, None, None, None

        if self.debug_mode:
             print(f"DEBUG: Volume contraction filter passed for single IB.")


        # For a single IB, the IBR range is simply the IB's range.
        IBR_High = IB_bar['High']
        IBR_Low = IB_bar['Low']
        ibr_cluster_start_index = potential_ib_index # The IBR cluster starts at the IB itself for single IB

        # Pattern is valid if we reached here.
        if self.debug_mode:
             print(f"DEBUG: Valid single IB pattern found for {symbol} at {current_time}. MB @{MB_bar.name}, IB @{IB_bar.name}. IBR High: {IBR_High:.2f}, IBR Low: {IBR_Low:.2f}, MB Vol: {MB_volume:.0f}, IB Vol: {IB_volume:.0f}")


        return True, MB_bar, IBR_High, IBR_Low, ibr_cluster_start_index


    def _manage_open_trade(self, current_bar, current_time):
        """Checks if the open trade hits SL or TP."""
        trade = self.open_trade # This method is not used in the multi-threaded version
        if not trade: return

        # Check for Close

        # 1. Check for Stop Loss (SL) Hit
        if (trade.direction == 'LONG' and current_bar['Low'] <= trade.sl) or \
           (trade.direction == 'SHORT' and current_bar['High'] >= trade.sl):

            # SL hit logic: Assuming price hits SL before TP/Reversal within the bar
            close_price = trade.sl
            trade.close(close_price, current_time)
            # In multi-threading, append to a thread-local list or use a lock
            with self._trades_lock:
                 self.trades.append(trade)
            self.open_trade = None
            if self.debug_mode:
                 print(f"DEBUG: Trade closed on {trade.symbol} at {current_time} - SL Hit. PnL: {trade.pnl:.2f}")
            return

        # 2. Check for Take Profit (TP) Hit
        if (trade.direction == 'LONG' and current_bar['High'] >= trade.tp) or \
           (trade.direction == 'SHORT' and current_bar['Low'] <= trade.tp):

            # TP hit logic: Assuming fill at TP
            close_price = trade.tp
            trade.close(close_price, current_time)
            # In multi-threading, append to a thread-local list or use a lock
            with self._trades_lock:
                 self.trades.append(trade)
            self.open_trade = None
            if self.debug_mode:
                 print(f"DEBUG: Trade closed on {trade.symbol} at {current_time} - TP Hit. PnL: {trade.pnl:.2f}")
            return

        # 3. Trailing Stop Logic (Simplified: Move to BE at 1R, then trail 2*ATR)
        r_distance = trade.initial_risk

        if trade.status == 'OPEN':

            # Check 1R BE move
            if trade.direction == 'LONG' and current_bar['Close'] >= trade.entry_price + r_distance:
                if trade.sl < trade.entry_price: # Move to BE
                    trade.sl = trade.entry_price
                    if self.debug_mode:
                         print(f"DEBUG: Trailing Stop: Moved SL to Break-Even ({trade.sl:.2f}) for {trade.symbol} at {current_time}.")
            elif trade.direction == 'SHORT' and current_bar['Close'] <= trade.entry_price - r_distance:
                if trade.sl > trade.entry_price: # Move to BE
                    trade.sl = trade.entry_price
                    if self.debug_mode:
                         print(f"DEBUG: Trailing Stop: Moved SL to Break-Even ({trade.sl:.2f}) for {trade.symbol} at {current_time}.")


            # Check ATR trailing stop (only after enough bars for ATR calculation and if not already at BE or beyond)
            if 'ATR' in current_bar and not pd.isna(current_bar['ATR']) and current_bar['ATR'] > 0:
                atr_stop_level = 2 * current_bar['ATR']

                if trade.direction == 'LONG':
                    new_sl = current_bar['Close'] - atr_stop_level
                    # Only trail up, and only if the new SL is above the current SL (and above initial SL/BE if applicable)
                    if new_sl > trade.sl:
                         trade.sl = new_sl
                         if self.debug_mode:
                              print(f"DEBUG: Trailing Stop: Moved SL up to {trade.sl:.2f} using ATR for {trade.symbol} at {current_time}.")
                 # Fix indentation for the SHORT direction trailing stop
                elif trade.direction == 'SHORT':
                    new_sl = current_bar['Close'] + atr_stop_level
                    # Only trail down, and only if the new SL is below the current SL (and below initial SL/BE if applicable)
                    if new_sl < trade.sl:
                         trade.sl = new_sl
                         if self.debug_mode:
                              print(f"DEBUG: Trailing Stop: Moved SL down to {trade.sl:.2f} using ATR for {trade.symbol} at {current_time}.")

        return None # Trade is still open


    def run_backtest(self):
        """
        Executes the backtest logic bar-by-bar for all loaded symbols using multi-threading.
        Includes closing open trades at the end of the backtest period.
        """
        if not self.ohlcv_data:
            print("Error: No market data loaded. Run load_all_market_data() first.")
            return

        self.trades = [] # Reset trades list before running
        self.open_trade_per_symbol = {} # Reset open trades tracking

        # Use ThreadPoolExecutor for parallel backtesting of symbols
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit a future for each symbol
            future_to_symbol = {executor.submit(self._run_backtest_for_symbol, symbol, data): symbol for symbol, data in self.ohlcv_data.items()}

            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    # The result of _run_backtest_for_symbol will be a list of closed trades for that symbol
                    trades_for_symbol, open_trade_symbol = future.result()
                    # Append closed trades from each symbol to the main trades list
                    with self._trades_lock:
                         self.trades.extend(trades_for_symbol)
                         # Store the final state of the open trade for this symbol
                         if open_trade_symbol:
                              self.open_trade_per_symbol[symbol] = open_trade_symbol

                except Exception as exc:
                    print(f' {symbol} generated an exception during backtesting: {exc}')

        # After all symbols are processed, close any remaining open trades at the last available price
        self._close_remaining_open_trades()


    def _run_backtest_for_symbol(self, symbol, data):
        """Runs the backtest logic for a single symbol."""
        df_et = data.get(self.entry_tf)
        df_lower_tf_symbol = data.get(self.lower_tf_for_volume) # Get lower TF data specific to this symbol

        if df_et is None:
             if self.debug_mode:
                  print(f"  DEBUG: Entry timeframe data ({self.entry_tf}) not available for {symbol}. Skipping symbol.")
             return [], None # Return empty list of trades and no open trade


        print(f"\n--- Starting Backtest for {symbol} on {self.entry_tf} ---")

        # Initialize open_trade for this thread/symbol
        open_trade_symbol = None
        trades_symbol = [] # List to store trades for this symbol

        # Iterate through the entry timeframe bars (starting after min data needed for indicators)
        min_bars = 20 # Need at least 20 bars for EMA and ATR calculation
        if len(df_et) <= min_bars + self.rvol_lookback_bars: # Ensure enough data for Rvol calculation as well
            if self.debug_mode:
                print(f"  DEBUG: Not enough data ({len(df_et)} bars) for {symbol} on {self.entry_tf} to start backtest (min required: {min_bars + self.rvol_lookback_bars}). Skipping.")
            return [], None # Return empty list of trades and no open trade


        for i in range(min_bars, len(df_et)):
            current_time = df_et.index[i]
            current_bar = df_et.iloc[i]

            # --- Part B: Trade Management ---
            if open_trade_symbol:
                # Manage the open trade for this symbol
                closed_trade = self._manage_open_trade_single_symbol(open_trade_symbol, current_bar, current_time)
                if closed_trade:
                     trades_symbol.append(closed_trade)
                     open_trade_symbol = None
                # Continue to next bar if trade is still open
                if open_trade_symbol:
                    continue

            # --- Part A: Pattern Detection & Order Placement ---

            # Check pattern on last closed bar (index i-1)
            # Pass bars up to index i (inclusive) to find_inside_bar_breakout
            # because the method expects the potential breakout bar's index (current_index)
            # and will look at the bar before it (current_index - 1) as the potential IB.
            is_valid, mb_bar, ibr_high, ibr_low, ib_start_index = \
                self.find_inside_bar_breakout(df_et.iloc[:i+1], i) # Pass current bar index 'i'


            if not is_valid: continue

            # Pattern found! Now check for breakout and MTF alignment and Volume confirmation

            # 1. Calculate parameters

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
            next_bar = df_et.iloc[i] # The bar where the breakout occurs (same as current_bar)

            long_breakout_condition = next_bar['High'] >= long_entry
            short_breakout_condition = next_bar['Low'] <= short_entry

            # --- LONG BREAKOUT CHECK ---
            if long_breakout_condition:
                if self.debug_mode:
                     print(f"DEBUG: Long breakout condition met at {next_bar.name} on {symbol}.")

                # Check MTF trend alignment
                mtf_aligned_long = self.check_mtf_trend(symbol, current_time, 'LONG')

                if mtf_aligned_long:
                    # Check Entry TF Volume Breakout Confirmation (Rvol > 1)
                    lookback_slice = df_et.iloc[max(0, i - self.rvol_lookback_bars):i] # Bars BEFORE the breakout bar
                    if not lookback_slice.empty and lookback_slice['Volume'].mean() > 0:
                         average_volume = lookback_slice['Volume'].mean()
                         breakout_volume = next_bar['Volume']
                         rvol = breakout_volume / average_volume

                         if self.debug_mode:
                              print(f"DEBUG: Checking Rvol for LONG breakout: Breakout Vol: {breakout_volume:.0f}, Avg Lookback Vol ({self.rvol_lookback_bars} bars): {average_volume:.0f}, Rvol: {rvol:.2f}")

                         if rvol > 1.0: # Rvol > 1 condition
                             # Rvol condition met. Now check Lower Timeframe Volume.

                             lower_tf_volume_confirmed = True # Assume true initially
                             if df_lower_tf_symbol is not None:
                                 # Get lower timeframe bars within the entry timeframe breakout bar's time range
                                 entry_tf_freq = pd.to_timedelta(self.entry_tf)
                                 # Calculate the start time of the current entry TF bar
                                 breakout_bar_start_time = current_time - entry_tf_freq

                                 # Select lower TF bars that fall exactly within the current entry TF bar's time range
                                 # Use inclusive start and end for robust selection
                                 lower_tf_bars_in_breakout_bar = df_lower_tf_symbol.loc[
                                     (df_lower_tf_symbol.index >= breakout_bar_start_time) & (df_lower_tf_symbol.index <= current_time)
                                 ]

                                 if not lower_tf_bars_in_breakout_bar.empty:
                                      # Check for a volume spike on the lower timeframe bars
                                      # For simplicity, check if max lower TF volume is > 1.5 * average recent lower TF volume?
                                      # Need to calculate average recent lower TF volume *before* the breakout bar started
                                      recent_lower_tf_bars = df_lower_tf_symbol.loc[df_lower_tf_symbol.index < breakout_bar_start_time]
                                      # Look back a window equivalent to 5 entry TF bars in the lower timeframe
                                      try:
                                          lower_tf_freq_td = pd.to_timedelta(self.lower_tf_for_volume)
                                          window_size_lower_tf = int(entry_tf_freq.total_seconds() / lower_tf_freq_td.total_seconds()) * 5
                                          recent_lower_tf_bars = recent_lower_tf_bars.tail(window_size_lower_tf)
                                      except Exception as e:
                                           if self.debug_mode:
                                                print(f"DEBUG: Error calculating lower TF window size for {symbol} at {current_time}: {e}. Skipping lower TF volume check.")
                                           lower_tf_volume_confirmed = False # Treat as not confirmed if calculation fails
                                           recent_lower_tf_bars = pd.DataFrame() # Ensure empty to skip avg volume calculation


                                      if not recent_lower_tf_bars.empty and recent_lower_tf_bars['Volume'].mean() > 0:
                                           avg_recent_lower_tf_volume = recent_lower_tf_bars['Volume'].mean()
                                           max_lower_tf_volume_in_breakout_bar = lower_tf_bars_in_breakout_bar['Volume'].max()
                                           lower_tf_volume_spike_threshold = 1.5 * avg_recent_lower_tf_volume

                                           if self.debug_mode:
                                                print(f"DEBUG: Checking LOWER TF ({self.lower_tf_for_volume}) LONG volume breakout: Max Vol: {max_lower_tf_volume_in_breakout_bar:.0f}, Avg Recent Vol: {avg_recent_lower_tf_volume:.0f}, Threshold: {lower_tf_volume_spike_threshold:.0f}")


                                           if max_lower_tf_volume_in_breakout_bar > lower_tf_volume_spike_threshold:
                                                lower_tf_volume_confirmed = True
                                                if self.debug_mode: print(f"DEBUG: LOWER TF LONG volume breakout confirmed.")
                                           elif self.debug_mode:
                                                lower_tf_volume_confirmed = False # Explicitly set to False for logging clarity
                                                print(f"REJECTED: LOWER TF LONG volume breakout filter failed.")
                                      elif self.debug_mode:
                                           lower_tf_volume_confirmed = False # Explicitly set to False
                                           print(f"DEBUG: Not enough recent {self.lower_tf_for_volume} data to calculate average for LOWER TF volume check for LONG.")

                                 elif self.debug_mode:
                                          lower_tf_volume_confirmed = False # Explicitly set to False
                                          print(f"DEBUG: No {self.lower_tf_for_volume} bars found within the entry TF breakout bar range ({breakout_bar_start_time} to {current_time}) for LONG.")
                             elif self.debug_mode:
                                lower_tf_volume_confirmed = False # Explicitly set to False
                                print(f"DEBUG: Lower timeframe data ({self.lower_tf_for_volume}) not available for {symbol}. Skipping lower TF volume check for LONG.")


                             if lower_tf_volume_confirmed: # This will be True only if lower TF check passed and df_lower_tf_symbol is not None
                                # Breakout, MTF, Rvol > 1, and Lower TF Volume confirmed, creating trade

                                # Calculate trade size
                                point_value = 1 # Assuming point value of 1 for stocks
                                stop_loss_distance = long_entry - long_sl
                                if stop_loss_distance > 0:
                                     trade_size = int((self.initial_capital * self.risk_per_trade_percent) / (stop_loss_distance * point_value))
                                else:
                                     trade_size = 0

                                if trade_size > 0:
                                     open_trade_symbol = Trade(symbol, next_bar.name, long_entry, 'LONG', long_sl, long_tp, long_risk, trade_size) # Pass trade_size
                                     if self.debug_mode:
                                         print(f"DEBUG: Opened LONG trade on {symbol} at {next_bar.name} Entry: {long_entry:.2f}, SL: {long_sl:.2f}, TP: {long_tp:.2f}, Size: {trade_size}.")
                                elif self.debug_mode:
                                     print(f"REJECTED: Calculated trade size is zero or negative ({trade_size}) for LONG at {next_bar.name} on {symbol}. Stop loss distance: {stop_loss_distance:.2f}")

                                 # If lower_tf_volume_confirmed is False and df_lower_tf_symbol is not None, rejection logged inside the check.


                             elif self.debug_mode:
                                  print(f"REJECTED: Rvol filter failed (Rvol <= 1.0) for LONG at {next_bar.name} on {symbol}.")
                         elif self.debug_mode:
                            print(f"DEBUG: Not enough lookback data ({len(lookback_slice)} bars) or zero average volume for Rvol check for LONG at {next_bar.name} on {symbol}.")

                    elif self.debug_mode:
                        # MTF alignment already logged as rejected inside check_mtf_trend
                        pass # No need to duplicate rejection message here

            # --- SHORT BREAKOUT CHECK ---
            if short_breakout_condition:
                if self.debug_mode:
                    print(f"DEBUG: Short breakout condition met at {next_bar.name} on {symbol}.")
                # Check MTF trend alignment
                mtf_aligned_short = self.check_mtf_trend(symbol, current_time, 'SHORT')

                if mtf_aligned_short:
                     # Check Volume Breakout Confirmation on the entry timeframe breakout bar
                    lookback_slice = df_et.iloc[max(0, i - self.rvol_lookback_bars):i] # Bars BEFORE the breakout bar
                    if not lookback_slice.empty and lookback_slice['Volume'].mean() > 0:
                         average_volume = lookback_slice['Volume'].mean()
                         breakout_volume = next_bar['Volume']
                         rvol = breakout_volume / average_volume

                         if self.debug_mode:
                              print(f"DEBUG: Checking Rvol for SHORT breakout: Breakout Vol: {breakout_volume:.0f}, Avg Lookback Vol ({self.rvol_lookback_bars} bars): {average_volume:.0f}, Rvol: {rvol:.2f}")

                         if rvol > 1.0: # Rvol > 1 condition
                             # Rvol condition met. Now check Lower Timeframe Volume.

                             lower_tf_volume_confirmed = True # Assume true initially
                             if df_lower_tf_symbol is not None:
                                 # Get lower timeframe bars within the entry timeframe breakout bar's time range
                                 entry_tf_freq = pd.to_timedelta(self.entry_tf)
                                 breakout_bar_start_time = current_time - entry_tf_freq

                                 lower_tf_bars_in_breakout_bar = df_lower_tf_symbol.loc[
                                     (df_lower_tf_symbol.index >= breakout_bar_start_time) & (df_lower_tf_symbol.index <= current_time)
                                 ]

                                 if not lower_tf_bars_in_breakout_bar.empty:
                                      # Check for a volume spike on the lower timeframe bars
                                      # For simplicity, check if max lower TF volume is > 1.5 * average recent lower TF volume
                                      recent_lower_tf_bars = df_lower_tf_symbol.loc[df_lower_tf_symbol.index < breakout_bar_start_time]
                                      window_size_lower_tf = int(entry_tf_freq.total_seconds()/60) * 5
                                      recent_lower_tf_bars = recent_lower_tf_bars.tail(window_size_lower_tf)

                                      if not recent_lower_tf_bars.empty:
                                           avg_recent_lower_tf_volume = recent_lower_tf_bars['Volume'].mean()
                                           max_lower_tf_volume_in_breakout_bar = lower_tf_bars_in_breakout_bar['Volume'].max()
                                           lower_tf_volume_spike_threshold = 1.5 * avg_recent_lower_tf_volume

                                           if self.debug_mode:
                                                print(f"DEBUG: Checking LOWER TF ({self.lower_tf_for_volume}) SHORT volume breakout: Max Vol: {max_lower_tf_volume_in_breakout_bar:.0f}, Avg Recent Vol: {avg_recent_lower_tf_volume:.0f}, Threshold: {lower_tf_volume_spike_threshold:.0f}")

                                           if max_lower_tf_volume_in_breakout_bar > lower_tf_volume_spike_threshold:
                                                lower_tf_volume_confirmed = True
                                                if self.debug_mode: print(f"DEBUG: LOWER TF SHORT volume breakout confirmed.")
                                           elif self.debug_mode:
                                                lower_tf_volume_confirmed = False # Explicitly set to False for logging clarity
                                                print(f"REJECTED: LOWER TF SHORT volume breakout filter failed.")
                                      elif self.debug_mode:
                                           lower_tf_volume_confirmed = False # Explicitly set to False
                                           print(f"DEBUG: Not enough recent {self.lower_tf_for_volume} data to calculate average for LOWER TF volume check for SHORT.")
                                 elif self.debug_mode:
                                      lower_tf_volume_confirmed = False # Explicitly set to False
                                      print(f"DEBUG: No {self.lower_tf_for_volume} bars found within the entry TF breakout bar range ({breakout_bar_start_time} to {current_time}) for SHORT.")
                                 elif self.debug_mode:
                                      lower_tf_volume_confirmed = False # Explicitly set to False
                                      print(f"DEBUG: Lower timeframe data ({self.lower_tf_for_volume}) not available for {symbol}. Skipping lower TF volume check for SHORT.")


                             if lower_tf_volume_confirmed: # This will be True only if lower TF check passed and df_lower_tf_symbol is not None
                                # Breakout, MTF, Rvol > 1, and Lower TF Volume confirmed, creating trade

                                # Calculate trade size
                                point_value = 1 # Assuming point value of 1 for stocks
                                stop_loss_distance = short_sl - short_entry
                                if stop_loss_distance > 0:
                                     trade_size = int((self.initial_capital * self.risk_per_trade_percent) / (stop_loss_distance * point_value))
                                else:
                                     trade_size = 0

                                if trade_size > 0:
                                     open_trade_symbol = Trade(symbol, next_bar.name, short_entry, 'SHORT', short_sl, short_tp, short_risk, trade_size) # Pass trade_size
                                     if self.debug_mode:
                                         print(f"DEBUG: Opened SHORT trade on {symbol} at {next_bar.name} Entry: {short_entry:.2f}, SL: {short_sl:.2f}, TP: {short_tp:.2f}, Size: {trade_size}.")
                                elif self.debug_mode:
                                     print(f"REJECTED: Calculated trade size is zero or negative ({trade_size}) for SHORT at {next_bar.name} on {symbol}. Stop loss distance: {stop_loss_distance:.2f}")

                             # If lower_tf_volume_confirmed is False and df_lower_tf_symbol is not None, rejection logged inside the check.


                         elif self.debug_mode:
                              print(f"REJECTED: Rvol filter failed (Rvol <= 1.0) for SHORT at {next_bar.name} on {symbol}.")
                    elif self.debug_mode:
                        print(f"DEBUG: Not enough lookback data ({len(lookback_slice)} bars) or zero average volume for Rvol check for SHORT at {next_bar.name} on {symbol}.")
                elif self.debug_mode:
                    # MTF alignment already logged as rejected inside check_mtf_trend
                    pass # No need to duplicate rejection message here

        # Return the list of trades and the final state of the open trade for this symbol
        return trades_symbol, open_trade_symbol


    def _manage_open_trade_single_symbol(self, trade, current_bar, current_time):
         """Checks if a single open trade for a symbol hits SL or TP and returns the closed trade."""
         if not trade: return None

         # Check for Close

         # 1. Check for Stop Loss (SL) Hit
         if (trade.direction == 'LONG' and current_bar['Low'] <= trade.sl) or \
            (trade.direction == 'SHORT' and current_bar['High'] >= trade.sl):

             # SL hit logic: Assuming price hits SL before TP/Reversal within the bar
             close_price = trade.sl
             trade.close(close_price, current_time)
             if self.debug_mode:
                  print(f"DEBUG: Trade closed on {trade.symbol} at {current_time} - SL Hit. PnL: {trade.pnl:.2f}")
             return trade # Return the closed trade


         # 2. Check for Take Profit (TP) Hit
         if (trade.direction == 'LONG' and current_bar['High'] >= trade.tp) or \
            (trade.direction == 'SHORT' and current_bar['Low'] <= trade.tp):

             # TP hit logic: Assuming fill at TP
             close_price = trade.tp
             trade.close(close_price, current_time)
             if self.debug_mode:
                  print(f"DEBUG: Trade closed on {trade.symbol} at {current_time} - TP Hit. PnL: {trade.pnl:.2f}")
             return trade # Return the closed trade

         # 3. Trailing Stop Logic (Simplified: Move to BE at 1R, then trail 2*ATR)
         # Calculate the risk distance in price points
         risk_distance_points = abs(trade.entry_price - trade.initial_risk) # Use initial_risk which is the price difference

         if trade.status == 'OPEN':

             # Check 1R BE move
             if trade.direction == 'LONG' and current_bar['Close'] >= trade.entry_price + risk_distance_points:
                 # Ensure SL is not already at or beyond entry price before moving to BE
                 if trade.sl < trade.entry_price:
                     trade.sl = trade.entry_price
                     if self.debug_mode:
                          print(f"DEBUG: Trailing Stop: Moved SL to Break-Even ({trade.sl:.2f}) for {trade.symbol} at {current_time}.")
             elif trade.direction == 'SHORT' and current_bar['Close'] <= trade.entry_price - risk_distance_points:
                 # Ensure SL is not already at or beyond entry price before moving to BE
                 if trade.sl > trade.entry_price:
                     trade.sl = trade.entry_price
                     if self.debug_mode:
                          print(f"DEBUG: Trailing Stop: Moved SL to Break-Even ({trade.sl:.2f}) for {trade.symbol} at {current_time}.")


             # Check ATR trailing stop (only after enough bars for ATR calculation and if not already at BE or beyond)
             if 'ATR' in current_bar and not pd.isna(current_bar['ATR']) and current_bar['ATR'] > 0:
                 atr_stop_level = 2 * current_bar['ATR']

                 if trade.direction == 'LONG':
                     new_sl = current_bar['Close'] - atr_stop_level
                     # Only trail up, and only if the new SL is above the current SL (which might be BE)
                     if new_sl > trade.sl:
                          trade.sl = new_sl
                          if self.debug_mode:
                               print(f"DEBUG: Trailing Stop: Moved SL up to {trade.sl:.2f} using ATR for {trade.symbol} at {current_time}.")
                 elif trade.direction == 'SHORT':
                     new_sl = current_bar['Close'] + atr_stop_level
                     # Only trail down, and only if the new SL is below the current SL (which might be BE)
                     if new_sl < trade.sl:
                          trade.sl = new_sl
                          if self.debug_mode:
                               print(f"DEBUG: Trailing Stop: Moved SL down to {trade.sl:.2f} using ATR for {trade.symbol} at {current_time}.")

         return None # Trade is still open


    def _close_remaining_open_trades(self):
        """Closes any trades that are still open at the end of the backtest."""
        print("\nClosing remaining open trades...")
        for symbol, trade in self.open_trade_per_symbol.items():
            if trade and trade.status == 'OPEN':
                # Find the last available price for this symbol on the entry timeframe
                df_et = self.ohlcv_data.get(symbol, {}).get(self.entry_tf)
                if df_et is not None and not df_et.empty:
                    last_bar = df_et.iloc[-1]
                    last_price = last_bar['Close']
                    last_time = last_bar.name
                    trade.close(last_price, last_time)
                    # Append the closed trade to the main trades list
                    with self._trades_lock:
                         self.trades.append(trade)
                    if self.debug_mode:
                         print(f"DEBUG: Trade closed on {trade.symbol} at {last_time} - End of Backtest. PnL: {trade.pnl:.2f}")
                elif self.debug_mode:
                    print(f"DEBUG: Could not close open trade for {symbol} - No data available on {self.entry_tf}.")

        self.open_trade_per_symbol = {} # Clear open trades after closing


    def calculate_drawdown(self):
        """Calculates the maximum drawdown."""
        if not self.trades:
            return 0.0

        # Sort trades by close time to calculate cumulative PnL correctly
        sorted_trades = sorted(self.trades, key=lambda t: t.close_time if t.close_time else pd.to_datetime('9999-12-31'))

        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_drawdown = 0.0

        for trade in sorted_trades:
            cumulative_pnl += trade.pnl
            peak_pnl = max(peak_pnl, cumulative_pnl)
            drawdown = peak_pnl - cumulative_pnl
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def calculate_sharpe_ratio(self, daily_returns, annualizing_factor=252):
        """Calculates the Annualized Sharpe Ratio from a daily returns series."""
        if len(daily_returns) == 0:
            return 0.0

        risk_free_rate_daily = 0.0 # Or use an actual daily risk-free rate
        excess_returns = daily_returns - risk_free_rate_daily

        avg_excess_return = excess_returns.mean()
        std_dev_excess_return = excess_returns.std()

        if std_dev_excess_return == 0 or np.isnan(std_dev_excess_return): # Handle nan std dev
            return 0.0

        # Correct annualization: first calculate the daily ratio, then annualize with sqrt(252)
        daily_sharpe = avg_excess_return / std_dev_excess_return
        return daily_sharpe * np.sqrt(annualizing_factor)


    def calculate_sortino_ratio(self, daily_returns, annualizing_factor=252):
        """Calculates the Annualized Sortino Ratio from a daily returns series."""
        if len(daily_returns) == 0:
            return 0.0

        risk_free_rate_daily = 0.0 # Or a daily Minimum Acceptable Return (MAR)
        excess_returns = daily_returns - risk_free_rate_daily

        # Calculate downside deviation using only negative excess returns
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return np.nan # Or 0.0 depending on preference

        downside_std_dev = downside_returns.std()

        if downside_std_dev == 0 or np.isnan(downside_std_dev): # Handle nan std dev
            return 0.0

        # Correct annualization
        avg_excess_return = excess_returns.mean() # Use mean of all excess returns
        daily_sortino = avg_excess_return / downside_std_dev
        return daily_sortino * np.sqrt(annualizing_factor)

    def calculate_per_symbol_metrics(self) -> dict:
        """Calculates performance metrics for each symbol."""
        per_symbol_metrics = {}
        trades_by_symbol = {}

        # Group trades by symbol
        for trade in self.trades:
            if trade.symbol not in trades_by_symbol:
                trades_by_symbol[trade.symbol] = []
            trades_by_symbol[trade.symbol].append(trade)

        for symbol, trades in trades_by_symbol.items():
            total_pnl = sum(t.pnl for t in trades)
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]

            total_trades = len(trades)
            win_rate = len(winning_trades) / total_trades if total_trades else 0

            # Calculate max drawdown for the symbol
            if trades:
                sorted_trades = sorted(trades, key=lambda t: t.close_time if t.close_time else pd.to_datetime('9999-12-31'))
                cumulative_pnl = 0.0
                peak_pnl = 0.0
                max_drawdown = 0.0
                for trade in sorted_trades:
                    cumulative_pnl += trade.pnl
                    peak_pnl = max(peak_pnl, cumulative_pnl)
                    drawdown = peak_pnl - cumulative_pnl
                    max_drawdown = max(max_drawdown, drawdown)
            else:
                max_drawdown = 0.0

            # Calculate daily returns for Sharpe and Sortino for the symbol
            closed_trades = [t for t in trades if t.status == 'CLOSED']
            sorted_closed_trades = sorted(closed_trades, key=lambda t: t.close_time)
            if sorted_closed_trades:
                trades_df = pd.DataFrame([{'date': t.close_time.date(), 'pnl': t.pnl} for t in sorted_closed_trades])
                date_range = pd.date_range(trades_df['date'].min(), trades_df['date'].max())
                daily_pnl = trades_df.groupby('date')['pnl'].sum().reindex(date_range).fillna(0)
                if self.initial_capital is not None and self.initial_capital > 0:
                     # Calculate daily returns relative to the initial capital for consistency
                     daily_returns = daily_pnl / self.initial_capital
                else:
                     daily_returns = pd.Series(0.0, index=date_range)
            else:
                daily_returns = pd.Series(dtype=float)


            # Using a fixed annualizing factor of 252 as requested
            annualizing_factor = 252

            sharpe_ratio = self.calculate_sharpe_ratio(daily_returns, annualizing_factor)
            sortino_ratio = self.calculate_sortino_ratio(daily_returns, annualizing_factor)


            per_symbol_metrics[symbol] = {
                "Total Trades": total_trades,
                "Winning Trades": len(winning_trades),
                "Losing Trades": len(losing_trades),
                "Win Rate (%)": win_rate * 100,
                "Total PnL (Currency)": total_pnl,
                "Maximum Drawdown (Currency)": max_drawdown,
                "Sharpe Ratio (Annualized)": sharpe_ratio,
                "Sortino Ratio (Annualized)": sortino_ratio
            }

        return per_symbol_metrics


    def generate_detailed_report(self) -> str:
        """Generates a detailed list of all closed trades and returns as a string."""
        report_output = ""
        report_output += "\n" + "="*50 + "\n"
        report_output += "              DETAILED TRADE REPORT\n"
        report_output += "="*50 + "\n"
        if not self.trades:
            report_output += "No trades were executed during the backtest.\n"
            return report_output

        # Define column headers
        header = ["Symbol", "Direction", "Entry Time", "Entry Price", "Exit Time", "Exit Price", "Size", "Initial Risk (Points)", "PnL (Currency)", "Status"]

        # Define format string for the header (all strings)
        header_format_string = "{:<10} {:<10} {:<20} {:<12} {:<20} {:<12} {:<8} {:<20} {:<15} {:<10}\n"

        # Define format string for the trade data (mix of strings, floats, ints)
        trade_format_string = "{:<10} {:<10} {:<20} {:<12.2f} {:<20} {:<12.2f} {:<8} {:<20.2f} {:<15.2f} {:<10}\n"


        report_output += header_format_string.format(*header) # Use the header format string for the header
        report_output += "-" * 150 + "\n" # Separator line

        for trade in self.trades:
            # Ensure close_price is not None before formatting as float
            close_price_formatted = trade.close_price if trade.close_price is not None else 0.0
            # Ensure initial_risk is not None before formatting as float
            initial_risk_formatted = abs(trade.entry_price - trade.initial_risk) if trade.initial_risk is not None else 0.0


            report_output += trade_format_string.format( # Use the trade format string for trade data
                trade.symbol,
                trade.direction,
                trade.entry_time.strftime('%Y-%m-%d %H:%M'),
                trade.entry_price,
                trade.close_time.strftime('%Y-%m-%d %H:%M') if trade.close_time else 'N/A',
                close_price_formatted,
                trade.size,
                initial_risk_formatted, # Use the formatted initial risk
                trade.pnl,
                trade.status
            )
        report_output += "="*50 + "\n"

        return report_output


    def generate_summary_report(self) -> str:
        """Generates a consolidated performance summary and returns as a string."""
        report_output = ""
        total_pnl = sum(t.pnl for t in self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades else 0
        max_dd = self.calculate_drawdown() # Calculate max drawdown

        # Using a fixed annualizing factor of 252 as requested
        annualizing_factor = 252

        # Calculate daily returns for Sharpe and Sortino
        closed_trades = [t for t in self.trades if t.status == 'CLOSED']
        sorted_closed_trades = sorted(closed_trades, key=lambda t: t.close_time)
        if sorted_closed_trades:
            trades_df = pd.DataFrame([{'date': t.close_time.date(), 'pnl': t.pnl} for t in sorted_closed_trades])
            date_range = pd.date_range(trades_df['date'].min(), trades_df['date'].max())
            daily_pnl = trades_df.groupby('date')['pnl'].sum().reindex(date_range).fillna(0)
            if self.initial_capital is not None and self.initial_capital > 0:
                 daily_returns = daily_pnl / self.initial_capital
            else:
                 daily_returns = pd.Series(0.0, index=date_range)
        else:
            daily_returns = pd.Series(dtype=float)


        sharpe_ratio = self.calculate_sharpe_ratio(daily_returns, annualizing_factor) # Pass daily_returns
        sortino_ratio = self.calculate_sortino_ratio(daily_returns, annualizing_factor) # Pass daily_returns


        report_output += "\n" + "="*50 + "\n"
        report_output += "         INSIDE BAR BREAKOUT BACKTEST SUMMARY\n"
        report_output += "="*50 + "\n"
        report_output += f"Initial Capital: {self.initial_capital:,.2f}\n"
        report_output += f"Risk per Trade: {self.risk_per_trade_percent*100:.2f}%\n"
        report_output += f"Entry Timeframe: {self.entry_tf}\n"
        report_output += f"HTF Confirmation TFs: {self.higher_timeframes}\n"
        report_output += f"Symbols Tested: {len(self.ohlcv_data)}\n"
        report_output += f"Total Trades: {total_trades}\n"
        report_output += f"Winning Trades: {len(winning_trades)}\n"
        report_output += f"Losing Trades: {len(losing_trades)}\n"
        report_output += f"Win Rate: {win_rate * 100:.2f}%\n"
        report_output += f"Total PnL (Currency): {total_pnl:.2f}\n" # Updated PnL unit
        report_output += f"Maximum Drawdown (Currency): {max_dd:.2f}\n" # Display max drawdown
        report_output += f"Sharpe Ratio (Annualized): {sharpe_ratio:.2f}\n" # Display Sharpe Ratio
        report_output += f"Sortino Ratio (Annualized): {sortino_ratio:.2f}\n" # Display Sortino Ratio
        report_output += "="*50 + "\n"

        # Add Per-Symbol Summary
        per_symbol_metrics = self.calculate_per_symbol_metrics()
        if per_symbol_metrics:
            report_output += "\n" + "="*50 + "\n"
            report_output += "           PER-SYMBOL PERFORMANCE SUMMARY\n"
            report_output += "="*50 + "\n"
            for symbol, metrics in per_symbol_metrics.items():
                report_output += f"\n--- {symbol} ---\n"
                report_output += f"Total Trades: {metrics['Total Trades']}\n"
                report_output += f"Winning Trades: {metrics['Winning Trades']}\n"
                report_output += f"Losing Trades: {metrics['Losing Trades']}\n"
                report_output += f"Win Rate (%): {metrics['Win Rate (%)']:.2f}%\n"
                report_output += f"Total PnL (Currency): {metrics['Total PnL (Currency)']:.2f}\n"
                report_output += f"Maximum Drawdown (Currency): {metrics['Maximum Drawdown (Currency)']:.2f}\n"
                report_output += f"Sharpe Ratio (Annualized): {metrics['Sharpe Ratio (Annualized)']:.2f}\n"
                report_output += f"Sortino Ratio (Annualized): {metrics['Sortino Ratio (Annualized)']:.2f}\n"
                report_output += "-"*20 + "\n" # Separator for symbols

            report_output += "="*50 + "\n"


        return report_output


    def generate_report(self, save_to_files=False):
        """Generates consolidated performance reports and optionally saves them to files."""
        summary_report = self.generate_summary_report()
        detailed_report = self.generate_detailed_report()

        # Print reports to console
        print(summary_report)
        print(detailed_report)

        if save_to_files:
            summary_filename = "backtest_summary_report.txt"
            detailed_filename = "backtest_detailed_report.txt"

            with open(summary_filename, "w") as f:
                f.write(summary_report)
            print(f"\nSummary report saved to {summary_filename}")

            with open(detailed_filename, "w") as f:
                f.write(detailed_report)
            print(f"Detailed report saved to {detailed_filename}")


# --- EXECUTION ---

DATA_FOLDER = "/content"
DATA_FOLDER = "D:\\py_code_workspace\\NSE _STOCK _DATA"

# Initialize the strategy and backtester
backtester = InsideBarBreakoutStrategy(
    data_path=DATA_FOLDER,
    entry_timeframe='3min',
    risk_reward=3.5,
    max_workers=os.cpu_count(), # Use number of CPU cores for max workers
    initial_capital=1000000, # Initial capital
    risk_per_trade_percent=0.02 # 2% risk per trade
)

# Load all data files and create required timeframes
backtester.load_all_market_data()

# Run the bar-by-bar backtest
backtester.run_backtest()

# Generate the performance summary and detailed report and save them to files
backtester.generate_report(save_to_files=True)



import matplotlib.pyplot as plt
import pandas as pd

# Ensure trades are sorted by time for correct cumulative PnL calculation
sorted_trades = sorted(backtester.trades, key=lambda t: t.close_time if t.close_time else pd.to_datetime('9999-12-31'))

# Group trades by symbol
trades_by_symbol = {}
for trade in sorted_trades:
    if trade.symbol not in trades_by_symbol:
        trades_by_symbol[trade.symbol] = []
    trades_by_symbol[trade.symbol].append(trade)

# Plot equity curve for each symbol
for symbol, trades in trades_by_symbol.items():
    equity_curve_symbol = pd.DataFrame({
        'time': [trade.close_time for trade in trades],
        'pnl': [trade.pnl for trade in trades]
    })
    # Add initial capital to the first trade's PnL to start the curve from the initial capital
    if not equity_curve_symbol.empty:
        equity_curve_symbol['cumulative_pnl'] = equity_curve_symbol['pnl'].cumsum() + backtester.initial_capital


        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve_symbol['time'], equity_curve_symbol['cumulative_pnl'])
        plt.title(f'Equity Curve for {symbol}')
        plt.xlabel('Time')
        plt.ylabel('Cumulative PnL')
        plt.grid(True)
        plt.show()

    else:
        print(f"No closed trades to plot equity curve for {symbol}.")
        
        

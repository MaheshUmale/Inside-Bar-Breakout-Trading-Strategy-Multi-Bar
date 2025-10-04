import pandas as pd
import numpy as np
# In a real environment, replace these mock functions with your exchange/broker API calls
# and use a library like 'talib' for indicators.

# --- MOCK INDICATOR FUNCTIONS (For demonstration) ---

def calculate_atr(df, period=14):
    """Mocks ATR calculation."""
    df['TR'] = np.maximum.reduce([
        df['High'] - df['Low'],
        np.abs(df['High'] - df['Close'].shift(1)),
        np.abs(df['Low'] - df['Close'].shift(1))
    ])
    df['ATR'] = df['TR'].rolling(period).mean()
    return df['ATR'].iloc[-1] if not df.empty else 0

def calculate_ema(df, period=20):
    """Mocks EMA calculation."""
    return df['Close'].ewm(span=period, adjust=False).mean()

# --- CORE STRATEGY CLASS ---

class InsideBarBreakoutStrategy:
    """
    Implements the Multi-Timeframe, Volume-Confirmed Inside Bar Breakout Strategy.
    """
    def __init__(self, entry_timeframe='15m', max_ibr_lookback=5, risk_reward=1.5):
        self.entry_tf = entry_timeframe
        self.max_ibr_lookback = max_ibr_lookback
        self.risk_reward = risk_reward
        self.higher_timeframes = ['60m', '2h', '1D', '1W']
        self.tick_points = 2  # 2 Ticks / Points buffer for entry
        self.safety_buffer = 5  # 5 Ticks / Points buffer for SL beyond MB

    def load_mock_data(self, timeframe):
        """Mocks loading historical data for a given timeframe."""
        if timeframe == '1D':
            # Generate 100 days of mock data
            data = {
                'Open': np.random.uniform(100, 110, 100),
                'High': np.random.uniform(110, 120, 100),
                'Low': np.random.uniform(90, 100, 100),
                'Close': np.random.uniform(100, 110, 100),
                'Volume': np.random.randint(1000, 5000, 100)
            }
        else: # Mock smaller timeframes data
            data = {
                'Open': np.random.uniform(105, 115, 50),
                'High': np.random.uniform(115, 125, 50),
                'Low': np.random.uniform(95, 105, 50),
                'Close': np.random.uniform(105, 115, 50),
                'Volume': np.random.randint(500, 3000, 50)
            }
        df = pd.DataFrame(data)
        # Ensure High >= Close and Low <= Close for validity
        df['High'] = np.maximum(df['High'], df['Close'])
        df['Low'] = np.minimum(df['Low'], df['Close'])
        return df.iloc[1:] # Drop first row for safe shift operations

    def is_inside_bar(self, current_bar, mother_bar):
        """Checks if the current bar is fully contained within the mother bar."""
        return (current_bar['High'] < mother_bar['High']) and \
               (current_bar['Low'] > mother_bar['Low'])

    def check_mtf_trend(self):
        """
        Checks the trend on higher timeframes using EMA(20) slope and price position.
        Returns: 'UP', 'DOWN', or 'SIDEWAYS'
        """
        up_count = 0
        down_count = 0

        for tf in self.higher_timeframes:
            df = self.load_mock_data(tf)
            if df.empty or len(df) < 20: continue

            df['EMA'] = calculate_ema(df, period=20)
            
            # Trend Check: Close vs EMA AND EMA slope (last two values)
            current_close = df['Close'].iloc[-1]
            current_ema = df['EMA'].iloc[-1]
            prev_ema = df['EMA'].iloc[-2]

            if current_close > current_ema and current_ema > prev_ema:
                up_count += 1
            elif current_close < current_ema and current_ema < prev_ema:
                down_count += 1
        
        # Require trend alignment on at least one HTF
        if up_count > 0:
            return 'UP'
        elif down_count > 0:
            return 'DOWN'
        else:
            return 'SIDEWAYS'

    def find_inside_bar_breakout(self, df):
        """
        Identifies the Mother Bar and the Inside Bar Range (IBR) cluster.
        Returns: (is_valid, MB_Bar, IBR_High, IBR_Low)
        """
        if len(df) < 2:
            return False, None, None, None

        # Start checking from the current bar (index -1) backwards
        for i in range(1, len(df)):
            current_bar = df.iloc[-i]
            prev_bar = df.iloc[-i-1]

            if not self.is_inside_bar(current_bar, prev_bar):
                # The first non-inside bar is the Mother Bar (MB)
                MB_bar = prev_bar
                IBR_cluster = df.iloc[-(i):] # The bars that were inside the preceding MB

                if len(IBR_cluster) == 0:
                    continue # Not an IBR pattern yet

                # 1. Volume Validation (Contraction Filter)
                MB_volume = MB_bar['Volume']
                IBR_max_volume = IBR_cluster['Volume'].max()
                
                # Check if IBR volume is lower than MB volume
                if IBR_max_volume >= MB_volume:
                    print(f"DEBUG: IBR rejected due to high volume: MB={MB_volume}, IBR Max={IBR_max_volume}")
                    return False, None, None, None
                
                # 2. Define IBR range
                IBR_High = IBR_cluster['High'].max()
                IBR_Low = IBR_cluster['Low'].min()
                
                # We found a valid MB and IBR cluster
                return True, MB_bar, IBR_High, IBR_Low

        return False, None, None, None

    def execute_strategy(self, df_entry_tf):
        """Main function to run the strategy detection and order placement."""
        print(f"\n--- Checking {self.entry_tf} for Setup ---")

        # Step 1 & 2: Pattern Detection (MB, IBR, Volume)
        is_valid, mb_bar, ibr_high, ibr_low = self.find_inside_bar_breakout(df_entry_tf)

        if not is_valid:
            print("No valid Inside Bar Breakout pattern found with volume contraction.")
            return

        print("SUCCESS: Valid IBR Pattern Found.")
        print(f"Mother Bar Range: {mb_bar['Low']}-{mb_bar['High']}")
        print(f"IBR Trigger Range: {ibr_low}-{ibr_high}")
        
        # Step 3: MTF Confirmation
        mtf_trend = self.check_mtf_trend()
        print(f"MTF Trend Consensus: {mtf_trend}")

        # Calculate Risk and Trade Parameters
        risk_distance = mb_bar['High'] - mb_bar['Low'] # A simple risk calc placeholder

        # --- LONG SETUP ---
        long_entry = ibr_high + self.tick_points
        long_sl = mb_bar['Low'] - self.safety_buffer
        long_risk = long_entry - long_sl # R distance
        long_tp = long_entry + (long_risk * self.risk_reward)
        
        # --- SHORT SETUP ---
        short_entry = ibr_low - self.tick_points
        short_sl = mb_bar['High'] + self.safety_buffer
        short_risk = short_sl - short_entry # R distance
        short_tp = short_entry - (short_risk * self.risk_reward)

        # 4. Final Order Placement Decisions
        if mtf_trend == 'UP':
            print("\nPLACING LONG ORDER:")
            print(f"  Entry (Buy Stop): {long_entry:.2f}")
            print(f"  SL (Initial): {long_sl:.2f}")
            print(f"  TP (1.5R): {long_tp:.2f}")
            # In a live environment, you would call your API here: place_order('BUY_STOP', long_entry, long_sl, long_tp)
        elif mtf_trend == 'DOWN':
            print("\nPLACING SHORT ORDER:")
            print(f"  Entry (Sell Stop): {short_entry:.2f}")
            print(f"  SL (Initial): {short_sl:.2f}")
            print(f"  TP (1.5R): {short_tp:.2f}")
            # In a live environment, you would call your API here: place_order('SELL_STOP', short_entry, short_sl, short_tp)
        else:
            print("\nSIDEWAYS MTF TREND: No trade placed as trend alignment is required.")

        # Note: Trailing stop logic (Part B) must be handled by a separate function 
        # monitoring live tick data and position status after a trade is filled.

# --- SIMULATION EXAMPLE ---
if __name__ == '__main__':
    # Initialize the strategy
    strategy = InsideBarBreakoutStrategy(entry_timeframe='15m')
    
    # 1. Load data for the entry timeframe
    df_entry = strategy.load_mock_data(strategy.entry_tf)
    
    # Manually inject a strong pattern at the end for demonstration purposes
    # Mother Bar (MB) is the second to last bar (index -2)
    # Inside Bars are the last bar (index -1)
    
    # Ensure a clean Mother Bar at C-2
    MB_index = len(df_entry) - 2
    df_entry.loc[MB_index, 'High'] = 118.0
    df_entry.loc[MB_index, 'Low'] = 102.0
    df_entry.loc[MB_index, 'Volume'] = 5000 # High Volume MB

    # Ensure an Inside Bar (IB) at C-1
    IB_index = len(df_entry) - 1
    df_entry.loc[IB_index, 'High'] = 117.0
    df_entry.loc[IB_index, 'Low'] = 103.0
    df_entry.loc[IB_index, 'Volume'] = 1000 # Low Volume IB (Contraction)
    
    # Execute the strategy
    strategy.execute_strategy(df_entry)

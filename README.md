# 📊 Inside Bar Breakout Trading Strategy (Multi-Bar Optimized)

This document outlines a high-probability price action strategy based on market consolidation (Inside Bars) and confirmed by volume dynamics. This strategy is optimized to detect strong expansionary moves following a period of compression.

## 1. Pattern Definition & Identification

The core pattern involves a "Mother Bar" followed by one or more "Inside Bars," forming the **Inside Bar Range (IBR)**.

| **Component** | **Definition** |
| :--- | :--- |
| **Mother Bar (MB)** | A large bar showing high volatility and a strong directional move. |
| **Inside Bar (IB)** | Any subsequent bar whose entire range (High and Low) is contained within the $MB$'s range. |
| **Inside Bar Range (IBR)** | The tightest consolidation range (highest high and lowest low) formed by all consecutive Inside Bars. This is our key trigger level. |

### 🔍 Crucial Volume Rules

* **Setup Filter (Contraction):** The volume during the $IBR$ consolidation period must generally be **lower** than the $MB$ volume, signaling decreased market participation and indecision (the classic coil).

* **Trigger Confirmation (Expansion):** The actual price breakout above $IBR_{High}$ or below $IBR_{Low}$ **must** be accompanied by a noticeable **spike in volume** to confirm strong institutional support for the new direction.

## 2. Trading Parameters (The Six Key Steps)

| **Parameter** | **Type** | **Details** |
| :--- | :--- | :--- |
| **1. SETUP** | Pattern | Valid $MB$ followed by 1 to 5 inside bars ($IBR$). Confirmed by **low volume** during the $IBR$. |
| **2. TRIGGER** | Price | Price movement of **1-3 Ticks / Points** beyond the $IBR$ **extreme**. |
| **3. ENTRY** | Order Type | Place a **Buy Stop** (Long) or **Sell Stop** (Short) pending order. |
| | **Long Entry** | $IBR_{High} + 2$ Ticks. |
| | **Short Entry** | $IBR_{Low} - 2$ Ticks. |
| **4. SL (Stop Loss)** | Invalidation | Set SL beyond the opposite extreme of the Mother Bar (MB). |
| | **Long SL** | $MB_{Low} - 5$ Ticks (safeguard). |
| | **Short SL** | $MB_{High} + 5$ Ticks (safeguard). |
| **5. TP (Take Profit)** | R:R Target | Initial target set at a minimum **1.5R** (Risk-to-Reward ratio). $R$ is the calculated Risk Distance, defined as $|\text{Entry} - \text{SL}|$. |
| **6. TRAILING** | Management | Move SL to **Break-Even** at 1R profit. Trail subsequently using the $2 \times ATR(14)$ method to capture extended moves. |

## 3. Algorithm High-Level Logic

The automated strategy logic is divided into two primary loops:

### A. Pattern Detection & Order Placement (Bar Close)

1. **Identify MB:** The Mother Bar ($C_{MB}$) is typically the bar immediately preceding the consolidation cluster ($C_{-2}$ relative to current bar $C_0$).

2. **Calculate IBR:** Check up to 5 subsequent bars, verifying they are all inside the $MB$. Determine the $IBR_{High}$ and $IBR_{Low}$ (the extremes of this consolidation cluster).

3. **Validate Volume:** Ensure the maximum volume within the IBR cluster is less than the Mother Bar volume.

4. **Place Pending Orders:** If the pattern is valid, place a Buy Stop order at $IBR_{High} + 2$ ticks and a Sell Stop order at $IBR_{Low} - 2$ ticks, using $MB$ extremes for the Stop Loss placement.

### B. Trade Management (Tick Update)

1. **Monitor Activation:** Wait for one of the pending orders to be filled (the breakout).

2. **Move to Break-Even (BE):** Once the trade has moved in profit by $1R$ (the distance from Entry to Initial SL), immediately move the Stop Loss to the Entry Price (BE).

3. **Trail Stop (ATR):** Once at BE, use the $2 \times ATR(14)$ calculation to dynamically move the Stop Loss, locking in profit as the expansion continues.

## 4. Backtesting, Reporting, and Visualization

The script includes a comprehensive backtesting engine that evaluates the strategy's performance across multiple symbols and generates detailed reports and visualizations.

### A. Backtesting Execution

The backtest is executed by running the `InsideBarBACKTEST.py` script. The script will:
1.  Load all `_minute.csv` data files from the root directory.
2.  Resample the data to the required timeframes.
3.  Execute the backtest logic for each symbol in parallel.
4.  Generate and save reports and charts.

### B. Reporting Suite

All generated reports and charts are saved in the `REPORTS/` directory. The following reports are generated:

*   **Consolidated Summary Report:** A single file (`consolidated_summary_report.txt`) that provides a high-level overview of the strategy's performance across all tested symbols.
*   **Per-Symbol Summary Reports:** Individual summary reports for each symbol (e.g., `360ONE_summary_report.txt`), detailing performance metrics for that specific symbol.
*   **Detailed Trade Report:** A comprehensive log of all trades executed during the backtest (`detailed_trade_report.txt`).

### C. Trade Visualization

To provide a visual representation of the strategy's performance, the script generates candlestick charts for each symbol with trades overlaid.

*   **Time-Based Grouping:** Trades are grouped into approximately 2-month intervals to ensure charts are clear and readable.
*   **Trade Markers:**
    *   **Entry:** Long entries are marked with a green upward-pointing triangle (`^`), and short entries are marked with a red downward-pointing triangle (`v`).
    *   **Exit:** Trade exits are marked with a blue 'x'.
    *   **Stop Loss (SL):** The initial SL is shown as a red line.
    *   **Take Profit (TP):** The TP is shown as a green line.
    *   **Trailing SL:** Adjustments to the trailing SL are marked with orange horizontal lines.
*   **Chart Files:** The charts are saved as PNG files in the `REPORTS/` directory, with one file per time-based group for each symbol (e.g., `360ONE_trades_1.png`).

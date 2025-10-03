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

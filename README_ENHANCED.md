# Hybrid Options Pricing & Analysis Tool - Enhanced Version

## 🎯 Overview

This enhanced version of the Hybrid Options Pricing Tool adds powerful visual analytics to help you better understand option risks and rewards. The tool now includes:

- **📊 Interactive Payoff Diagrams**: Visual P&L scenarios at expiration
- **💰 Premium Breakdown**: Intrinsic vs Time Value analysis with Breakeven prices
- **📈 Greeks Sensitivity Charts**: Dynamic visualization of how Greeks change with price
- **🎯 Enhanced Risk Analysis**: Comprehensive market sentiment and valuation insights

## ✨ New Features

### 1. **Payoff Diagram**
Visual representation of profit/loss at expiration for both Call and Put options:
- Shows breakeven points clearly marked
- Indicates current underlying price and strike price
- Color-coded profit/loss zones
- Interactive hover tooltips with exact values

### 2. **Enhanced Pricing Table**
Now includes additional metrics:
- **Intrinsic Value**: The option's immediate exercise value
- **Time Value**: Premium paid for uncertainty (Extrinsic Value)
- **Breakeven Price**: Exact price needed at expiration to break even
- All original metrics retained (Theoretical Price, Market Premium, Mispricing, etc.)

### 3. **Greeks Sensitivity Analysis**
Two comprehensive charts showing:
- **Delta & Gamma**: How position exposure changes with price
- **Vega & Theta**: Volatility risk and time decay across prices
- Visual identification of maximum risk zones
- Strike price and current price markers

## 📁 File Structure

```
your_project/
│
├── option_pricing_core.py          # Original core pricing module (unchanged)
├── option_visuals.py               # NEW: Visualization functions
├── streamlit_app_enhanced.py       # NEW: Enhanced Streamlit UI
└── README_ENHANCED.md              # This file
```

## 🚀 Installation

### Prerequisites
Make sure you have Python 3.8+ installed.

### Step 1: Install Required Packages

```bash
pip install streamlit pandas numpy scipy plotly
```

### Step 2: Organize Your Files

1. Keep your existing `option_pricing_core.py` file (no changes needed)
2. Add the new `option_visuals.py` file
3. Replace your Streamlit app with `streamlit_app_enhanced.py`

## 🎮 How to Run

### Launch the Application

```bash
streamlit run streamlit_app_enhanced.py
```

The app will open in your default browser at `http://localhost:8501`

## 📖 User Guide

### Input Parameters (Sidebar)

1. **Asset Type**: Select Index, Stock, or Commodity
2. **Asset Name/Ticker**: Enter the asset name (e.g., NIFTY, RELIANCE)
3. **Strike Price (K)**: The option's strike price in ₹
4. **Days to Expiration**: Days until option expiry
5. **Risk-Free Rate (r)**: Annual risk-free rate (e.g., 0.05 for 5%)
6. **User Volatility (σ)**: Your estimated annual volatility (e.g., 0.20 for 20%)
7. **Market Premiums**: Current market prices for Call and Put options
8. **Underlying Price**: Current spot/futures price
9. **Dividend Yield (q)**: For stocks/indices only
10. **PCR Ratio**: Put-Call Ratio for sentiment analysis

### Understanding the Output

#### 1. **Input Summary**
Quick reference of all parameters you entered

#### 2. **Payoff Diagram** 🆕
- **Green Line**: Call option P&L
- **Red Line**: Put option P&L
- **Diamond Markers**: Breakeven points
- **Purple Dashed Line**: Strike price
- **Orange Dashed Line**: Current price
- **Shaded Areas**: Profit (green) and Loss (red) zones

**Key Insight**: This shows your maximum risk (the premium paid) and unlimited profit potential for calls (or limited to strike for puts).

#### 3. **Volatility Analysis**
- **User σ vs Market IV**: Compare your volatility estimate to market expectations
- **IV Skew Detection**: Identifies pricing imbalances between calls and puts
- **Actionable Insight**: Whether options are overvalued or undervalued

#### 4. **Enhanced Pricing Analysis** 🆕

The table now shows:

| Metric | What It Means |
|--------|---------------|
| **Intrinsic Value** | Value if exercised now: max(0, S-K) for calls, max(0, K-S) for puts |
| **Time Value** | Premium for time = Market Premium - Intrinsic Value |
| **Breakeven Price** | Price needed at expiry to break even: K + Premium (call) or K - Premium (put) |

**Example Interpretation**:
```
Call Premium: ₹5.00
Intrinsic Value: ₹0.00 (OTM)
Time Value: ₹5.00 (all premium is time value)
Breakeven: ₹105.00 (if Strike = ₹100)
```
This means the underlying must rise above ₹105 at expiration for profit.

#### 5. **Greeks Comparison Table**
Side-by-side comparison of Greeks at:
- Your volatility estimate (σ)
- Market implied volatility (IV)

#### 6. **Greeks Sensitivity Charts** 🆕

**Delta & Gamma Chart**:
- **Delta**: Shows how much the option price changes with ₹1 move in underlying
  - ATM options: Delta ≈ 0.5 (50% of underlying move)
  - ITM options: Delta → 1.0 (moves 1:1 with underlying)
  - OTM options: Delta → 0 (little movement)
- **Gamma**: Peaks at ATM, shows where Delta changes fastest
  - High Gamma = High risk/reward (position delta changes rapidly)

**Vega & Theta Chart**:
- **Vega**: Peaks at ATM, shows volatility sensitivity
  - High Vega = Profits if IV increases, loses if IV decreases
- **Theta**: Most negative at ATM, shows time decay
  - Accelerates near expiration
  - Always negative for long options (you lose money daily)

#### 7. **Greeks Interpretation**
Detailed risk assessment in plain language:
- Gamma: How stable/responsive your position is
- Vega: Your exposure to volatility changes
- Theta: Daily time decay impact
- Rho: Interest rate sensitivity

#### 8. **Market Sentiment**
PCR-based analysis:
- **Bullish**: PCR > 1.10 (heavy put buying = hedging = market may rise)
- **Bearish**: PCR < 0.90 (heavy call buying = speculation = vulnerable to correction)
- **Neutral**: Balanced positioning

## 💡 Practical Trading Applications

### For Option Buyers

1. **Check Payoff Diagram**: Understand your maximum loss (premium paid) and profit potential
2. **Analyze Time Value**: High time value? You're paying for uncertainty - decay works against you
3. **Monitor Theta**: How much you lose per day - accelerates near expiry
4. **Check IV vs Your σ**: If Market IV > Your σ, options are expensive (consider waiting)
5. **Find Breakeven**: Set realistic targets based on breakeven price

### For Option Sellers

1. **Payoff Diagram**: Shows your maximum profit (premium collected) and loss exposure
2. **Theta is Your Friend**: You profit from time decay (positive theta for short positions)
3. **Gamma Risk**: High gamma near ATM means position delta changes fast - risky
4. **Vega Risk**: Short options lose if IV spikes - watch vega exposure
5. **Breakeven Buffer**: Your profit zone is between breakevens

### For Volatility Traders

1. **IV Skew**: Exploit pricing inefficiencies between calls and puts
2. **Vega Sensitivity Chart**: Identify strikes with maximum volatility exposure
3. **User σ vs Market IV**: Trade mean reversion when IV deviates from historical
4. **Time Value Ratio**: High time value % = potential volatility trading opportunity

## 🎨 Customization

### Adjusting Chart Ranges

In `option_visuals.py`, modify the price range:

```python
# For payoff diagram (default ±40%)
price_range = np.linspace(K * 0.6, K * 1.4, 200)

# For Greeks sensitivity (default ±30%)
price_range = np.linspace(F_or_S * 0.7, F_or_S * 1.3, 100)
```

### Changing Color Schemes

In `streamlit_app_enhanced.py`, customize colors:

```python
# CSS color variables
.main-header { color: #1f77b4; }  # Blue header
.sub-header { color: #ff7f0e; }   # Orange subheader
```

In `option_visuals.py`:

```python
line=dict(color='#2ecc71', width=3),  # Call line (green)
line=dict(color='#e74c3c', width=3),  # Put line (red)
```

## 🔧 Troubleshooting

### Charts Not Displaying
- Ensure `plotly` is installed: `pip install plotly`
- Clear Streamlit cache: Click "Clear cache" in the hamburger menu

### Calculation Errors
- Verify all inputs are positive (except q and r can be 0)
- Days to expiry must be ≥ 1
- Volatility should be reasonable (0.01 to 2.0)

### Performance Issues
- Reduce Greek sensitivity chart resolution:
  ```python
  price_range = np.linspace(F_or_S * 0.7, F_or_S * 1.3, 50)  # Reduce from 100 to 50
  ```

## 📊 Example Use Case

### Scenario: NIFTY Call Option Analysis

**Inputs**:
- Asset: NIFTY (Index)
- Strike: ₹26,000
- Days to Expiry: 6 days
- Risk-Free Rate: 5%
- User Volatility: 11.30%
- Call Premium: ₹109.01
- Spot Price: ₹25,868.60

**Analysis**:

1. **Payoff Diagram** shows:
   - Breakeven at ₹26,109.01 (₹26,000 + ₹109.01)
   - Maximum loss: ₹109.01 (limited to premium)
   - Current price ₹131.40 away from breakeven

2. **Premium Breakdown**:
   - Intrinsic Value: ₹0 (OTM - spot below strike)
   - Time Value: ₹109.01 (100% of premium)
   - This is pure time value - will decay to zero by expiry

3. **Greeks Sensitivity**:
   - Delta peaks around ₹26,000 (ATM region)
   - Gamma highest at strike (rapid delta changes)
   - Theta most negative at ATM (fastest decay)

4. **IV Analysis**:
   - Market IV: 11.91% (higher than your 11.30%)
   - Option slightly overvalued based on your estimate
   - IV skew: Check if call IV > put IV

**Trading Decision**:
- **Risk**: Limited to ₹109.01 per contract
- **Reward**: Unlimited above ₹26,109
- **Required Move**: +0.93% (₹241 from current spot) to breakeven
- **Time Decay**: Losing value daily (check Theta)
- **Verdict**: With only 6 days left, need sharp rally for profit. High time decay risk.

## 🆕 Changelog - v2.2 (Enhanced)

### Added
- Interactive payoff diagrams showing P&L at expiration
- Intrinsic value and time value breakdown
- Breakeven price calculation and display
- Greeks sensitivity charts (Delta, Gamma, Vega, Theta vs price)
- Enhanced pricing table with premium components
- Detailed explanatory tooltips and guides

### Maintained
- All original calculations (Black-Scholes-Merton, Black-76)
- Greeks calculation accuracy
- Implied volatility solver
- Market sentiment analysis
- Input validation
- State management

### Files
- **New**: `option_visuals.py` (visualization module)
- **Updated**: `streamlit_app_enhanced.py` (enhanced UI)
- **Unchanged**: `option_pricing_core.py` (core pricing engine)

## 📝 Notes

- All calculations remain identical to the original version
- The enhanced version only adds visualization layers
- Performance impact is minimal (charts render in <1 second)
- Works with all asset types (Index, Stock, Commodity)
- Compatible with existing workflows

## ⚠️ Disclaimer

This tool is for educational and analytical purposes only. It is NOT financial advice. Options trading involves significant risk and is not suitable for all investors. Always consult with a qualified financial advisor before making trading decisions.

---

**Version**: 2.2 (Enhanced with Visualizations)  
**Last Updated**: October 2025  
**Author**: Built with ❤️ for options traders  
**License**: Educational Use Only

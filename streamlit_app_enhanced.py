"""
Streamlit Interactive UI for Hybrid Option Pricing Tool
(Enhanced with Visual Payoff Diagrams and Premium Breakdown)
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from option_pricing_core import get_analysis_results, HybridAnalysisConfig 
from option_visuals import (
    create_payoff_diagram,
    create_greeks_sensitivity_chart,
    get_enhanced_pricing_table
)

# Page configuration
st.set_page_config(
    page_title="Hybrid Option Pricing Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better table styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stDataFrame {
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">🎯 Hybrid Options Pricing & Analysis Tool</div>', unsafe_allow_html=True)

# --- Helper for Input Hashing ---
def get_input_hash(current_params):
    """Generates a hash of all essential inputs to detect changes."""
    param_string = "|".join([str(current_params[k]) for k in sorted(current_params.keys())])
    return hashlib.md5(param_string.encode()).hexdigest()

# ========== SIDEBAR INPUTS ==========
with st.sidebar:
    st.header("📊 Input Parameters")
    
    asset_type = st.selectbox("Asset Type", options=['Index', 'Stock', 'Commodity'])
    asset_name = st.text_input("Asset Name/Ticker", value="NIFTY")
    
    st.divider()
    st.subheader("Common Parameters")
    
    K = st.number_input("Strike Price (K)", min_value=0.01, value=26000.0, step=100.0, format="%.2f")
    Days_to_expiry = st.number_input("Days to Expiration", min_value=1, value=6, step=1)
    r = st.number_input("Risk-Free Rate (r) [e.g., 0.05 for 5%]", min_value=-0.1, max_value=1.0, value=0.05, step=0.01, format="%.4f")
    sigma = st.number_input("User Defined Volatility (σ) [e.g., 0.20 for 20%]", min_value=0.01, max_value=2.0, value=0.1130, step=0.01, format="%.4f")
    
    st.divider()
    st.subheader("Market Option Premiums")
    
    call_market = st.number_input("Call Option Market Premium", min_value=0.0, value=109.01, step=0.1, format="%.2f")
    put_market = st.number_input("Put Option Market Premium", min_value=0.0, value=180.95, step=0.1, format="%.2f")
    
    st.divider()
    st.subheader("Asset-Specific Parameters")
    
    if asset_type in ['Stock', 'Index']:
        S = st.number_input("Current Spot Price (S)", min_value=0.01, value=25868.6, step=1.0, format="%.2f")
        q = st.number_input("Dividend Yield (q) [e.g., 0.01 for 1%]", min_value=0.0, max_value=1.0, value=0.0001, step=0.001, format="%.4f")
        F_or_S = S
    else:
        F = st.number_input("Current Futures Price (F)", min_value=0.01, value=100.0, step=1.0, format="%.2f")
        q = 0.0
        F_or_S = F

    st.divider()
    st.subheader("Market Sentiment")
    
    PCR_Ratio = st.number_input("Put/Call Ratio (PCR)", min_value=0.0, value=0.79, step=0.1, format="%.2f")
    
    st.divider()
    
    calculate_button = st.button("🔍 Calculate & Analyze", type="primary", use_container_width=True)

# Collect all current parameters for hashing
current_params = {
    'asset_type': asset_type, 'K': K, 'Days_to_expiry': Days_to_expiry, 'r': r, 
    'sigma_user': sigma, 'call_market': call_market, 'put_market': put_market, 
    'PCR_Ratio': PCR_Ratio, 'F_or_S': F_or_S, 'q': q
}
current_hash = get_input_hash(current_params)

# Determine if a recalculation is necessary
recalculate_needed = (
    calculate_button or 
    'results' not in st.session_state or
    st.session_state.get('params_hash') != current_hash
)

# ========== MAIN PAGE OUTPUTS ==========
if recalculate_needed:
    try:
        # Call the robust backend function
        results = get_analysis_results(
            asset_ticker=asset_name, **current_params
        )
        st.session_state.results = results
        st.session_state.params_hash = current_hash
        st.session_state.results_calculated = True
        
    except ValueError as e:
        st.error(f"❌ **Validation Error:**\n\n{str(e)}")
        st.session_state.results_calculated = False
    except Exception as e:
        st.error(f"❌ **Calculation Error:** An unexpected error occurred. Please check inputs.\n\n{str(e)}")
        st.session_state.results_calculated = False

# Display results if calculated successfully
if st.session_state.get('results_calculated', False):
    results = st.session_state.results
    input_summary = results['input_summary']
    
    # ========== SECTION 1: INPUT SUMMARY ==========
    st.markdown(f'<div class="sub-header">📋 Summary of Inputs for {asset_name.upper()}</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Strike Price (K)", f"₹{input_summary['Strike Price (K)']:,}")
        st.metric("Days to Expiry", f"{input_summary['Time to Expiry (Days)']:.0f} days")
    
    with col2:
        st.metric("Risk-Free Rate (r)", input_summary['Risk-Free Rate (r)'])
        st.metric("User Volatility (σ)", input_summary['User Volatility (σ)'])
    
    with col3:
        st.metric("Call Market Premium", f"₹{input_summary['Call Market Premium']:,.2f}")
        st.metric("Put Market Premium", f"₹{input_summary['Put Market Premium']:,.2f}")
    
    with col4:
        st.metric("Underlying Price (S/F)", f"₹{input_summary['Underlying Price (S/F)']:,}")
        st.metric("Dividend Yield (q)", input_summary['Dividend Yield (q)'])

    # ========== NEW SECTION: PAYOFF DIAGRAM ==========
    st.markdown('<div class="sub-header">📈 Option Payoff Diagram</div>', unsafe_allow_html=True)
    
    try:
        payoff_fig = create_payoff_diagram(
            K=K,
            call_market=call_market,
            put_market=put_market,
            F_or_S=F_or_S,
            asset_name=asset_name
        )
        st.plotly_chart(payoff_fig, use_container_width=True)
        
        # Add explanatory text
        st.info("""
        **How to Read This Chart:**
        - **Green Line (Call)**: Profit/Loss if you buy a Call option
        - **Red Line (Put)**: Profit/Loss if you buy a Put option
        - **Diamond Markers**: Breakeven points where P&L = 0
        - **Purple Dashed Line**: Strike price
        - **Orange Dashed Line**: Current underlying price
        - **Shaded Regions**: Green = Profit zone, Red = Loss zone
        """)
        
    except Exception as e:
        st.error(f"Error generating payoff diagram: {str(e)}")

    # ========== SECTION 2: VOLATILITY ANALYSIS ==========
    st.markdown('<div class="sub-header">📊 Implied Volatility (IV) Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    sigma_user = results['sigma_user']
    
    with col1:
        st.metric("User Defined σ", f"{sigma_user:.2%}")
    with col2:
        iv_diff_call = ((results['IV_call'] - sigma_user) / sigma_user) * 100
        st.metric("Call IV", f"{results['IV_call']:.2%}", delta=f"{iv_diff_call:+.1f}%")
    with col3:
        iv_diff_put = ((results['IV_put'] - sigma_user) / sigma_user) * 100
        st.metric("Put IV", f"{results['IV_put']:.2%}", delta=f"{iv_diff_put:+.1f}%")
    with col4:
        st.metric("Average IV", f"{results['IV_avg']:.2%}")
    
    # IV Skew/Actionable Insight
    iv_diff = results['IV_call'] - results['IV_put']
    if abs(iv_diff) > 0.01:
        skew_type = "Smirk (higher Call IV)" if iv_diff > 0 else "Skew (higher Put IV)"
        st.warning(f"⚠️ **Volatility Skew Detected ({skew_type})** \nIV Difference: {iv_diff:+.2%} (Indicates pricing imbalance)")
    
    if results['IV_avg'] > (sigma_user * (1 + HybridAnalysisConfig.IV_DIFFERENCE_PCT)):
        st.info(f"💡 **Market IV ({results['IV_avg']:.2%})** is significantly **HIGHER** than your input σ ({sigma_user:.2%}). "
                "Options are currently **OVERVALUED** based on your estimated risk.")
    elif results['IV_avg'] < (sigma_user * (1 - HybridAnalysisConfig.IV_DIFFERENCE_PCT)):
        st.info(f"💡 **Market IV ({results['IV_avg']:.2%})** is significantly **LOWER** than your input σ ({sigma_user:.2%}). "
                "Options are currently **UNDERVALUED** based on your estimated risk.")

    # ========== SECTION 3: ENHANCED PRICING ANALYSIS ==========
    st.markdown('<div class="sub-header">💰 Option Pricing Analysis (Enhanced)</div>', unsafe_allow_html=True)
    
    try:
        # Get enhanced pricing table with intrinsic/time value breakdown
        enhanced_pricing_df = get_enhanced_pricing_table(
            results=results,
            call_market=call_market,
            put_market=put_market,
            F_or_S=F_or_S,
            K=K
        )
        
        st.dataframe(enhanced_pricing_df, use_container_width=True, hide_index=True)
        
        # Add explanatory notes
        with st.expander("📚 Understanding the Pricing Metrics"):
            st.markdown("""
            **Intrinsic Value**: The option's value if exercised immediately
            - Call: max(0, Current Price - Strike)
            - Put: max(0, Strike - Current Price)
            
            **Time Value (Extrinsic Value)**: Premium paid for time until expiration
            - Formula: Market Premium - Intrinsic Value
            - Represents uncertainty and potential for profit before expiry
            - Decays as expiration approaches (measured by Theta)
            
            **Breakeven Price**: The underlying price at expiration where you break even
            - Call Breakeven = Strike + Call Premium
            - Put Breakeven = Strike - Put Premium
            - Price must move beyond this point for profit
            """)
            
    except Exception as e:
        st.error(f"Error generating enhanced pricing table: {str(e)}")

    # ========== SECTION 4: GREEKS COMPARISON TABLE ==========
    st.markdown('<div class="sub-header">📊 Option Greeks Analysis (Comparison)</div>', unsafe_allow_html=True)
    st.dataframe(results['greeks_df'], use_container_width=True, hide_index=True)

    # ========== NEW SECTION: GREEKS VISUALIZATION ==========
    st.markdown('<div class="sub-header">📉 Greeks Sensitivity Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    The charts below show how option Greeks change as the underlying price moves. 
    This helps visualize risk exposure across different price levels.
    """)
    
    try:
        # Prepare parameters for Greeks visualization
        S = F_or_S if asset_type in ['Index', 'Stock'] else 0.0
        F = F_or_S if asset_type == 'Commodity' else 0.0
        T = Days_to_expiry / 365
        
        # Create Greeks sensitivity charts
        fig_delta_gamma, fig_vega_theta = create_greeks_sensitivity_chart(
            S=S,
            F=F,
            K=K,
            T=T,
            r=r,
            sigma=results['IV_avg'],  # Use average market IV
            q=q,
            asset_type=asset_type,
            asset_name=asset_name
        )
        
        # Display in tabs for better organization
        tab1, tab2 = st.tabs(["📊 Delta & Gamma", "📈 Vega & Theta"])
        
        with tab1:
            st.plotly_chart(fig_delta_gamma, use_container_width=True)
            st.info("""
            **Delta**: Rate of change of option price with respect to underlying price
            - Call Delta: 0 to 1 (positive exposure)
            - Put Delta: -1 to 0 (negative exposure)
            - ATM options have Delta ≈ ±0.5
            
            **Gamma**: Rate of change of Delta (curvature of option price)
            - Peaks at ATM (strike price)
            - Measures how quickly Delta changes
            - High Gamma = High risk/reward potential
            """)
        
        with tab2:
            st.plotly_chart(fig_vega_theta, use_container_width=True)
            st.info("""
            **Vega**: Sensitivity to volatility changes
            - Peaks at ATM
            - Measures profit/loss from 1% IV change
            - Long options = positive Vega (benefit from rising IV)
            
            **Theta**: Time decay (loss per day)
            - Always negative for long options
            - Accelerates near expiration
            - ATM options have highest absolute Theta
            """)
            
    except Exception as e:
        st.error(f"Error generating Greeks sensitivity charts: {str(e)}")

    # ========== SECTION 5: GREEKS INTERPRETATION ==========
    st.markdown('<div class="sub-header">🔍 Greeks Interpretation & Risk Assessment</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📍 Market Risk Profile (at IV)", "📊 Estimated Risk Profile (at User σ)"])
    
    with tab1:
        st.markdown("**Based on current market implied volatility:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Call Option Greeks**")
            st.info(f"**Gamma:** {HybridAnalysisConfig.get_gamma_implication(results['gamma_call_iv'])}")
            st.info(f"**Vega:** {HybridAnalysisConfig.get_vega_implication(results['vega_call_iv'], results['F_or_S'])}")
            st.info(f"**Theta:** {HybridAnalysisConfig.get_theta_implication(results['daily_theta_call_iv'], results['model_call_iv'])}")
            st.info(f"**Rho:** {HybridAnalysisConfig.get_rho_implication(results['rho_call_iv'], results['F_or_S'])}")
        
        with col2:
            st.markdown("**Put Option Greeks**")
            st.info(f"**Gamma:** {HybridAnalysisConfig.get_gamma_implication(results['gamma_put_iv'])}")
            st.info(f"**Vega:** {HybridAnalysisConfig.get_vega_implication(results['vega_put_iv'], results['F_or_S'])}")
            st.info(f"**Theta:** {HybridAnalysisConfig.get_theta_implication(results['daily_theta_put_iv'], results['model_put_iv'])}")
            st.info(f"**Rho:** {HybridAnalysisConfig.get_rho_implication(results['rho_put_iv'], results['F_or_S'])}")
    
    with tab2:
        st.markdown("**Based on your estimated volatility:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Call Option Greeks**")
            st.success(f"**Gamma:** {HybridAnalysisConfig.get_gamma_implication(results['gamma_user'])}")
            st.success(f"**Vega:** {HybridAnalysisConfig.get_vega_implication(results['vega_user'], results['F_or_S'])}")
            st.success(f"**Theta:** {HybridAnalysisConfig.get_theta_implication(results['daily_theta_call_user'], results['model_call_user'])}")
            st.success(f"**Rho:** {HybridAnalysisConfig.get_rho_implication(results['rho_call_user'], results['F_or_S'])}")
        
        with col2:
            st.markdown("**Put Option Greeks**")
            st.success(f"**Gamma:** {HybridAnalysisConfig.get_gamma_implication(results['gamma_user'])}")
            st.success(f"**Vega:** {HybridAnalysisConfig.get_vega_implication(results['vega_user'], results['F_or_S'])}")
            st.success(f"**Theta:** {HybridAnalysisConfig.get_theta_implication(results['daily_theta_put_user'], results['model_put_user'])}")
            st.success(f"**Rho:** {HybridAnalysisConfig.get_rho_implication(results['rho_put_user'], results['F_or_S'])}")

    # ========== SECTION 6: MARKET SENTIMENT ==========
    st.markdown('<div class="sub-header">🎯 Overall Market Sentiment Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Put/Call Ratio (PCR)", f"{PCR_Ratio:.2f}")
    with col2:
        pcr_sentiment = results['pcr_sentiment']
        sentiment_color = "🟢" if "Bullish" in pcr_sentiment else "🔴" if "Bearish" in pcr_sentiment else "🟡"
        st.metric("Market Sentiment", f"{sentiment_color} {pcr_sentiment}")
    
    # Sentiment interpretation
    if "Bullish" in pcr_sentiment:
        st.success("📈 **Bullish Sentiment**: The higher PCR suggests heavy Put buying (hedging or bearish bets). Often, when everyone is hedged, the market is positioned for a rise.")
    elif "Bearish" in pcr_sentiment:
        st.error("📉 **Bearish Sentiment**: The lower PCR suggests heavy Call buying (speculation). Excessive bullishness can make the market vulnerable to a correction.")
    else:
        st.info("⚖️ **Neutral Sentiment**: Balanced put-call ratio indicates market equilibrium or indecision.")
            
else:
    # Initial state or error state
    st.info("👈 **Please enter your parameters in the sidebar and click 'Calculate & Analyze' to view results.**")
    st.markdown("### 📖 How to Use This Tool")
    st.markdown("""
    1. **Select Asset Type**: Choose between Index, Stock, or Commodity.
    2. **Enter Parameters**: Fill in the required fields in the sidebar, including your personal volatility estimate (User σ).
    3. **Market Prices**: Input current market option premiums to allow for Implied Volatility (IV) calculation.
    4. **Calculate**: Click the "Calculate & Analyze" button.
    5. **Analyze Results**: Review:
       - **Payoff Diagrams**: Visual representation of profit/loss at expiration
       - **Premium Breakdown**: Intrinsic vs Time Value analysis
       - **Greeks Sensitivity**: How risk factors change with price
       - **Market Sentiment**: PCR-based directional bias
    
    ### ✨ New Features in This Version:
    - 📊 **Interactive Payoff Diagrams**: See profit/loss scenarios at expiration
    - 💰 **Premium Breakdown**: Understand Intrinsic vs Time Value components
    - 📈 **Greeks Visualization**: Dynamic charts showing risk exposure across prices
    - 🎯 **Breakeven Analysis**: Know exactly where you profit
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>Built with ❤️ using Streamlit | Hybrid Option Pricing Model v2.2 (Enhanced with Visualizations)</p>
    <p>⚠️ For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

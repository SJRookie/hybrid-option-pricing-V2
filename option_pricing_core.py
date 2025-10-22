
"""
Core option pricing functions - Black-Scholes-Merton and Black-76 models
with Greeks calculations, implied volatility, and master analysis wrapper.

Includes numerical stability enhancements based on code review.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


# --- CONFIGURATION & NUMERICAL CONSTANTS ---
class HybridAnalysisConfig:
    """Hybrid thresholds and numerical constants for the model."""
    
    # Numerical Constants
    IV_TOLERANCE = 1e-6          # Convergence tolerance for IV solver
    IV_MAX_ITERATIONS = 150      # Max iterations for Newton-Raphson
    EPSILON_DENOMINATOR = 1e-10  # Safety floor for denominators in Greeks/Pricing

    # Valuation thresholds (absolute)
    OVERVALUED_THRESHOLD = 0.50
    UNDERVALUED_THRESHOLD = -0.50
    
    # PCR sentiment thresholds
    PCR_BULLISH = 1.10
    PCR_BEARISH = 0.90
    
    # Greeks thresholds
    GAMMA_HIGH_ABSOLUTE = 0.005
    VEGA_HIGH_PCT = 0.015  # 1.5% of underlying
    RHO_SIGNIFICANT_PCT = 0.001  # 0.1%
    THETA_SIGNIFICANT_PCT = 0.05  # 5% of option price
    
    # IV comparison threshold
    IV_DIFFERENCE_PCT = 0.03
    
    # Moneyness thresholds
    DELTA_DEEP_ITM = 0.75
    DELTA_ATM_LOWER = 0.30
    
    # Interpretation methods (as before)
    @staticmethod
    def get_gamma_implication(gamma):
        """Determine delta risk based on Gamma absolute threshold"""
        if gamma > HybridAnalysisConfig.GAMMA_HIGH_ABSOLUTE:
            return "Highly Responsive/Volatile: Delta changes rapidly with underlying moves."
        else:
            return "Stable/Less Responsive: Delta changes slowly with underlying moves."
    
    @staticmethod
    def get_vega_implication(vega, underlying_price):
        """Get Vega interpretation using normalized threshold"""
        vega_pct = (abs(vega) / underlying_price) * 100
        
        if vega_pct > HybridAnalysisConfig.VEGA_HIGH_PCT:
            return (f"Volatility Risk is High ({vega_pct:.3f}% of underlying).\n"
                    f"  → Vega: {vega:.2f}\n"
                    f"  → 1% IV change ≈ ₹{abs(vega)/100:.2f} price change")
        else:
            return (f"Volatility Risk is Moderate ({vega_pct:.3f}% of underlying).\n"
                    f"  → Vega: {vega:.2f}\n"
                    f"  → 1% IV change ≈ ₹{abs(vega)/100:.2f} price change")
    
    @staticmethod
    def get_theta_implication(theta, model_price):
        """Determine if Theta (time decay) is significant based on % of option price"""
        if model_price <= 0.01:
            return f"Option near worthless. Theta: {theta:.4f} (decay rate undefined)."
        
        theta_pct = abs(theta) / model_price
        
        if theta_pct > HybridAnalysisConfig.THETA_SIGNIFICANT_PCT:
            return (f"Significant time decay.\n"
                    f"  → Daily Theta: {theta:.4f} ({theta_pct:.2%} of option value)\n"
                    f"  → Losing ₹{abs(theta):.2f} per day. Favorable for sellers.")
        else:
            return (f"Moderate time decay.\n"
                    f"  → Daily Theta: {theta:.4f} ({theta_pct:.2%} of option value)\n"
                    f"  → Losing ₹{abs(theta):.2f} per day.")
    
    @staticmethod
    def get_rho_implication(rho, underlying_price):
        """Get Rho interpretation using normalized threshold"""
        rho_pct = (abs(rho) / underlying_price) * 100
        
        if abs(rho_pct) > HybridAnalysisConfig.RHO_SIGNIFICANT_PCT:
            direction = "increases" if rho > 0 else "decreases"
            return (f"Significant rate risk.\n"
                    f"  → Rho: {rho:+.2f} per 1% rate change ({rho_pct:.3f}% of underlying)\n"
                    f"  → If rates rise 1%, option value {direction} by ₹{abs(rho):.2f}")
        else:
            return (f"Moderate rate risk.\n"
                    f"  → Rho: {rho:+.2f} per 1% rate change ({rho_pct:.3f}% of underlying)")


# --- VALIDATION ---
def validate_inputs(K, T, r, sigma, S=None, F=None, q=None):
    """Validate all inputs before processing"""
    errors = []
    
    if K <= 0:
        errors.append("Strike price must be positive")
    if T <= 0:
        errors.append("Time to expiration must be positive")
    if sigma < 0:
        errors.append("Volatility cannot be negative")
    if r < -1 or r > 1:
        errors.append("Risk-free rate should be between -1 and 1")
    if S is not None and S <= 0:
        errors.append("Spot price must be positive")
    if F is not None and F <= 0:
        errors.append("Futures price must be positive")
    if q is not None and (q < 0 or q > 1):
        errors.append("Dividend yield should be between 0 and 1")
    
    if errors:
        raise ValueError("\n".join(errors))
    
    return True


# --- CORE PRICING MODELS ---
def black_scholes_merton(S, K, T, r, sigma, q):
    """Calculates Call and Put prices using the BSM model (for Index/Stock)"""
    T_safe = np.maximum(T, HybridAnalysisConfig.EPSILON_DENOMINATOR)
    sigma_safe = np.maximum(sigma, HybridAnalysisConfig.EPSILON_DENOMINATOR)

    if T <= 0 or sigma <= 0:
        call_price = np.maximum(0, S * np.exp(-q * T_safe) - K * np.exp(-r * T_safe))
        put_price = np.maximum(0, K * np.exp(-r * T_safe) - S * np.exp(-q * T_safe))
        return call_price, put_price, 0.0, 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * np.sqrt(T_safe))
    d2 = d1 - sigma_safe * np.sqrt(T_safe)

    call_price = S * np.exp(-q * T_safe) * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    put_price = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S * np.exp(-q * T_safe) * norm.cdf(-d1)
    return call_price, put_price, d1, d2


def black_76(F, K, T, r, sigma):
    """Calculates Call and Put prices using the Black-76 model (for Futures/Commodities)"""
    T_safe = np.maximum(T, HybridAnalysisConfig.EPSILON_DENOMINATOR)
    sigma_safe = np.maximum(sigma, HybridAnalysisConfig.EPSILON_DENOMINATOR)
    
    if T <= 0 or sigma <= 0:
        call_price = np.maximum(0, F - K) * np.exp(-r * T_safe)
        put_price = np.maximum(0, K - F) * np.exp(-r * T_safe)
        return call_price, put_price, 0.0, 0.0

    d1 = (np.log(F / K) + 0.5 * sigma_safe ** 2 * T_safe) / (sigma_safe * np.sqrt(T_safe))
    d2 = d1 - sigma_safe * np.sqrt(T_safe)
    discount_factor = np.exp(-r * T_safe)

    call_price = discount_factor * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put_price = discount_factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return call_price, put_price, d1, d2


# --- HELPER FOR ZERO-TIME GREEKS ---
def handle_zero_time_greeks(S, K, r, q, T, asset_type):
    """Handles Greeks calculation when Time to Expiry is near zero (T=0)"""
    intrinsic_call = np.maximum(0, S - K) if asset_type == 'Stock' else np.maximum(0, S*np.exp(-q*T) - K*np.exp(-r*T))
    intrinsic_put = np.maximum(0, K - S) if asset_type == 'Stock' else np.maximum(0, K*np.exp(-r*T) - S*np.exp(-q*T))
    
    # Delta approaches 1 or -1 for ITM, or 0 for OTM
    delta_call = 1.0 if S > K else 0.0
    delta_put = -1.0 if S < K else 0.0
    
    # Gamma and Vega approach infinity/zero rapidly. Set to a large proxy or zero.
    gamma = 0.0 
    vega = 0.0
    
    # Theta (time decay) is zero for a price that is just intrinsic value.
    theta_call = 0.0
    theta_put = 0.0
    
    # Rho is small near expiration
    rho_call = 0.0
    rho_put = 0.0
    
    return delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put


# --- GREEKS CALCULATION ---
def calculate_greeks_BSM(S, K, d1, d2, T, r, q, sigma):
    """Calculates all 8 Greeks for BSM model (Index/Stock) with stability checks"""
    
    sigma_sqrt_t = sigma * np.sqrt(T)
    
    if sigma_sqrt_t < HybridAnalysisConfig.EPSILON_DENOMINATOR or T < HybridAnalysisConfig.EPSILON_DENOMINATOR:
        return handle_zero_time_greeks(S, K, r, q, T, 'Stock')

    common_term = np.exp(-q * T)
    
    delta_call = common_term * norm.cdf(d1)
    delta_put = delta_call - common_term
    
    # Safe gamma calculation
    gamma = (common_term * norm.pdf(d1)) / (S * sigma_sqrt_t)
    
    vega = S * common_term * norm.pdf(d1) * np.sqrt(T)
    
    theta_call_term1 = -(S * sigma * common_term * norm.pdf(d1)) / (2 * np.sqrt(T))
    theta_call_term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
    theta_call_term3 = q * S * common_term * norm.cdf(d1)
    theta_call = theta_call_term1 - theta_call_term2 + theta_call_term3
    theta_put = theta_call + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * common_term * norm.cdf(-d1)
    
    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2)
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put


def calculate_greeks_Black76(F, K, d1, d2, T, r, sigma):
    """Calculates all 8 Greeks for Black-76 model (Commodity) with stability checks"""
    
    sigma_sqrt_t = sigma * np.sqrt(T)
    
    if sigma_sqrt_t < HybridAnalysisConfig.EPSILON_DENOMINATOR or T < HybridAnalysisConfig.EPSILON_DENOMINATOR:
        return handle_zero_time_greeks(F, K, r, 0.0, T, 'Commodity')

    discount_factor = np.exp(-r * T)
    
    delta_call = discount_factor * norm.cdf(d1)
    delta_put = delta_call - discount_factor
    
    # Safe gamma calculation
    gamma = (discount_factor * norm.pdf(d1)) / (F * sigma_sqrt_t)
    
    vega = F * discount_factor * norm.pdf(d1) * np.sqrt(T)
    
    theta_call_term1 = -(F * sigma * discount_factor * norm.pdf(d1)) / (2 * np.sqrt(T))
    theta_call_term2 = -r * K * discount_factor * norm.cdf(d2)
    theta_call_term3 = r * F * discount_factor * norm.cdf(d1)
    theta_call = theta_call_term1 + theta_call_term2 + theta_call_term3
    theta_put = theta_call + r * K * discount_factor * norm.cdf(-d2) - r * F * discount_factor * norm.cdf(-d1)

    rho_call = T * discount_factor * (F * norm.cdf(d1) - K * norm.cdf(d2)) + K * T * discount_factor * norm.cdf(d2)
    rho_put = T * discount_factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1)) - K * T * discount_factor * norm.cdf(-d2)

    return delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put


# --- IMPLIED VOLATILITY (ENHANCED) ---
def bisection_iv_solver(market_price, option_type, F_or_S, K, T, r, q, asset_type, vol_low, vol_high):
    """Fallback IV calculation using Bisection Method"""
    
    tolerance = HybridAnalysisConfig.IV_TOLERANCE
    
    for i in range(100): # Hard limit for bisection
        sigma_mid = (vol_low + vol_high) / 2
        
        if asset_type in ['Index', 'Stock']:
            call_price, put_price, _, _ = black_scholes_merton(F_or_S, K, T, r, sigma_mid, q)
        else:
            call_price, put_price, _, _ = black_76(F_or_S, K, T, r, sigma_mid)
        
        model_price = call_price if option_type == 'Call' else put_price
        
        if abs(model_price - market_price) < tolerance:
            return sigma_mid
        
        if model_price < market_price:
            vol_low = sigma_mid
        else:
            vol_high = sigma_mid

    return (vol_low + vol_high) / 2 # Return best estimate if max iterations reached


def calculate_implied_volatility(market_price, option_type, F_or_S, K, T, r, q, asset_type):
    """Calculates Implied Volatility (IV) using Newton-Raphson with Bisection fallback"""
    
    # Handle zero/negative price first
    if market_price <= 0:
        return 0.001

    sigma_guess = 0.20
    vol_low, vol_high = 0.001, 3.0 # Initial safe bounds
    tolerance = HybridAnalysisConfig.IV_TOLERANCE
    
    # Set S and F based on asset type
    S = F_or_S if asset_type in ['Index', 'Stock'] else 0.0
    F = F_or_S if asset_type == 'Commodity' else 0.0

    # 1. Newton-Raphson Attempt
    for i in range(HybridAnalysisConfig.IV_MAX_ITERATIONS):
        
        if asset_type in ['Index', 'Stock']:
            call_price, put_price, d1, d2 = black_scholes_merton(S, K, T, r, sigma_guess, q)
            _, _, _, vega, _, _, _, _ = calculate_greeks_BSM(S, K, d1, d2, T, r, q, sigma_guess)
        else: # Commodity
            call_price, put_price, d1, d2 = black_76(F, K, T, r, sigma_guess)
            _, _, _, vega, _, _, _, _ = calculate_greeks_Black76(F, K, d1, d2, T, r, sigma_guess)
        
        model_price = call_price if option_type == 'Call' else put_price
        price_error = model_price - market_price
        
        if abs(price_error) < tolerance:
            return max(0.001, sigma_guess)
        
        # Check for problematic vega (zero or near-zero) - Switch to bisection if needed
        if abs(vega) < HybridAnalysisConfig.EPSILON_DENOMINATOR:
            break
        
        sigma_new = sigma_guess - price_error / vega
        
        # Simple bounds check for N-R update
        sigma_guess = max(vol_low, min(vol_high, sigma_new))

    # 2. Bisection Fallback
    # If N-R failed to converge, use the more reliable but slower bisection method
    return bisection_iv_solver(market_price, option_type, F_or_S, K, T, r, q, asset_type, vol_low, vol_high)


# --- WRAPPER & ANALYSIS FUNCTIONS ---
def options_pricing_wrapper(S, F, K, T, r, sigma, q, asset_type):
    """Routes pricing request, calculates Greeks, and returns all results as a tuple"""
    
    if asset_type in ['Index', 'Stock']:
        call_price, put_price, d1, d2 = black_scholes_merton(S, K, T, r, sigma, q)
        greeks = calculate_greeks_BSM(S, K, d1, d2, T, r, q, sigma)
        
    elif asset_type == 'Commodity':
        call_price, put_price, d1, d2 = black_76(F, K, T, r, sigma)
        greeks = calculate_greeks_Black76(F, K, d1, d2, T, r, sigma)
        
    else:
        # Return a list of Nones/zeros if asset type is invalid
        return (0.0, 0.0, 0.0, 0.0) + (0.0,) * 8 

    # Combine prices, d1/d2, and greeks into a single tuple
    return (call_price, put_price, d1, d2) + greeks

def get_moneyness_status(delta):
    """Determines the moneyness status based on the option's delta."""
    abs_delta = abs(delta)
    if abs_delta >= HybridAnalysisConfig.DELTA_DEEP_ITM:
        return "Deep ITM (High Probability)"
    elif abs_delta >= HybridAnalysisConfig.DELTA_ATM_LOWER:
        return "ATM/Near-Term Risk (~50% Prob.)"
    else:
        return "OTM (Low Probability)"

def get_analysis_results(asset_ticker, asset_type, K, Days_to_expiry, r, sigma_user, call_market, put_market, PCR_Ratio, F_or_S, q):
    """Runs all analysis steps and returns a comprehensive results dictionary."""
    
    # 1. PREP & VALIDATION
    T = Days_to_expiry / 365
    S = F_or_S if asset_type in ['Index', 'Stock'] else 0.0
    F = F_or_S if asset_type == 'Commodity' else 0.0

    # Use the existing validation which raises ValueError on error
    if asset_type in ['Index', 'Stock']:
        validate_inputs(K, T, r, sigma_user, S=F_or_S, q=q)
    else:
        validate_inputs(K, T, r, sigma_user, F=F_or_S)
    
    # 2. CALCULATIONS at USER SIGMA
    results_user_sigma = options_pricing_wrapper(S, F, K, T, r, sigma_user, q, asset_type)
    (model_call_user, model_put_user, d1_user, d2_user, delta_call_user, delta_put_user, 
     gamma_user, vega_user, theta_call_user, theta_put_user, rho_call_user, rho_put_user) = results_user_sigma

    daily_theta_call_user = theta_call_user / 365
    daily_theta_put_user = theta_put_user / 365
    
    # Mispricing
    call_mispricing = call_market - model_call_user
    put_mispricing = put_market - model_put_user
    
    # Mispricing Status
    call_mispricing_status = 'OVERVALUED (Potential Sell)' if call_mispricing > HybridAnalysisConfig.OVERVALUED_THRESHOLD else \
                             'UNDERVALUED (Potential Buy)' if call_mispricing < HybridAnalysisConfig.UNDERVALUED_THRESHOLD else \
                             'FAIRLY VALUED'
    put_mispricing_status = 'OVERVALUED (Potential Sell)' if put_mispricing > HybridAnalysisConfig.OVERVALUED_THRESHOLD else \
                            'UNDERVALUED (Potential Buy)' if put_mispricing < HybridAnalysisConfig.UNDERVALUED_THRESHOLD else \
                            'FAIRLY VALUED'

    # 3. CALCULATIONS at MARKET IV
    IV_call = calculate_implied_volatility(call_market, 'Call', F_or_S, K, T, r, q, asset_type)
    IV_put = calculate_implied_volatility(put_market, 'Put', F_or_S, K, T, r, q, asset_type)
    IV_avg = (IV_call + IV_put) / 2

    # Recalculate Greeks/Price at IV
    results_call_iv = options_pricing_wrapper(S, F, K, T, r, IV_call, q, asset_type)
    (model_call_iv, _, d1_call_iv, d2_call_iv, delta_call_iv, _, gamma_call_iv, vega_call_iv, theta_call_iv, _, rho_call_iv, _) = results_call_iv

    results_put_iv = options_pricing_wrapper(S, F, K, T, r, IV_put, q, asset_type)
    (_, model_put_iv, d1_put_iv, d2_put_iv, _, delta_put_iv, gamma_put_iv, vega_put_iv, _, theta_put_iv, _, rho_put_iv) = results_put_iv
    
    daily_theta_call_iv = theta_call_iv / 365
    daily_theta_put_iv = theta_put_iv / 365

    call_moneyness_status = get_moneyness_status(delta_call_iv)
    put_moneyness_status = get_moneyness_status(delta_put_iv)

    # PCR Sentiment
    sentiment = "Bullish (More Puts than Calls)" if PCR_Ratio > HybridAnalysisConfig.PCR_BULLISH else \
                "Bearish (More Calls than Puts)" if PCR_Ratio < HybridAnalysisConfig.PCR_BEARISH else \
                "Neutral (Balanced)"

    # Greeks DataFrame data (Formatted as strings for display)
    greeks_table_data = {
        'Greek': ['Delta', 'Gamma', 'Vega', 'Theta (Day)', 'Rho (1% r)'],
        'Call Value (σ)': [f"{delta_call_user:.4f}", f"{gamma_user:.4f}", f"{vega_user:.4f}", f"{daily_theta_call_user:.4f}", f"{rho_call_user / 100:.4f}"],
        'Put Value (σ)': [f"{delta_put_user:.4f}", f"{gamma_user:.4f}", f"{vega_user:.4f}", f"{daily_theta_put_user:.4f}", f"{rho_put_user / 100:.4f}"],
        'Call Value (IV)': [f"{delta_call_iv:.4f}", f"{gamma_call_iv:.4f}", f"{vega_call_iv:.4f}", f"{daily_theta_call_iv:.4f}", f"{rho_call_iv / 100:.4f}"],
        'Put Value (IV)': [f"{delta_put_iv:.4f}", f"{gamma_put_iv:.4f}", f"{vega_put_iv:.4f}", f"{daily_theta_put_iv:.4f}", f"{rho_put_iv / 100:.4f}"]
    }
    greeks_df = pd.DataFrame(greeks_table_data)

    # 4. FINAL RESULTS DICTIONARY
    input_summary = {
        'Asset': asset_ticker, 'Asset Type': asset_type, 'Strike Price (K)': K,
        'Time to Expiry (Days)': Days_to_expiry, 'Risk-Free Rate (r)': f"{r*100:.2f}%",
        'User Volatility (σ)': f"{sigma_user*100:.2f}%", 'Call Market Premium': call_market,
        'Put Market Premium': put_market, 'Current PCR Ratio': PCR_Ratio,
        'Underlying Price (S/F)': F_or_S, 'Dividend Yield (q)': f"{q*100:.2f}%"
    }

    return {
        'input_summary': input_summary, 'IV_call': IV_call, 'IV_put': IV_put, 'IV_avg': IV_avg,
        'call_mispricing': call_mispricing, 'put_mispricing': put_mispricing,
        'call_mispricing_status': call_mispricing_status, 'put_mispricing_status': put_mispricing_status,
        'call_moneyness_status': call_moneyness_status, 'put_moneyness_status': put_moneyness_status,
        'greeks_df': greeks_df, 'pcr_sentiment': sentiment, 'model_call_user': model_call_user,
        'model_put_user': model_put_user, 'model_call_iv': model_call_iv, 'model_put_iv': model_put_iv,
        'gamma_call_iv': gamma_call_iv, 'gamma_put_iv': gamma_put_iv, 'vega_call_iv': vega_call_iv,
        'vega_put_iv': vega_put_iv, 'daily_theta_call_iv': daily_theta_call_iv,
        'daily_theta_put_iv': daily_theta_put_iv, 'rho_call_iv': rho_call_iv / 100, 
        'rho_put_iv': rho_put_iv / 100, 'gamma_user': gamma_user, 'vega_user': vega_user,
        'daily_theta_call_user': daily_theta_call_user, 'daily_theta_put_user': daily_theta_put_user,
        'rho_call_user': rho_call_user / 100, 'rho_put_user': rho_put_user / 100,
        'sigma_user': sigma_user, 'F_or_S': F_or_S, 'q': q, 'r': r, 'K': K, 'T': T
    }

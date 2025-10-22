"""
Option Visualization Module
Provides payoff diagrams and Greeks visualization functions for the Hybrid Options Pricing Tool
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from option_pricing_core import options_pricing_wrapper

def calculate_intrinsic_value(S, K, option_type):
    """Calculate intrinsic value for an option"""
    if option_type == 'Call':
        return max(0, S - K)
    else:  # Put
        return max(0, K - S)

def calculate_time_value(market_premium, intrinsic_value):
    """Calculate time value (extrinsic value)"""
    return market_premium - intrinsic_value

def calculate_breakeven(K, market_premium, option_type):
    """Calculate breakeven price for an option"""
    if option_type == 'Call':
        return K + market_premium
    else:  # Put
        return K - market_premium

def create_payoff_diagram(K, call_market, put_market, F_or_S, asset_name):
    """
    Creates an interactive payoff diagram showing P&L at expiration for both Call and Put options
    
    Parameters:
    -----------
    K : float
        Strike price
    call_market : float
        Call option market premium
    put_market : float
        Put option market premium
    F_or_S : float
        Current underlying price
    asset_name : str
        Name of the underlying asset
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive payoff diagram
    """
    
    # Create price range around strike (±40%)
    price_range = np.linspace(K * 0.6, K * 1.4, 200)
    
    # Calculate Call P&L at expiration
    call_intrinsic = np.maximum(0, price_range - K)
    call_pnl = call_intrinsic - call_market
    
    # Calculate Put P&L at expiration
    put_intrinsic = np.maximum(0, K - price_range)
    put_pnl = put_intrinsic - put_market
    
    # Calculate breakeven points
    call_breakeven = K + call_market
    put_breakeven = K - put_market
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[f'Option Payoff Diagram at Expiration (Strike: ₹{K:,.2f})']
    )
    
    # Add Call P&L line
    fig.add_trace(
        go.Scatter(
            x=price_range,
            y=call_pnl,
            name='Call P&L',
            line=dict(color='#2ecc71', width=3),
            hovertemplate='<b>Call Option</b><br>Price: ₹%{x:.2f}<br>P&L: ₹%{y:.2f}<extra></extra>'
        )
    )
    
    # Add Put P&L line
    fig.add_trace(
        go.Scatter(
            x=price_range,
            y=put_pnl,
            name='Put P&L',
            line=dict(color='#e74c3c', width=3),
            hovertemplate='<b>Put Option</b><br>Price: ₹%{x:.2f}<br>P&L: ₹%{y:.2f}<extra></extra>'
        )
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5)
    
    # Add strike price line
    fig.add_vline(x=K, line_dash="dot", line_color="purple", line_width=2, 
                  annotation_text=f"Strike: ₹{K:.2f}", annotation_position="top")
    
    # Add current price line
    fig.add_vline(x=F_or_S, line_dash="dot", line_color="orange", line_width=2,
                  annotation_text=f"Current: ₹{F_or_S:.2f}", annotation_position="bottom")
    
    # Add Call breakeven marker
    fig.add_trace(
        go.Scatter(
            x=[call_breakeven],
            y=[0],
            mode='markers',
            name='Call Breakeven',
            marker=dict(size=12, color='#2ecc71', symbol='diamond', line=dict(width=2, color='white')),
            hovertemplate=f'<b>Call Breakeven</b><br>₹{call_breakeven:.2f}<extra></extra>'
        )
    )
    
    # Add Put breakeven marker
    fig.add_trace(
        go.Scatter(
            x=[put_breakeven],
            y=[0],
            mode='markers',
            name='Put Breakeven',
            marker=dict(size=12, color='#e74c3c', symbol='diamond', line=dict(width=2, color='white')),
            hovertemplate=f'<b>Put Breakeven</b><br>₹{put_breakeven:.2f}<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=f'{asset_name} Price at Expiration (₹)',
        yaxis_title='Profit / Loss (₹)',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(240,242,246,0.5)',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    # Add shaded profit/loss regions
    fig.add_hrect(y0=0, y1=max(call_pnl.max(), put_pnl.max()) * 1.1, 
                  fillcolor="green", opacity=0.05, line_width=0)
    fig.add_hrect(y0=min(call_pnl.min(), put_pnl.min()) * 1.1, y1=0, 
                  fillcolor="red", opacity=0.05, line_width=0)
    
    return fig

def create_greeks_sensitivity_chart(S, F, K, T, r, sigma, q, asset_type, asset_name):
    """
    Creates visualization showing how Greeks change with underlying price
    
    Parameters:
    -----------
    S : float
        Spot price (for Stock/Index)
    F : float
        Futures price (for Commodity)
    K : float
        Strike price
    T : float
        Time to expiration (in years)
    r : float
        Risk-free rate
    sigma : float
        Volatility
    q : float
        Dividend yield
    asset_type : str
        'Stock', 'Index', or 'Commodity'
    asset_name : str
        Name of the underlying asset
    
    Returns:
    --------
    tuple of (fig_delta_gamma, fig_vega_theta)
        Two plotly figures showing Greeks sensitivity
    """
    
    F_or_S = S if asset_type in ['Index', 'Stock'] else F
    
    # Create price range around current price (±30%)
    price_range = np.linspace(F_or_S * 0.7, F_or_S * 1.3, 100)
    
    # Initialize arrays for Greeks
    delta_call_arr = []
    delta_put_arr = []
    gamma_arr = []
    vega_arr = []
    theta_call_arr = []
    theta_put_arr = []
    
    # Calculate Greeks for each price point
    for price in price_range:
        if asset_type in ['Index', 'Stock']:
            results = options_pricing_wrapper(price, F, K, T, r, sigma, q, asset_type)
        else:
            results = options_pricing_wrapper(S, price, K, T, r, sigma, q, asset_type)
        
        # Unpack results
        (call_price, put_price, d1, d2, delta_call, delta_put, 
         gamma, vega, theta_call, theta_put, rho_call, rho_put) = results
        
        delta_call_arr.append(delta_call)
        delta_put_arr.append(delta_put)
        gamma_arr.append(gamma)
        vega_arr.append(vega / 100)  # Scale Vega for display (change per 1% IV move)
        theta_call_arr.append(theta_call / 365)  # Daily theta
        theta_put_arr.append(theta_put / 365)
    
    # Create Delta & Gamma chart
    fig_delta_gamma = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Delta Sensitivity', 'Gamma Sensitivity'),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Delta plot
    fig_delta_gamma.add_trace(
        go.Scatter(x=price_range, y=delta_call_arr, name='Call Delta',
                   line=dict(color='#2ecc71', width=2.5),
                   hovertemplate='Price: ₹%{x:.2f}<br>Delta: %{y:.4f}<extra></extra>'),
        row=1, col=1
    )
    
    fig_delta_gamma.add_trace(
        go.Scatter(x=price_range, y=delta_put_arr, name='Put Delta',
                   line=dict(color='#e74c3c', width=2.5),
                   hovertemplate='Price: ₹%{x:.2f}<br>Delta: %{y:.4f}<extra></extra>'),
        row=1, col=1
    )
    
    # Gamma plot
    fig_delta_gamma.add_trace(
        go.Scatter(x=price_range, y=gamma_arr, name='Gamma',
                   line=dict(color='#9b59b6', width=2.5),
                   fill='tozeroy', fillcolor='rgba(155,89,182,0.1)',
                   hovertemplate='Price: ₹%{x:.2f}<br>Gamma: %{y:.6f}<extra></extra>'),
        row=2, col=1
    )
    
    # Add strike line to both subplots
    fig_delta_gamma.add_vline(x=K, line_dash="dot", line_color="purple", line_width=1.5, row=1, col=1)
    fig_delta_gamma.add_vline(x=K, line_dash="dot", line_color="purple", line_width=1.5, row=2, col=1)
    
    # Add current price line
    fig_delta_gamma.add_vline(x=F_or_S, line_dash="dash", line_color="orange", line_width=1.5, row=1, col=1)
    fig_delta_gamma.add_vline(x=F_or_S, line_dash="dash", line_color="orange", line_width=1.5, row=2, col=1)
    
    fig_delta_gamma.update_xaxes(title_text=f'{asset_name} Price (₹)', row=2, col=1)
    fig_delta_gamma.update_yaxes(title_text='Delta', row=1, col=1)
    fig_delta_gamma.update_yaxes(title_text='Gamma', row=2, col=1)
    
    fig_delta_gamma.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='rgba(240,242,246,0.5)',
        paper_bgcolor='white'
    )
    
    # Create Vega & Theta chart
    fig_vega_theta = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Vega Sensitivity (₹ per 1% IV change)', 'Theta (Time Decay) Sensitivity'),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Vega plot (now scaled by 100 for per 1% IV change)
    fig_vega_theta.add_trace(
        go.Scatter(x=price_range, y=vega_arr, name='Vega',
                   line=dict(color='#3498db', width=2.5),
                   fill='tozeroy', fillcolor='rgba(52,152,219,0.1)',
                   hovertemplate='Price: ₹%{x:.2f}<br>Vega: ₹%{y:.4f} per 1% IV<extra></extra>'),
        row=1, col=1
    )
    
    # Theta plot
    fig_vega_theta.add_trace(
        go.Scatter(x=price_range, y=theta_call_arr, name='Call Theta',
                   line=dict(color='#e67e22', width=2.5),
                   hovertemplate='Price: ₹%{x:.2f}<br>Daily Theta: ₹%{y:.4f}<extra></extra>'),
        row=2, col=1
    )
    
    fig_vega_theta.add_trace(
        go.Scatter(x=price_range, y=theta_put_arr, name='Put Theta',
                   line=dict(color='#e74c3c', width=2.5),
                   hovertemplate='Price: ₹%{x:.2f}<br>Daily Theta: ₹%{y:.4f}<extra></extra>'),
        row=2, col=1
    )
    
    # Add strike and current price lines
    fig_vega_theta.add_vline(x=K, line_dash="dot", line_color="purple", line_width=1.5, row=1, col=1)
    fig_vega_theta.add_vline(x=K, line_dash="dot", line_color="purple", line_width=1.5, row=2, col=1)
    fig_vega_theta.add_vline(x=F_or_S, line_dash="dash", line_color="orange", line_width=1.5, row=1, col=1)
    fig_vega_theta.add_vline(x=F_or_S, line_dash="dash", line_color="orange", line_width=1.5, row=2, col=1)
    
    fig_vega_theta.update_xaxes(title_text=f'{asset_name} Price (₹)', row=2, col=1)
    fig_vega_theta.update_yaxes(title_text='Vega (₹ per 1% IV)', row=1, col=1)
    fig_vega_theta.update_yaxes(title_text='Daily Theta (₹/day)', row=2, col=1)
    
    fig_vega_theta.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='rgba(240,242,246,0.5)',
        paper_bgcolor='white'
    )
    
    return fig_delta_gamma, fig_vega_theta

def get_enhanced_pricing_table(results, call_market, put_market, F_or_S, K):
    """
    Creates an enhanced pricing analysis table with Intrinsic/Time Value breakdown and Breakeven
    
    Parameters:
    -----------
    results : dict
        Results dictionary from get_analysis_results()
    call_market : float
        Call option market premium
    put_market : float
        Put option market premium
    F_or_S : float
        Current underlying price
    K : float
        Strike price
    
    Returns:
    --------
    pandas.DataFrame
        Enhanced pricing table with additional metrics
    """
    
    # Calculate intrinsic and time values
    call_intrinsic = calculate_intrinsic_value(F_or_S, K, 'Call')
    put_intrinsic = calculate_intrinsic_value(F_or_S, K, 'Put')
    
    call_time_value = calculate_time_value(call_market, call_intrinsic)
    put_time_value = calculate_time_value(put_market, put_intrinsic)
    
    # Calculate breakeven prices
    call_breakeven = calculate_breakeven(K, call_market, 'Call')
    put_breakeven = calculate_breakeven(K, put_market, 'Put')
    
    # Extract probability from moneyness status
    call_moneyness = results['call_moneyness_status']
    put_moneyness = results['put_moneyness_status']
    
    # Create enhanced pricing data
    pricing_data = {
        'Metric': [
            'Theoretical Price (at User σ)',
            'Market Premium',
            'Intrinsic Value',
            'Time Value (Extrinsic)',
            'Theoretical Price (at IV)',
            'Mispricing (Market vs User σ)',
            'Valuation Status (vs User σ)',
            'Moneyness Status (at IV)',
            'Breakeven Price'
        ],
        'Call Option': [
            f"₹{results['model_call_user']:.2f}",
            f"₹{call_market:.2f}",
            f"₹{call_intrinsic:.2f}",
            f"₹{call_time_value:.2f}",
            f"₹{results['model_call_iv']:.2f}",
            f"{results['call_mispricing']:+.2f}",
            results['call_mispricing_status'],
            call_moneyness,
            f"₹{call_breakeven:.2f}"
        ],
        'Put Option': [
            f"₹{results['model_put_user']:.2f}",
            f"₹{put_market:.2f}",
            f"₹{put_intrinsic:.2f}",
            f"₹{put_time_value:.2f}",
            f"₹{results['model_put_iv']:.2f}",
            f"{results['put_mispricing']:+.2f}",
            results['put_mispricing_status'],
            put_moneyness,
            f"₹{put_breakeven:.2f}"
        ]
    }
    
    return pd.DataFrame(pricing_data)

# 📊 QuantifiLib

**A comprehensive, user-friendly Python library for quantitative finance and investment analysis**

QuantifiLib is a production-ready Python library designed for both beginners and professionals in quantitative finance. Whether you're learning your first stock analysis or building sophisticated trading strategies, QuantifiLib provides the tools you need with robust error handling and comprehensive tutorials.

## 🎯 **What Makes QuantifiLib Special**

- **🚀 Beginner-Friendly:** Works even with missing dependencies - learn concepts first, add features later
- **🔧 Robust & Reliable:** Production-tested with comprehensive error handling and graceful degradation
- **📚 Educational:** Interactive Jupyter tutorials that teach finance concepts step-by-step
- **⚡ Professional-Grade:** Advanced features for portfolio optimization and risk management
- **🛡️ Secure:** Input validation, dependency checks, and security best practices built-in

## ✨ **Key Features**

### **📈 Data Loading & Management**
- **Yahoo Finance Integration:** Download real-time and historical stock data
- **FRED Economic Data:** Access Federal Reserve economic indicators
- **International Markets:** Support for Korean markets via Naver Finance
- **Robust Error Handling:** Graceful handling of network issues and missing data
- **Smart Caching:** Automatic data caching for improved performance

### **🔍 Analysis & Metrics**
- **Technical Indicators:** Moving averages, volatility, momentum indicators
- **Risk Metrics:** VaR, Sharpe ratio, correlation analysis, drawdown analysis
- **Performance Attribution:** Understand what drives portfolio returns
- **Market Microstructure:** Bid-ask spreads, market impact, liquidity metrics

### **🎯 Strategy Development**
- **Event-Driven Strategies:** Triple barrier labeling, regime detection
- **Machine Learning Integration:** SVM, feature engineering, walk-forward analysis
- **Signal Generation:** Buy/sell signals with backtesting capabilities
- **Portfolio Optimization:** Modern portfolio theory, risk parity, efficient frontier

### **📊 Visualization & Reporting**
- **Interactive Charts:** Price charts, correlation heatmaps, performance attribution
- **Risk-Return Analysis:** Scatter plots, efficient frontier visualization
- **Portfolio Analytics:** Asset allocation, rebalancing recommendations
- **Professional Reports:** Export-ready analysis and documentation

## 🚀 **Quick Start**

### **📦 Installation**

**Option 1: Full Installation (Recommended)**
```bash
# Clone or download QuantifiLib
git clone <repository-url>
cd quantifilib

# Install with all dependencies
pip install -e .

# Or install dependencies manually
pip install pandas numpy yfinance matplotlib seaborn jupyter
```

**Option 2: Minimal Installation**
```bash
# Install just QuantifiLib (works with limited features)
pip install -e .

# Add dependencies as needed
pip install pandas numpy  # For data analysis
pip install matplotlib    # For charts
pip install yfinance      # For stock data
```

### **🎯 Your First Analysis (3 minutes)**

```python
import quantifilib as ql

# Load stock data
loader = ql.YahooFinanceLoader()
data = loader.load('AAPL', start='2023-01-01', end='2024-01-01')

# Calculate returns
returns = data['Close'].pct_change() * 100
total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100

print(f"AAPL Total Return: {total_return:.2f}%")
print(f"Average Daily Return: {returns.mean():.2f}%")
print(f"Volatility: {returns.std() * (252**0.5):.2f}%")
```

### **📚 Interactive Tutorials**

**Perfect for beginners!** Start with our interactive Jupyter notebooks:

1. **`examples/Getting_Started.ipynb`** - Learn the basics (30 minutes)
2. **`examples/Stock_Analysis.ipynb`** - Advanced analysis (1 hour)  
3. **`examples/Portfolio_Builder.ipynb`** - Portfolio optimization (1 hour)

```bash
# Launch Jupyter and start learning
jupyter notebook examples/
```

## 🔧 **Robust Dependency Handling**

**✅ Works Even With Missing Dependencies!**

QuantifiLib is designed to be educational-first. Even if you're missing dependencies, you can:
- ✅ Learn concepts through example outputs
- ✅ See what real analysis looks like
- ✅ Get clear installation instructions
- ✅ Progress at your own pace

**Missing pandas?** → See example data and calculations  
**Missing matplotlib?** → View text-based chart demonstrations  
**Missing yfinance?** → Learn with sample data examples

### 📁 Project Structure

<pre lang="markdown">

quantifilib/
├── data/
│   ├── data_loader/
│   │   ├── yfinance_loader.py
│   │   ├── fred_loader.py
│   │   └── naver_loader.py
│   └── stock_universe/
│       └── wikipedia.py
├── features/
│   ├── bar_sampling/
│   │   ├── bar_feature.py
│   │   ├── base_bars.py
│   │   ├── core.py
│   │   ├── imbalance_data_structures.py
│   │   ├── microstructure.py
│   │   ├── run_data_structures.py
│   │   ├── standard_data_structures.py
│   │   └── time_data_structures.py
├── metrics/
│   ├── liquidity/
│   │   ├── corwin_schultz.py
│   │   ├── lambda.py
│   │   ├── pin.py
│   │   └── roll_models.py
│   └── risk/
│   │   ├── market.py
│   │   └── strategy.py
├── strategy/   
│   ├── fundamental_based/
│   ├── ml_based/
│   │   └── suppport_vector_machine.py
│   ├── price_based/
│   │   ├── technical.py
│   │   └── triple_barrier.py
│   ├── statistical_based/
│   │   └── trend_search.py
│   └── base_label.py
└── utils/
    ├── multiprocess/
    │   ├── parts.py
    │   └── process_job.py
    └── fast_ewma.py
    
</pre>

## 💡 **Usage Examples**

### **📊 Basic Stock Analysis**

```python
import quantifilib as ql
import matplotlib.pyplot as plt

# Load multiple stocks
loader = ql.YahooFinanceLoader()
stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
all_data = {}

for stock in stocks:
    all_data[stock] = loader.load(stock, start='2023-01-01', end='2024-01-01')

# Compare performance
for stock, data in all_data.items():
    start_price = float(data.iloc[0]['Close'].iloc[0])
    end_price = float(data.iloc[-1]['Close'].iloc[0])
    total_return = (end_price - start_price) / start_price * 100
    print(f"{stock}: {total_return:.2f}% return")
```

### **🎯 Risk Analysis**

```python
# Calculate risk metrics
returns = data['Close'].pct_change().dropna() * 100
volatility = returns.std() * (252 ** 0.5)  # Annualized
sharpe_ratio = (returns.mean() * 252 - 3) / volatility  # Assuming 3% risk-free rate

print(f"Annual Volatility: {volatility:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")

# Risk categories
if volatility < 15:
    risk_level = "LOW (Conservative)"
elif volatility < 25:
    risk_level = "MEDIUM (Moderate)"
else:
    risk_level = "HIGH (Aggressive)"
    
print(f"Risk Level: {risk_level}")
```

### **📈 Portfolio Analysis**

```python
# Create equal-weight portfolio
portfolio_weights = {stock: 1/len(stocks) for stock in stocks}
portfolio_returns = 0

for stock, weight in portfolio_weights.items():
    stock_returns = all_data[stock]['Close'].pct_change()
    portfolio_returns += weight * stock_returns

# Portfolio metrics
portfolio_total_return = (portfolio_returns + 1).prod() - 1
portfolio_volatility = portfolio_returns.std() * (252 ** 0.5) * 100
portfolio_sharpe = (portfolio_returns.mean() * 252 - 0.03) / (portfolio_returns.std() * (252 ** 0.5))

print(f"Portfolio Return: {portfolio_total_return * 100:.2f}%")
print(f"Portfolio Volatility: {portfolio_volatility:.2f}%")
print(f"Portfolio Sharpe: {portfolio_sharpe:.3f}")
```

### **🔍 Technical Analysis**

```python
# Moving averages and signals
data = all_data['AAPL'].copy()
close_prices = data['Close'].apply(lambda x: float(x.iloc[0]))

# Calculate moving averages
data['MA_20'] = close_prices.rolling(window=20).mean()
data['MA_50'] = close_prices.rolling(window=50).mean()

# Generate signals
data['Signal'] = 0
data.loc[data['MA_20'] > data['MA_50'], 'Signal'] = 1  # Buy signal
data.loc[data['MA_20'] < data['MA_50'], 'Signal'] = -1  # Sell signal

# Count signals
buy_signals = (data['Signal'] == 1).sum()
sell_signals = (data['Signal'] == -1).sum()

print(f"Buy signals: {buy_signals}")
print(f"Sell signals: {sell_signals}")
```

## 🛡️ **Error Handling & Best Practices**

### **🔒 Built-in Security Features**

```python
# Input validation
from quantifilib.utils.validation import validate_symbol, validate_date_range

# Validate stock symbols
valid_symbols = validate_symbol(['AAPL', 'INVALID_SYMBOL'], raise_on_error=False)
print(f"Valid symbols: {valid_symbols}")

# Validate date ranges
start, end = validate_date_range('2023-01-01', '2024-01-01')
print(f"Validated date range: {start} to {end}")
```

### **⚠️ Graceful Error Handling**

```python
# Robust data loading with error handling
def safe_load_data(symbols, start_date, end_date):
    loader = ql.YahooFinanceLoader()
    results = {}
    failed = []
    
    for symbol in symbols:
        try:
            data = loader.load(symbol, start=start_date, end=end_date)
            results[symbol] = data
            print(f"✅ {symbol}: Loaded successfully")
        except Exception as e:
            failed.append(symbol)
            print(f"❌ {symbol}: Failed - {e}")
    
    return results, failed

# Usage
data, failed_symbols = safe_load_data(['AAPL', 'MSFT', 'INVALID'], '2023-01-01', '2024-01-01')
```

## 🚀 **Advanced Features**

### **🎯 Event-Driven Strategies**

```python
# Triple barrier labeling for ML
from quantifilib.strategy.event_driven import TripleBarrierLabeling

labeler = TripleBarrierLabeling()
labels = labeler.get_labels(
    prices=data['Close'],
    events=signal_events,
    pt_sl=[0.02, 0.02],  # 2% profit/stop loss
    t1=t1_events  # Time barriers
)
```

### **📊 Advanced Risk Metrics**

```python
# Market risk metrics
from quantifilib.metrics.risk import garman_klass_volatility, parkinson_volatility

# High-frequency volatility estimators
gk_vol = garman_klass_volatility(data['High'], data['Low'], data['Open'], data['Close'])
pk_vol = parkinson_volatility(data['High'], data['Low'])

print(f"Garman-Klass Volatility: {gk_vol:.4f}")
print(f"Parkinson Volatility: {pk_vol:.4f}")
```

### **🔍 Market Microstructure**

```python
# Liquidity analysis
from quantifilib.metrics.illiquidity import CorwinSchultz, PIN

# Bid-ask spread estimation
cs = CorwinSchultz()
spread = cs.estimate_spread(data['High'], data['Low'], data['Close'])

# Probability of informed trading
pin = PIN()
pin_estimate = pin.calculate(data['Volume'], data['Close'])

print(f"Estimated Spread: {spread:.4f}")
print(f"PIN Estimate: {pin_estimate:.4f}")
```

## 🎓 **Learning Path**

### **👶 Beginner (Start Here)**
1. **Getting Started Tutorial** - Learn basic concepts
2. **Your First Analysis** - Analyze one stock 
3. **Compare Stocks** - Multi-stock analysis
4. **Basic Portfolio** - Equal-weight portfolio

### **📈 Intermediate**
1. **Technical Analysis** - Moving averages, signals
2. **Risk Management** - Volatility, Sharpe ratios
3. **Correlation Analysis** - Diversification concepts
4. **Market Comparison** - Benchmark analysis

### **🎯 Advanced**
1. **Portfolio Optimization** - Modern portfolio theory
2. **Machine Learning** - Predictive models
3. **Event-Driven Strategies** - Signal generation
4. **Market Microstructure** - Liquidity analysis

## ⚙️ **Configuration & Customization**

### **🔧 Configuration Management**

```python
# Configure QuantifiLib
from quantifilib import get_config, set_config

# View current configuration
config = get_config()
print(config.get(['data', 'cache_enabled']))

# Customize settings
set_config(['data', 'cache_ttl'], 3600)  # Cache for 1 hour
set_config(['security', 'max_symbol_length'], 10)
```

### **📊 Custom Data Sources**

```python
# Extend with custom data loader
from quantifilib.data.data_loader import BaseDataLoader

class CustomDataLoader(BaseDataLoader):
    def load(self, symbols, start, end, **kwargs):
        # Your custom implementation
        return formatted_data
```

## 🆘 **Troubleshooting**

### **🔧 Common Issues & Solutions**

| Issue | Solution |
|-------|----------|
| `No module named 'pandas'` | `pip install pandas` |
| `YahooFinanceLoader requires...` | Install dependencies shown in error |
| `No data returned` | Try different symbol or date range |
| Charts not showing | `pip install matplotlib` |
| Internet/network errors | Check connection, try again |

### **📋 Getting Help**

- **📖 Documentation:** Check examples/ folder
- **💬 Issues:** Report problems with detailed error messages
- **🔍 Debugging:** Use verbose error messages for diagnosis
- **📚 Tutorials:** Start with Getting_Started.ipynb

## 🤝 **Contributing**

QuantifiLib welcomes contributions! Whether you're:
- 🐛 **Reporting bugs**
- 💡 **Suggesting features** 
- 📝 **Improving documentation**
- 🔧 **Contributing code**

## 📄 **License**

MIT License - see LICENSE file for details.

## 🎯 **Why Choose QuantifiLib?**

✅ **Educational First:** Learn concepts even without full setup  
✅ **Production Ready:** Robust error handling and testing  
✅ **Comprehensive:** From basics to advanced portfolio theory  
✅ **Flexible:** Works with partial dependencies  
✅ **Modern:** Best practices in Python development  
✅ **Extensible:** Easy to customize and extend  

---

**Ready to start your quantitative finance journey?** 🚀

**[Begin with examples/Getting_Started.ipynb →](examples/Getting_Started.ipynb)**
# Leonardo AI

A multi-agent AI system that simulates crowd intelligence predictions using **real search capabilities**.

## 🤖 Four-Agent System

### 🤖 Optimistic Investor
- Searches financial news (NewsAPI, Google News RSS)
- Fetches stock data (Yahoo Finance, yfinance)
- Provides bullish probability estimates

### 🛡️ Risk-Control Investor  
- Analyzes risk factors and volatility
- Monitors earnings and corporate events
- Provides conservative probability estimates

### 😤 Hater Agent
- Searches Reddit discussions (`site:reddit.com`)
- Finds Twitter complaints (`site:twitter.com`)
- Provides pessimistic probability estimates

### 🔬 Technical Expert
- Web research via DuckDuckGo
- Real-time financial data analysis
- Provides objective probability estimates

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Run Meta-Integrator (coordinates all agents):**
```bash
python meta_integrator_agent.py
```

**Run Individual Agents:**
```bash
python optimistic_investor_agent.py
python risk_control_investor_agent.py
python technical_expert_agent.py
python hater_agent.py
```

## Example Questions

- Will Apple release Vision Pro 2 in 2025?
- Will Tesla launch a $50k electric car in 2025?
- Will NVIDIA stock reach $200 in 2025?

## How It Works

1. **Meta-Integrator** coordinates all four agents simultaneously
2. Each agent searches different data sources and provides probability estimates
3. System aggregates weighted probabilities and synthesizes final recommendation

## Data Sources

- **Financial Data**: NewsAPI.org, Google News RSS, Yahoo Finance
- **Social Media**: Reddit (`site:reddit.com`), Twitter (`site:twitter.com`)
- **Web Search**: DuckDuckGo for general research
- **No API keys required** - uses free public data sources

---

Made with ❤️ for NVIDIA Hackathon

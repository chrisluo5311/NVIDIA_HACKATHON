# AI Betting Market Simulator

An AI prediction market simulator that uses multiple AI Agents with **REAL SEARCH CAPABILITIES** to simulate crowd intelligence predictions.

## 🎯 NEW: Four-Agent System with Real Search Capabilities!

All four AI Agents now have **actual internet search abilities** and specialized tools:

### 🤖 Optimistic Investor Agent
**Tools Used:**
- `get_finance_news` - Searches financial news and market updates
- `get_stock_price` - Fetches real-time stock prices and historical data
- `get_company_events` - Retrieves earnings dates and corporate events

**Websites Searched:**
- **NewsAPI.org** - Professional financial news aggregation
- **Google News RSS** - Finance-focused news feeds (`news.google.com/rss/search`)
- **Yahoo Finance** - Stock prices via `query1.finance.yahoo.com/v8/finance/chart/`
- **yfinance library** - Real-time market data and company calendars

### 🛡️ Risk-Control Investor Agent
**Tools Used:**
- `get_finance_news` - Searches for risk-related financial news
- `get_stock_price` - Analyzes price volatility and trends
- `get_company_events` - Monitors earnings and corporate announcements

**Websites Searched:**
- **NewsAPI.org** - Risk-focused financial news
- **Google News RSS** - Market analysis and risk reports
- **Yahoo Finance** - Volatility data and price history
- **yfinance library** - Risk metrics and company event calendars

### 😤 Hater Agent
**Tools Used:**
- `SocialMediaTool.search_reddit_sentiment` - Finds negative Reddit discussions
- `SocialMediaTool.search_twitter_sentiment` - Discovers critical tweets
- `WebSearchTool.search` - Searches for general negative opinions

**Websites Searched:**
- **Reddit** - Via Google search (`site:reddit.com`) for negative discussions
- **Twitter/X** - Via Google search (`site:twitter.com OR site:x.com`) for complaints
- **DuckDuckGo** - General web search for criticism and negative sentiment
- **Various forums and blogs** - For genuine complaints and negative reviews

### 🔬 Technical Expert Agent
**Tools Used:**
- `WebSearchTool.search` - General web research and analysis
- `FinancialDataTool.get_stock_info` - Real-time financial data
- `FinancialDataTool.search_company_news` - Company-specific news

**Websites Searched:**
- **DuckDuckGo** - Web search via `html.duckduckgo.com/html/`
- **Yahoo Finance** - Stock data via `query1.finance.yahoo.com/v8/finance/chart/`
- **Various news sites** - Technical analysis and industry reports
- **Company websites** - Official announcements and technical specifications

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. API Key is already built into the code, no additional configuration needed!

## Usage

**Run Meta-Integrator (coordinates all four agents):**
```bash
python meta_integrator_agent.py
```

**Run Individual Agents:**

**Optimistic Investor (with financial tools):**
```bash
python optimistic_investor_agent.py
```

**Risk-Control Investor (with risk analysis tools):**
```bash
python risk_control_investor_agent.py
```

**Technical Expert (with real data search):**
```bash
python technical_expert_agent.py
```

**Hater (with real social media search):**
```bash
python hater_agent.py
```

**Test Search Tools Only:**
```bash
python search_tools.py
```

## Example Questions

Try asking these questions:
- Will Apple release Vision Pro 2 in 2025?
- Will Tesla launch a $50k electric car in 2025?
- Will TSMC mass-produce 1nm chips in 2026?
- Will OpenAI release GPT-5 in 2024?
- Will Meta's metaverse succeed?
- Will NVIDIA stock reach $200 in 2025?

## How It Works

### Meta-Integrator System Flow:
1. 📡 **Coordinates All Agents**
   - Runs all four specialized agents simultaneously
   - Each agent searches different data sources
   - Collects probability estimates and reasoning from each

2. 📊 **Integrates Perspectives**
   - Aggregates weighted probabilities from all agents
   - Analyzes disagreement and consensus patterns
   - Synthesizes final recommendation

3. 💰 **Returns Comprehensive Analysis**
   - Single final probability with detailed reasoning
   - Shows individual agent perspectives
   - Provides investor-grade synthesis

### Individual Agent Flows:

#### 🤖 Optimistic Investor Agent:
1. **Gathers Financial Data**
   - Searches NewsAPI.org for bullish financial news
   - Fetches Yahoo Finance stock prices and trends
   - Monitors Google News RSS for market updates

2. **Analyzes Upside Potential**
   - AI analyzes catalysts and growth opportunities
   - Creates event timelines and monitoring plans
   - Provides bullish probability estimates

#### 🛡️ Risk-Control Investor Agent:
1. **Identifies Risk Factors**
   - Searches for risk-related financial news
   - Analyzes volatility and downside scenarios
   - Monitors earnings and corporate events

2. **Develops Risk Management**
   - Creates hedging strategies and stop-loss levels
   - Provides conservative probability estimates
   - Focuses on capital preservation

#### 😤 Hater Agent:
1. **Searches for Negativity**
   - Scans Reddit discussions for complaints
   - Searches Twitter/X for critical tweets
   - Finds negative reviews and opinions

2. **Channels Real Hate**
   - AI reads actual negative comments
   - Mimics real hater sentiment and language
   - Provides pessimistic probability estimates

#### 🔬 Technical Expert Agent:
1. **Gathers Technical Data**
   - Searches DuckDuckGo for technical analysis
   - Fetches Yahoo Finance financial metrics
   - Collects industry reports and specifications

2. **Performs Data-Driven Analysis**
   - AI analyzes technical feasibility
   - Considers market demand and competition
   - Provides objective probability estimates

## Features

### 🤖 Optimistic Investor Agent
- ✅ Uses moderate temperature (0.6) for balanced optimism
- ✅ **REAL financial news** via NewsAPI.org and Google News RSS
- ✅ **REAL stock data** via Yahoo Finance and yfinance
- ✅ Event planning with concrete timelines and triggers
- ✅ Bullish analysis with risk awareness

### 🛡️ Risk-Control Investor Agent
- ✅ Uses lower temperature (0.5) for conservative analysis
- ✅ **REAL financial news** via NewsAPI.org and Google News RSS
- ✅ **REAL stock data** via Yahoo Finance and yfinance
- ✅ Risk-first approach with hedging strategies
- ✅ Capital preservation focus with stop-loss levels

### 😤 Hater Agent
- ✅ Uses higher temperature (0.9) for emotional responses
- ✅ **REAL Reddit discussions** via Google search (`site:reddit.com`)
- ✅ **REAL Twitter opinions** via Google search (`site:twitter.com OR site:x.com`)
- ✅ **REAL web search** via DuckDuckGo for negative sentiment
- ✅ Bases negativity on actual online comments and complaints

### 🔬 Technical Expert Agent
- ✅ Uses moderate temperature (0.6) for rational analysis
- ✅ **REAL internet search** via DuckDuckGo (`html.duckduckgo.com/html/`)
- ✅ **REAL stock data** via Yahoo Finance (`query1.finance.yahoo.com/v8/finance/chart/`)
- ✅ Structured analysis based on actual data
- ✅ Considers technical feasibility, market demand, and competition

## Technical Details

### Search Tools (`search_tools.py`)
- **WebSearchTool** - DuckDuckGo search via `html.duckduckgo.com/html/` (no API key needed)
- **FinancialDataTool** - Yahoo Finance data via `query1.finance.yahoo.com/v8/finance/chart/` (free)
- **SocialMediaTool** - Reddit & Twitter search via Google (`site:reddit.com`, `site:twitter.com`)

### Agent-Specific Tools
- **Optimistic & Risk-Control Investors**: `get_finance_news`, `get_stock_price`, `get_company_events`
- **Hater Agent**: `SocialMediaTool.search_reddit_sentiment`, `SocialMediaTool.search_twitter_sentiment`
- **Technical Expert**: `WebSearchTool.search`, `FinancialDataTool.get_stock_info`

### Websites and APIs Used
- **NewsAPI.org** - Professional financial news (`newsapi.org/v2/everything`)
- **Google News RSS** - Finance news feeds (`news.google.com/rss/search`)
- **Yahoo Finance** - Stock data (`query1.finance.yahoo.com/v8/finance/chart/`)
- **DuckDuckGo** - Web search (`html.duckduckgo.com/html/`)
- **Reddit** - Via Google search (`site:reddit.com`)
- **Twitter/X** - Via Google search (`site:twitter.com OR site:x.com`)
- **yfinance library** - Python library for financial data

### No API Keys Required!
All search tools use free, public data sources:
- Web search via DuckDuckGo HTML scraping
- Stock data via Yahoo Finance public API
- Social media via Google search scraping
- News via NewsAPI.org (with optional API key) or Google News RSS

## Example Output

```
>>> Meta-Integrator Agent (NVIDIA Llama-3.3 Nemotron Super 49B v1.5) <<<
Analyzing: Will Apple release Vision Pro 2 in 2025?

🔄 Running all four agents in parallel...

=== Running Optimistic Investor ===
📡 Searching NewsAPI.org for Apple Vision Pro news...
💰 Fetching Yahoo Finance data for AAPL...
[Optimistic analysis with 75% probability]

=== Running Risk-Control Investor ===
🛡️ Analyzing risk factors and volatility...
📊 Monitoring earnings and corporate events...
[Risk-control analysis with 45% probability]

=== Running Hater Agent ===
📡 Searching for Real Hater Comments...
🔍 Searching Reddit for negative opinions...
🐦 Searching Twitter/X for complaints...
[Hater analysis with 15% probability]

=== Running Technical Expert Agent ===
📡 Gathering Real-Time Data...
🔍 Searching DuckDuckGo for technical analysis...
💰 Fetching Yahoo Finance financial data...
[Technical analysis with 60% probability]

--- Integration Preview ---
Weighted Probability: 48.75% | Disagreement≈22.5pp | N=4

=== Final Integrated Analysis ===
[Comprehensive synthesis of all perspectives with final probability]
```

## Future Expansion

Possible additions:
- 🎯 Sentiment Analysis Agent
- 📊 Market Maker Agent
- 🎲 Weighted prediction market system
- 📈 Historical performance tracking
- 🔄 Real-time market integration

## Quick Test

```bash
# Run the complete four-agent system
python meta_integrator_agent.py

# Test individual agents
python optimistic_investor_agent.py
python risk_control_investor_agent.py
python technical_expert_agent.py
python hater_agent.py
```

## Performance Notes

- First search may take 10-20 seconds (gathering real data from multiple sources)
- Web searches are rate-limited to avoid blocking
- Stock data updates in real-time during market hours
- Meta-integrator coordinates all agents for comprehensive analysis

---

Made with ❤️ for NVIDIA Hackathon  
**NOW WITH FOUR-AGENT SYSTEM AND REAL SEARCH CAPABILITIES!** 🚀

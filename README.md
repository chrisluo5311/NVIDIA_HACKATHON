# AI Betting Market Simulator

An AI prediction market simulator that uses multiple AI Agents with **REAL SEARCH CAPABILITIES** to simulate crowd intelligence predictions.

## 🎯 NEW: Real Search Capabilities!

Both AI Agents now have **actual internet search abilities**:

### 🔬 Technical Expert Agent
- ✅ **Real-time web search** - Searches latest news and information
- ✅ **Live financial data** - Fetches actual stock prices and market data
- ✅ **Company news** - Gathers recent company announcements and reports
- ✅ **Data-driven analysis** - Makes predictions based on REAL data

### 😤 Hater Agent
- ✅ **Reddit search** - Finds actual Reddit discussions and comments
- ✅ **Twitter/X search** - Discovers real tweets and opinions
- ✅ **Negative sentiment mining** - Collects genuine criticism and complaints
- ✅ **Real hate, real comments** - Bases opinions on actual online negativity

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. API Key is already built into the code, no additional configuration needed!

## Usage

**Run Technical Expert (with real data search):**
```bash
python run_technical_expert.py
```

**Run Hater (with real social media search):**
```bash
python run_hater.py
```

**Run Full Demo (compare both Agents with real data):**
```bash
python run_demo.py
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

### Technical Expert Agent Flow:
1. 📡 **Gathers Real Data**
   - Searches the web for latest information
   - Fetches live stock prices if relevant
   - Collects recent company news

2. 📊 **Analyzes Real Data**
   - AI analyzes the actual data gathered
   - Makes informed predictions based on facts
   - Provides probability and investment recommendation

3. 💰 **Returns Data-Driven Prediction**
   - Prediction backed by real research
   - Includes all source URLs for verification

### Hater Agent Flow:
1. 📡 **Searches for Negativity**
   - Scans Reddit discussions
   - Searches Twitter/X for complaints
   - Finds critical reviews and opinions

2. 💬 **Channels Real Hate**
   - AI reads actual negative comments
   - Mimics real hater sentiment
   - Provides pessimistic prediction

3. 😤 **Returns Hate-Driven Prediction**
   - Prediction influenced by real online negativity
   - Includes actual hater comments found

## Features

### Technical Expert Agent
- ✅ Uses lower temperature (0.6) to stay rational
- ✅ **REAL internet search** via DuckDuckGo
- ✅ **REAL stock data** via Yahoo Finance
- ✅ Structured analysis based on actual data
- ✅ Considers market, technology, finance, competition

### Hater Agent
- ✅ Uses higher temperature (0.9) for emotional responses
- ✅ **REAL Reddit discussions** via web search
- ✅ **REAL Twitter opinions** via social search
- ✅ Bases negativity on actual online comments
- ✅ Casual, emotional expression style

## Technical Details

### Search Tools (`search_tools.py`)
- **WebSearchTool** - DuckDuckGo search (no API key needed)
- **FinancialDataTool** - Yahoo Finance data (free)
- **SocialMediaTool** - Reddit & Twitter search (web scraping)

### No API Keys Required!
All search tools use free, public data sources:
- Web search via DuckDuckGo HTML
- Stock data via Yahoo Finance public API
- Social media via web scraping

## Example Output

```
🔬 Technical Expert AI Agent Starting Analysis...
   WITH REAL-TIME DATA SEARCH

📡 Gathering Real-Time Data...
🔍 Searching web for: Will Apple release Vision Pro 2...
   1. Vision Pro 2 Release Date, Features, and Upgrades...
   2. Apple Vision Pro 2 - all the rumors so far...
   3. Vision Pro 2 is coming soon with three upgrades...

💰 Fetching financial data for: AAPL...
   AAPL: $249.15

📰 Searching for company news...
   1. Apple announces new AR features...
   2. Vision Pro sales exceed expectations...

📊 Technical Expert's Analysis (Based on Real Data):
[AI analyzes the actual data gathered above...]
```

## Future Expansion

Possible additions:
- 🎯 Optimistic Investor Agent
- 🛡️ Risk Manager Agent
- 📊 Sentiment Analysis Agent
- 🎲 Weighted prediction market system

## Quick Test

```bash
# Set encoding (Windows CMD)
chcp 65001

# Run full demo with real search
python run_demo.py
```

## Performance Notes

- First search may take 10-20 seconds (gathering real data)
- Web searches are rate-limited to avoid blocking
- Stock data updates in real-time during market hours

---

Made with ❤️ for NVIDIA Hackathon  
**NOW WITH REAL SEARCH CAPABILITIES!** 🚀

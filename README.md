# NVIDIA Hackathon - Meta Integrator Agent

這是一個整合多個投資人代理觀點的 AI 系統，使用 NVIDIA Nemotron 模型來協調樂觀投資人和風險控制投資人的分析。

## 功能特色

- **多代理協調**: 同時運行樂觀投資人和風險控制投資人代理
- **觀點整合**: 使用 NVIDIA Llama-3.3 Nemotron Super 49B 模型整合不同觀點
- **機率計算**: 自動計算加權平均機率和分歧度
- **投資級分析**: 提供專業的投資建議和風險評估

## 文件結構

```
├── meta_integrator_agent.py      # 主要整合代理
├── optimistic_investor_agent.py   # 樂觀投資人代理
├── risk_control_investor_agent.py # 風險控制投資人代理
├── requirements.txt              # Python 依賴
├── test_meta_integrator.py       # 測試腳本
└── README.md                    # 說明文件
```

## 安裝與設置

1. **安裝依賴**:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **設置環境變數**:
   創建 `.env` 文件並添加你的 API 金鑰：
   ```
   NVIDIA_API_KEY=your_nvidia_api_key_here
   NEWSAPI_KEY=your_newsapi_key_here  # 可選
   ```

## 使用方法

### 交互式模式
```bash
python3 meta_integrator_agent.py
```

然後輸入你的投資問題，例如：
- "Will NVDA outperform SPY over the next quarter?"
- "What are the risks and opportunities for AAPL in the next 90 days?"
- "Should I invest in TSLA given current market conditions?"

### 測試模式
```bash
python3 test_meta_integrator.py
```

## 工作流程

1. **接收查詢**: 用戶輸入投資問題
2. **代理協調**: 
   - 運行樂觀投資人代理 (Optimistic Investor)
   - 運行風險控制投資人代理 (Risk-Control Investor)
3. **觀點整合**: 
   - 收集兩個代理的機率和推理
   - 計算加權平均機率和分歧度
   - 使用 Meta-Integrator 模型整合觀點
4. **最終分析**: 提供綜合的投資建議和機率評估

## 代理特色

### 樂觀投資人 (Optimistic Investor)
- 專注於上行潛力和催化劑
- 提供事件規劃和時間表
- 使用工具獲取新聞、股價和公司事件
- 給出樂觀但基於事實的評估

### 風險控制投資人 (Risk-Control Investor)
- 優先考慮資本保護和風險控制
- 量化風險和提供對沖建議
- 建立監控時間表和觸發機制
- 給出風險優先的評估

### Meta-Integrator
- 整合兩個代理的觀點
- 解決衝突並解釋分歧
- 提供最終的投資級綜合分析
- 給出單一的最終機率評估

## 技術架構

- **LLM**: NVIDIA Llama-3.3 Nemotron Super 49B v1.5
- **API**: NVIDIA Integrate API
- **工具**: yfinance, NewsAPI, RSS feeds
- **框架**: OpenAI SDK, Pydantic

## 範例輸出

```
>>> Meta-Integrator Agent (NVIDIA Llama-3.3 Nemotron Super 49B v1.5) <<<
Analyzing: Will NVDA outperform SPY over the next quarter?

=== Running Optimistic Investor ===
[樂觀投資人的分析...]

=== Running Risk-Control Investor ===
[風險控制投資人的分析...]

--- Integration Preview ---
Question: Will NVDA outperform SPY over the next quarter?
Inputs:
 • Optimistic Investor: 75.00%  (w=1.0)  — AI chip demand tailwinds...
 • Risk-Control Investor: 45.00%  (w=1.0)  — High valuation concerns...
Weighted Probability: 60.00%  | Disagreement≈15.00pp | N=2

=== Final Integrated Analysis ===
[整合後的最終分析...]
Overall Probability: 62.50%
```

## 注意事項

- 需要有效的 NVIDIA API 金鑰
- 某些功能需要網路連接來獲取即時數據
- 投資建議僅供參考，不構成投資建議
- 請根據自己的風險承受能力做出投資決策

## 故障排除

如果遇到問題，請檢查：
1. API 金鑰是否正確設置
2. 網路連接是否正常
3. 所有依賴是否已正確安裝
4. Python 版本是否為 3.9 或更高

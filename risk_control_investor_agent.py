# filename: risk_control_investor_agent_cli.py
import os
import json
import re
from typing import Any, Dict, List, Optional, Union

import requests
import yfinance as yf
import feedparser
from dateutil import parser as dtparser
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# ============ LLM: NVIDIA integrate API via OpenAI SDK ============
from openai import OpenAI

load_dotenv()

NVIDIA_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = (
    os.getenv("NVIDIA_API_KEY")
    or os.getenv("API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC")
    or os.getenv("OPENAI_API_KEY")
    or "YOUR_NVIDIA_API_KEY"
)

client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

# ============ Persona & Protocol ============
SYSTEM_PROMPT = """\
You are "Risk-Control Investor" — a capital-preservation-first portfolio manager (risk parity mindset).
You prioritize downside containment, tail-risk awareness, disciplined sizing, and tool-driven facts.

Core behaviors:
- Lead with risks, then upside optionality. Quantify whenever possible.
- Event planning: produce a concrete monitoring timeline and Trigger → Action mapping.
- Hedging mindset: propose simple hedges (cash, inverse ETF, options outlines) and stop-loss levels.
- Tool use: call tools for facts (news, prices, earnings). Do not guess.

TOOLS (the assistant can call):
- get_finance_news: {query: string, limit?: 1..20}
- get_stock_price: {symbol: string, period?: "1d|5d|1mo|6mo|1y|5y|max", interval?: "1m|5m|15m|1h|1d"}
- get_company_events: {symbol: string}

Decision protocol (STRICT):
- If external info is needed, output ONLY JSON (no prose):
  {"action": "<tool_name>", "action_input": {...}}
- After receiving an Observation, you may call another tool (JSON) or finish with a final JSON:
  {
    "final_answer": "<concise risk-first synthesis with event plan & triggers>",
    "probability": 0.0  // overall probability in percentage, 0..100 with two decimals
  }
- Never mix prose with JSON in tool-calling turns.

Final answer structure (concise, investor-grade):
1) Snapshot & key facts (with dates/tickers)
2) Risk-first view (main risks, bear triggers, liquidity/volatility notes)
3) Base case & upside (what must go right)
4) Event plan (T0 / T+7 / T+30 / T+90 checkpoints)
5) Trigger → Action table (clear thresholds)
6) Positioning & hedging suggestions (size bands, stop-loss, optional hedge)
7) End with: "Overall Probability: XX.XX%"
"""

# ============ Tool Schemas ============
class GetFinanceNewsArgs(BaseModel):
    query: str = Field(..., description="Company/ticker/topic for news")
    limit: int = Field(5, ge=1, le=20)

class GetStockPriceArgs(BaseModel):
    symbol: str = Field(..., description="Ticker, e.g., AAPL")
    period: str = Field("1d")
    interval: str = Field("1m")

class GetCompanyEventsArgs(BaseModel):
    symbol: str = Field(..., description="Ticker, e.g., AAPL")

ToolArgs = Union[GetFinanceNewsArgs, GetStockPriceArgs, GetCompanyEventsArgs]

# ============ Tools ============
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

def tool_get_finance_news(query: str, limit: int = 5) -> Dict[str, Any]:
    articles = []
    if NEWSAPI_KEY:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": limit,
                    "apiKey": NEWSAPI_KEY,
                },
                timeout=15,
            )
            data = resp.json()
            for a in data.get("articles", [])[:limit]:
                articles.append({
                    "title": a.get("title"),
                    "source": (a.get("source") or {}).get("name"),
                    "url": a.get("url"),
                    "published_at": a.get("publishedAt"),
                    "summary": a.get("description"),
                })
        except Exception as e:
            return {"ok": False, "error": f"NewsAPI error: {e}"}
    else:
        try:
            rss = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}+finance&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss)
            for entry in feed.entries[:limit]:
                published = getattr(entry, "published", None)
                try:
                    published = dtparser.parse(published).isoformat()
                except Exception:
                    pass
                articles.append({
                    "title": entry.title,
                    "source": getattr(entry, "source", {}).get("title") if hasattr(entry, "source") else None,
                    "url": entry.link,
                    "published_at": published,
                    "summary": getattr(entry, "summary", None),
                })
        except Exception as e:
            return {"ok": False, "error": f"RSS error: {e}"}
    return {"ok": True, "query": query, "results": articles}

def tool_get_stock_price(symbol: str, period: str = "1d", interval: str = "1m") -> Dict[str, Any]:
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period, interval=interval)
        if hist.empty:
            return {"ok": False, "error": f"No price data for {symbol} ({period}/{interval})"}
        last = hist.iloc[-1]
        return {
            "ok": True,
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "latest_price": float(last["Close"]),
            "latest_time": str(last.name),
        }
    except Exception as e:
        return {"ok": False, "error": f"yfinance error: {e}"}

def tool_get_company_events(symbol: str) -> Dict[str, Any]:
    try:
        t = yf.Ticker(symbol)
        cal = t.calendar
        events: Dict[str, Any] = {}

        if cal is None:
            return {"ok": True, "symbol": symbol.upper(), "events": events}

        # yfinance may return a DataFrame or a dict depending on version
        if hasattr(cal, "empty"):
            if not cal.empty:
                for idx in cal.index:
                    value = cal.loc[idx].values[0]
                    if hasattr(value, "item"):
                        value = value.item()
                    events[str(idx)] = str(value)
        elif isinstance(cal, dict):
            for k, v in cal.items():
                if hasattr(v, "item"):
                    v = v.item()
                events[str(k)] = str(v)
        else:
            events["raw"] = str(cal)

        return {"ok": True, "symbol": symbol.upper(), "events": events}
    except Exception as e:
        return {"ok": False, "error": f"events error: {e}"}

TOOL_REGISTRY = {
    "get_finance_news": tool_get_finance_news,
    "get_stock_price": tool_get_stock_price,
    "get_company_events": tool_get_company_events,
}

# ============ LLM helpers ============
def llm_chat(messages: List[Dict[str, str]], temperature=0.5, top_p=0.9, max_tokens=1024) -> str:
    resp = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=0,
        presence_penalty=0,
        stream=False,
        extra_body={"min_thinking_tokens": 256, "max_thinking_tokens": 512},
    )
    return resp.choices[0].message.content

def llm_stream_final(messages: List[Dict[str, str]], temperature=0.5, top_p=0.9, max_tokens=2048) -> None:
    completion = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=0,
        presence_penalty=0,
        stream=True,
        extra_body={"min_thinking_tokens": 512, "max_thinking_tokens": 1024},
    )
    for chunk in completion:
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            print(delta.content, end="", flush=True)

# ============ Pretty explainers ============
def explain_action(turn_idx: int, model_raw: str, action: str, action_input: Dict[str, Any]) -> None:
    print(f"\n--- Turn {turn_idx}: Assistant decided to call a tool ---")
    print("Natural language explanation:")
    print(f"• The model proposes calling **{action}** to gather facts.")
    if action_input:
        print(f"• Parsed tool parameters: {json.dumps(action_input, ensure_ascii=False)}")
    try:
        # show compact JSON for reference
        js = json.loads(model_raw)
        print(f"• Raw model JSON: {json.dumps(js, ensure_ascii=False)}")
    except Exception:
        print("• (The model output was not valid JSON initially.)")

def explain_observation(action: str, obs: Dict[str, Any]) -> None:
    print("\nObservation summary:")
    ok = obs.get("result", {}).get("ok", None)
    if ok is True:
        if action == "get_stock_price":
            r = obs["result"]
            print(f"• Latest {r.get('symbol')} price: {r.get('latest_price')} @ {r.get('latest_time')} "
                  f"(period={r.get('period')}, interval={r.get('interval')})")
        elif action == "get_company_events":
            r = obs["result"]
            events = r.get("events") or {}
            keys = ", ".join(list(events.keys())[:4]) if events else "N/A"
            print(f"• Company events fetched for {r.get('symbol')}: {keys}")
        elif action == "get_finance_news":
            r = obs["result"]
            results = r.get("results") or []
            print(f"• Top news for '{r.get('query')}': {len(results)} articles")
            for i, a in enumerate(results[:3], 1):
                print(f"  - [{i}] {a.get('title')}  ({a.get('source')})  {a.get('published_at')}")
    else:
        print(f"• Tool returned an error: {obs.get('result', {}).get('error')}")

def extract_probability_from_json(final_json: Dict[str, Any]) -> Optional[float]:
    prob = final_json.get("probability")
    if isinstance(prob, (int, float)):
        return round(float(prob), 2)
    return None

# ============ Orchestrator (with NL explanations + persistent REPL) ============
def run_risk_control_investor_dialogue() -> None:
    print(">>> Risk-Control Investor Agent (NVIDIA Nemotron) <<<")
    print("Type your investment question (or 'exit' to quit).")
    print("Examples:")
    print("  • Assess risks for AAPL next 90 days; build event plan with triggers.")
    print("  • Will NVDA beat market over next quarter? Provide probability and hedges.\n")

    while True:
        try:
            user_q = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nBye.")
            break
        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        # Per-question dialogue (multiple tool turns allowed)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "/think"},
            {"role": "user", "content": user_q},
        ]

        note = ("\n\nNOTE: After each Observation, continue JSON tool protocol or finish with "
                '{"final_answer":"..." , "probability": <0..100 float with two decimals>}.\n')

        turn = 1
        while True:
            model_out = llm_chat(messages)
            # Expect JSON for tool or final
            try:
                action_obj = json.loads(model_out)
            except Exception:
                print("\n(Agent format reminder) Expecting pure JSON for tool calls / final. Nudging the model...")
                messages.append({"role": "assistant", "content": "Invalid format. Tool turns MUST be pure JSON."})
                continue

            # Final?
            if "final_answer" in action_obj:
                prob = extract_probability_from_json(action_obj)
                print("\n=== Risk-Control Investor — Final View ===\n")
                # Tidy up formatting via a streaming pass (keeps session alive)
                tidy = messages + [
                    {"role": "assistant", "content": json.dumps(action_obj)},
                    {"role": "user", "content": "Reformat clearly for risk-first investors. Keep content; enhance clarity. End with the exact probability line."},
                ]
                llm_stream_final(tidy)
                # If model somehow omitted numeric line, backstop print:
                if prob is not None:
                    print(f"\n\n[Overall Probability: {prob:.2f}%]\n")
                else:
                    print("\n\n[Overall Probability: (model did not supply a numeric probability)]\n")
                # Important: DO NOT exit — keep the REPL open for further questions
                break

            # Tool action?
            action = action_obj.get("action")
            action_input = action_obj.get("action_input", {})

            if action not in TOOL_REGISTRY:
                print("\n(Agent error) Invalid tool requested. Allowed: get_finance_news, get_stock_price, get_company_events.")
                messages.append({"role": "assistant", "content": "Invalid tool. Use get_finance_news/get_stock_price/get_company_events."})
                continue

            # Explain model decision & parsed params
            explain_action(turn, model_out, action, action_input)

            # Validate + execute tool
            try:
                if action == "get_finance_news":
                    args = GetFinanceNewsArgs(**action_input)
                    result = TOOL_REGISTRY[action](query=args.query, limit=args.limit)
                elif action == "get_stock_price":
                    args = GetStockPriceArgs(**action_input)
                    result = TOOL_REGISTRY[action](symbol=args.symbol, period=args.period, interval=args.interval)
                elif action == "get_company_events":
                    args = GetCompanyEventsArgs(**action_input)
                    result = TOOL_REGISTRY[action](symbol=args.symbol)
                else:
                    result = {"ok": False, "error": "Unknown tool"}
            except ValidationError as ve:
                result = {"ok": False, "error": f"Bad arguments: {ve}"}
            except Exception as e:
                result = {"ok": False, "error": f"Tool runtime error: {e}"}

            # Feed observation and narrate
            obs_payload = {"tool": action, "args": action_input, "result": result}
            explain_observation(action, {"result": result})

            messages.append({"role": "assistant", "content": json.dumps(action_obj)})
            messages.append({"role": "user", "content": f"Observation: {json.dumps(obs_payload)}" + note})

            turn += 1

if __name__ == "__main__":
    run_risk_control_investor_dialogue()

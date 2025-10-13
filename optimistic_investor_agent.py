import os
import json
import time
from typing import Any, Dict, List, Optional, Union

import requests
import yfinance as yf
import feedparser
from dateutil import parser as dtparser
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# ========= LLM: NVIDIA integrate API via OpenAI SDK =========
from openai import OpenAI

load_dotenv()

NVIDIA_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC") or os.getenv("OPENAI_API_KEY")

client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY or "YOUR_NVIDIA_API_KEY"
)

# ========= Persona: 樂觀投資人（含工具協作與事件規劃能力） =========
SYSTEM_PROMPT = """\
You are "Optimistic Investor" — a pragmatic, upbeat buy-side analyst who believes in long-run innovation compounding, but avoids naive hopium.

Core traits:
- Optimism with discipline: highlight upside scenarios, catalysts, optionality; also list main risks and mitigating steps.
- Event planning: structure near-term catalysts into an actionable timeline (what to monitor, when, and how to react).
- Tool use: when you need facts, call tools instead of guessing.
- Probability clarity: every final answer must include a clear percentage probability for the core thesis/outcome.

Decision protocol (VERY IMPORTANT):
- If you need external info (news, prices, earnings dates), output ONLY a JSON dict:
  {"action": "<tool_name>", "action_input": {...}}
  Valid tool_name: "get_finance_news", "get_stock_price", "get_company_events"
- After receiving an Observation from the tool, continue reasoning and either call more tools or finish with:
  {"final_answer": "<your concise, optimistic-but-grounded investment take including event plan>"}
- Never mix prose with JSON in tool-calling steps. For final_answer, return prose (no JSON).

Output quality:
- Be specific: name tickers, dates (YYYY-MM-DD), concrete metrics if available.
- Event plan structure: T0 (today actions), T+7/30/90 (what to check), Trigger → Action mapping.
- Keep it brief, punchy, and investor-grade.
"""

# ========= Tool Schemas for safety parsing =========
class GetFinanceNewsArgs(BaseModel):
    query: str = Field(..., description="Company name, ticker, or topic to search news for")
    limit: int = Field(5, ge=1, le=20)

class GetStockPriceArgs(BaseModel):
    symbol: str = Field(..., description="Ticker symbol, e.g., AAPL")
    period: str = Field("1d", description="yfinance period, e.g., 1d,5d,1mo,6mo,1y,5y,max")
    interval: str = Field("1m", description="yfinance interval, e.g., 1m,5m,15m,1h,1d")

class GetCompanyEventsArgs(BaseModel):
    symbol: str = Field(..., description="Ticker symbol, e.g., AAPL")

ToolArgs = Union[GetFinanceNewsArgs, GetStockPriceArgs, GetCompanyEventsArgs]

# ========= Tools: Finance News / Stock Price / Company Events =========
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

def tool_get_finance_news(query: str, limit: int = 5) -> Dict[str, Any]:
    """
    Strategy:
    1) If NEWSAPI_KEY available, use NewsAPI (business endpoints).
    2) Else fallback to free RSS (e.g., Yahoo Finance, Google News).
    """
    articles: List[Dict[str, Any]] = []

    if NEWSAPI_KEY:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "apiKey": NEWSAPI_KEY,
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            for a in data.get("articles", [])[:limit]:
                articles.append({
                    "title": a.get("title"),
                    "source": a.get("source", {}).get("name"),
                    "url": a.get("url"),
                    "published_at": a.get("publishedAt"),
                    "summary": a.get("description"),
                })
        except Exception as e:
            return {"ok": False, "error": f"NewsAPI error: {e}"}
    else:
        # Fallback: Google News RSS (finance flavor via topic query)
        rss_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}+finance&hl=en-US&gl=US&ceid=US:en"
        try:
            feed = feedparser.parse(rss_url)
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
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if hist.empty:
            return {"ok": False, "error": f"No price data for {symbol} ({period}/{interval})"}
        latest_row = hist.iloc[-1]
        price = float(latest_row["Close"])
        return {
            "ok": True,
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "latest_price": price,
            "latest_time": str(latest_row.name),
        }
    except Exception as e:
        return {"ok": False, "error": f"yfinance error: {e}"}

def tool_get_company_events(symbol: str) -> Dict[str, Any]:
    """
    Pulls basic calendar info: earnings dates where available.
    Note: yfinance calendar fields vary by ticker support.
    """
    try:
        t = yf.Ticker(symbol)
        cal = t.calendar  # pandas dataframe w/ rows like 'Earnings Date'
        events = {}
        if cal is not None and not cal.empty:
            for idx in cal.index:
                key = str(idx)
                val = cal.loc[idx].values[0]
                events[key] = str(val)
        return {"ok": True, "symbol": symbol.upper(), "events": events}
    except Exception as e:
        return {"ok": False, "error": f"events error: {e}"}

# Tool registry
TOOL_REGISTRY = {
    "get_finance_news": tool_get_finance_news,
    "get_stock_price": tool_get_stock_price,
    "get_company_events": tool_get_company_events,
}

# ========= LLM helpers =========
def llm_chat(messages: List[Dict[str, str]], temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 1024) -> str:
    """
    Simpler non-streaming call for agent control turns (tool selection, JSON parsing).
    """
    resp = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=0,
        presence_penalty=0,
        # Do not stream for action turns (easier to parse)
        stream=False,
        extra_body={
            "min_thinking_tokens": 256,
            "max_thinking_tokens": 512
        }
    )
    return resp.choices[0].message.content

def llm_stream_final(messages: List[Dict[str, str]], temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 2048) -> None:
    """
    Optional pretty streaming for final investor-grade synthesis.
    """
    completion = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=0,
        presence_penalty=0,
        stream=True,
        extra_body={
            "min_thinking_tokens": 512,
            "max_thinking_tokens": 1024
        }
    )
    for chunk in completion:
        # If the model emits reasoning_content, we ignore printing it to avoid exposing chain-of-thought logs.
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            print(delta.content, end="", flush=True)

# ========= Agent Orchestrator (ReAct-style loop) =========
def run_optimistic_investor(query: str, max_turns: int = 20, stream_final: bool = True, verbose: bool = True) -> None:
    """
    Orchestrates tool calls until the model returns {"final_answer": "..."}.
    Prints the final answer (streamed by default).
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]

    observation_note = ("\n\nNOTE to assistant: When you receive tool results (Observation), "
                        "continue your decision protocol. Either call another tool with JSON or finish with final_answer JSON.\n")

    for turn in range(max_turns):
        model_out = llm_chat(messages)

        if verbose:
            print(f"\n=== Turn {turn + 1} — Model Proposal ===")
            print("The model responded with the following plan in structured JSON (raw output below):")
            print(model_out)

        # Try to parse JSON action
        action_dict: Optional[Dict[str, Any]] = None
        try:
            action_dict = json.loads(model_out)
        except Exception:
            # If it's not JSON, nudge the model
            messages.append({"role": "assistant", "content": "Invalid format. Remember: tool step must be pure JSON."})
            if verbose:
                print("The response was not valid JSON, so we reminded the model to answer with pure JSON.")
            continue

        # Final answer?
        if "final_answer" in action_dict:
            final_text = action_dict["final_answer"]
            if verbose:
                print("\nThe model has provided a final answer draft (before polishing):")
                print(final_text)
            if stream_final:
                # Stream a polished recap using the model (optional)
                final_messages = messages + [
                    {"role": "assistant", "content": model_out},  # the JSON final from earlier step
                    {"role": "user", "content": "Format the final answer cleanly for an investor, preserving content but improving flow. Explicitly state the probability (in %) of the primary investment thesis in the final paragraphs."}
                ]
                print("\n=== Optimistic Investor — Final View ===\n")
                llm_stream_final(final_messages)
                print("\n")
            else:
                print("\n=== Optimistic Investor — Final View ===\n")
                print(final_text)
            return

        # Tool action?
        action = action_dict.get("action")
        action_input = action_dict.get("action_input", {})

        if verbose:
            print("\nThe model wants to use a tool. Interpreted intent:")
            print(json.dumps({"action": action, "action_input": action_input}, indent=2))
            if action in TOOL_REGISTRY:
                print(f"Explanation: call '{action}' with the above arguments to gather supporting data.")

        if action not in TOOL_REGISTRY:
            messages.append({"role": "assistant", "content": "Invalid tool. Use one of: get_finance_news, get_stock_price, get_company_events"})
            if verbose:
                print("The requested tool is not supported, so we asked the model to try again.")
            continue

        # Validate and execute according to tool schema
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

        # Feed Observation back
        obs_payload = {
            "tool": action,
            "args": action_input,
            "result": result
        }
        messages.append({"role": "assistant", "content": json.dumps(action_dict)})  # echo the action JSON
        messages.append({"role": "user", "content": f"Observation: {json.dumps(obs_payload)}" + observation_note})

        if verbose:
            print("Tool observation received and relayed back to the model:")
            print(json.dumps(obs_payload, indent=2))
            if result.get("ok"):
                print("Summary: tool succeeded, the model will use this info in the next turn.")
            else:
                print("Summary: tool reported an error; the model will need to adjust its plan.")

    # If we exit the loop without final_answer:
    print("\nReached max turns without a final answer. You can increase max_turns.\n")

# ========= Example CLI =========
if __name__ == "__main__":
    print(">>> Optimistic Investor Agent (NVIDIA Nemotron) <<<")
    print("Enter an investment question, e.g.:")
    print("  Will AAPL outperform SPY over the next quarter?")
    print("  What are the catalysts for NVDA in the next 90 days?")

    try:
        while True:
            try:
                user_q = input("\nYour question (or type 'exit' to quit): ").strip()
            except EOFError:
                print("\nInput stream closed. Exiting.")
                break
            except KeyboardInterrupt:
                print("\nBye.")
                break

            if not user_q:
                print("Using default question about NVDA outperformance and event plan.")
                user_q = "Will NVDA outperform over the next quarter? Draft an event plan with concrete checkpoints."

            if user_q.lower() in {"exit", "quit"}:
                print("Exiting. See you next time.")
                break

            try:
                run_optimistic_investor(user_q, max_turns=30, stream_final=True, verbose=True)
            except Exception as run_err:
                print(f"Error while running agent: {run_err}")
    finally:
        pass

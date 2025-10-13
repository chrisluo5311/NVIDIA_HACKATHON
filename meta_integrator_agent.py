# filename: meta_integrator_agent.py
import os
import json
import subprocess
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tempfile
import threading
import queue
import time
import io
from contextlib import redirect_stdout, redirect_stderr

from pydantic import BaseModel, Field, ValidationError
from typing import List as TypingList
from dotenv import load_dotenv

load_dotenv()

# ============ NVIDIA integrate API via OpenAI SDK ============
from openai import OpenAI

NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = (
    os.getenv("NVIDIA_API_KEY")
    or os.getenv("API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC")
    or os.getenv("OPENAI_API_KEY")
    or "YOUR_NVIDIA_API_KEY"
)

client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

# ============ Import Agent Functions ============
# Import the agent functions from the other modules
from optimistic_investor_agent import run_optimistic_investor, llm_chat as optimistic_llm_chat, SYSTEM_PROMPT as OPTIMISTIC_SYSTEM_PROMPT
from risk_control_investor_agent import run_risk_control_investor_dialogue, llm_chat as risk_control_llm_chat, SYSTEM_PROMPT as RISK_CONTROL_SYSTEM_PROMPT

# ============ Schema ============
class AgentReport(BaseModel):
    name: str = Field(..., min_length=1)
    probability: float = Field(..., ge=0, le=100)  # percentage
    reasoning: str = Field(..., min_length=3)
    weight: Optional[float] = Field(None, gt=0)  # optional weight; default 1.0

class AggregationInput(BaseModel):
    question: str = Field(..., min_length=3)
    reports: TypingList[AgentReport] = Field(..., min_length=1)

# ============ System Prompt ============
SYSTEM_PROMPT = """\
You are "Meta-Integrator" — a senior meta-analyst that combines multiple agents' forecasts.
Your job:
- Read each agent's probability and reasoning.
- Compute a coherent final view: reconcile conflicts, highlight consensus vs. disagreement.
- Use your own broad knowledge to stress test their reasoning (but do not fabricate facts).
- Output a crisp investor-grade synthesis with a single final probability.

Rules:
- Be concise and precise. Avoid hedging clichés; show specific drivers and uncertainties.
- If agents disagree, explain why and which evidence should dominate.
- End with a single line: "Overall Probability: XX.XX%"
"""

# ============ Agent Coordination Functions ============
def run_agent_with_capture(agent_name: str, query: str, max_turns: int = 10) -> Dict[str, Any]:
    """
    Run an agent and capture its output to extract reasoning and probability.
    """
    print(f"\n=== Running {agent_name} ===")
    
    # Capture stdout to get the agent's output
    captured_output = io.StringIO()
    
    try:
        with redirect_stdout(captured_output):
            if agent_name == "Optimistic Investor":
                # Run optimistic investor
                run_optimistic_investor(query, max_turns=max_turns, stream_final=False, verbose=False)
            elif agent_name == "Risk-Control Investor":
                # For risk control investor, we need to simulate the dialogue
                messages = [
                    {"role": "system", "content": RISK_CONTROL_SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ]
                
                # Run a simplified version that gets to final answer
                for turn in range(max_turns):
                    model_out = risk_control_llm_chat(messages)
                    
                    try:
                        action_dict = json.loads(model_out)
                        if "final_answer" in action_dict:
                            # Extract probability if available
                            prob = action_dict.get("probability", 50.0)  # default if not provided
                            final_text = action_dict["final_answer"]
                            
                            # Format the output similar to optimistic investor
                            print(f"\n=== {agent_name} — Final View ===\n")
                            print(final_text)
                            if "probability" in action_dict:
                                print(f"\n[Overall Probability: {prob:.2f}%]\n")
                            break
                        else:
                            # Handle tool calls (simplified - just continue)
                            messages.append({"role": "assistant", "content": model_out})
                            messages.append({"role": "user", "content": "Please provide a final answer with your reasoning and probability assessment."})
                    except json.JSONDecodeError:
                        # If not JSON, treat as final answer
                        print(f"\n=== {agent_name} — Final View ===\n")
                        print(model_out)
                        break
                        
    except Exception as e:
        print(f"Error running {agent_name}: {e}")
        return {
            "name": agent_name,
            "probability": 50.0,
            "reasoning": f"Error occurred while running agent: {str(e)}",
            "weight": 1.0
        }
    
    # Get the captured output
    output = captured_output.getvalue()
    
    # Extract probability and reasoning from output
    probability = 50.0  # default
    reasoning = output.strip()
    
    # Try to extract probability from the output
    import re
    prob_match = re.search(r'Overall Probability:\s*(\d+\.?\d*)%', output)
    if prob_match:
        probability = float(prob_match.group(1))
    
    # Clean up reasoning text
    if len(reasoning) > 500:
        reasoning = reasoning[:500] + "..."
    
    return {
        "name": agent_name,
        "probability": probability,
        "reasoning": reasoning,
        "weight": 1.0
    }

def run_optimistic_agent_direct(query: str) -> Dict[str, Any]:
    """
    Run optimistic investor agent directly and extract results.
    """
    print(f"\n=== Running Optimistic Investor ===")
    
    # Create a simplified version that gets to final answer quickly
    messages = [
        {"role": "system", "content": OPTIMISTIC_SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    
    # Run a few turns to get to final answer
    for turn in range(5):
        model_out = optimistic_llm_chat(messages)
        
        try:
            action_dict = json.loads(model_out)
            if "final_answer" in action_dict:
                final_text = action_dict["final_answer"]
                
                # Extract probability from the text
                import re
                prob_match = re.search(r'(\d+\.?\d*)%', final_text)
                probability = float(prob_match.group(1)) if prob_match else 50.0
                
                print(f"\n=== Optimistic Investor — Final View ===\n")
                print(final_text)
                print(f"\n[Overall Probability: {probability:.2f}%]\n")
                
                return {
                    "name": "Optimistic Investor",
                    "probability": probability,
                    "reasoning": final_text[:500] + "..." if len(final_text) > 500 else final_text,
                    "weight": 1.0
                }
            else:
                # Handle tool calls - for simplicity, just ask for final answer
                messages.append({"role": "assistant", "content": model_out})
                messages.append({"role": "user", "content": "Please provide your final investment assessment with reasoning and probability."})
        except json.JSONDecodeError:
            # If not JSON, treat as final answer
            print(f"\n=== Optimistic Investor — Final View ===\n")
            print(model_out)
            print(f"\n[Overall Probability: 50.00%]\n")
            
            return {
                "name": "Optimistic Investor",
                "probability": 50.0,
                "reasoning": model_out[:500] + "..." if len(model_out) > 500 else model_out,
                "weight": 1.0
            }
    
    # Fallback if we don't get a final answer
    return {
        "name": "Optimistic Investor",
        "probability": 50.0,
        "reasoning": "Unable to get complete assessment from optimistic investor",
        "weight": 1.0
    }

def run_risk_control_agent_direct(query: str) -> Dict[str, Any]:
    """
    Run risk control investor agent directly and extract results.
    """
    print(f"\n=== Running Risk-Control Investor ===")
    
    # Create a simplified version that gets to final answer quickly
    messages = [
        {"role": "system", "content": RISK_CONTROL_SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    
    # Run a few turns to get to final answer
    for turn in range(5):
        model_out = risk_control_llm_chat(messages)
        
        try:
            action_dict = json.loads(model_out)
            if "final_answer" in action_dict:
                final_text = action_dict["final_answer"]
                probability = action_dict.get("probability", 50.0)
                
                print(f"\n=== Risk-Control Investor — Final View ===\n")
                print(final_text)
                print(f"\n[Overall Probability: {probability:.2f}%]\n")
                
                return {
                    "name": "Risk-Control Investor",
                    "probability": float(probability),
                    "reasoning": final_text[:500] + "..." if len(final_text) > 500 else final_text,
                    "weight": 1.0
                }
            else:
                # Handle tool calls - for simplicity, just ask for final answer
                messages.append({"role": "assistant", "content": model_out})
                messages.append({"role": "user", "content": "Please provide your final risk assessment with reasoning and probability."})
        except json.JSONDecodeError:
            # If not JSON, treat as final answer
            print(f"\n=== Risk-Control Investor — Final View ===\n")
            print(model_out)
            print(f"\n[Overall Probability: 50.00%]\n")
            
            return {
                "name": "Risk-Control Investor",
                "probability": 50.0,
                "reasoning": model_out[:500] + "..." if len(model_out) > 500 else model_out,
                "weight": 1.0
            }
    
    # Fallback if we don't get a final answer
    return {
        "name": "Risk-Control Investor",
        "probability": 50.0,
        "reasoning": "Unable to get complete assessment from risk control investor",
        "weight": 1.0
    }

# ============ Helper: aggregation ============
def aggregate_probability(reports: List[AgentReport]) -> Dict[str, Any]:
    # Default equal weights unless provided
    weights = [float(r.weight) if r.weight else 1.0 for r in reports]
    probs = [float(r.probability) for r in reports]
    total_w = sum(weights)
    weighted = sum(w * p for w, p in zip(weights, probs)) / total_w if total_w > 0 else sum(probs) / len(probs)

    # simple disagreement metric: weighted std-like magnitude
    mean = weighted
    var = sum(w * (p - mean) ** 2 for w, p in zip(weights, probs)) / total_w if total_w > 0 else 0.0
    disagreement = var ** 0.5  # pseudo-std (percentage points)

    return {
        "weighted_probability": round(weighted, 2),
        "weights": weights,
        "disagreement": round(disagreement, 2),
        "count": len(reports),
    }

# ============ LLM Calls ============
def llm_stream_synthesis(question: str, reports: List[AgentReport], agg: Dict[str, Any]) -> None:
    """
    Streams the final integrator reasoning. Suppresses hidden chain-of-thought.
    """
    # Prepare a compact, model-friendly payload
    payload = {
        "question": question,
        "reports": [
            {"name": r.name, "probability": r.probability, "reasoning": r.reasoning, "weight": float(r.weight) if r.weight else 1.0}
            for r in reports
        ],
        "aggregate": agg,
        "instructions": {
            "style": "concise investor-grade synthesis",
            "requirements": [
                "Reconcile conflicts; explain why.",
                "Incorporate model knowledge prudently; no hallucinated specifics.",
                "Conclude with 'Overall Probability: XX.XX%' where XX.XX equals the best integrated estimate (two decimals)."
            ]
        }
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "/think"},
        {"role": "user", "content": "Integrate these agent forecasts into a final view:\n" + json.dumps(payload, ensure_ascii=False)}
    ]

    completion = client.chat.completions.create(
        model=NVIDIA_MODEL,
        messages=messages,
        temperature=0.5,
        top_p=0.9,
        max_tokens=1600,
        frequency_penalty=0,
        presence_penalty=0,
        stream=True,
        extra_body={
            "min_thinking_tokens": 512,
            "max_thinking_tokens": 1024
        }
    )
    for chunk in completion:
        delta = chunk.choices[0].delta
        # IMPORTANT: do not print reasoning_content (hidden chain-of-thought).
        if getattr(delta, "content", None):
            print(delta.content, end="", flush=True)

# ============ Pretty helper ============
def pretty_preview(question: str, reports: List[AgentReport], agg: Dict[str, Any]) -> None:
    print("\n--- Integration Preview ---")
    print(f"Question: {question}")
    print("Inputs:")
    for r in reports:
        w = r.weight if r.weight else 1.0
        print(f" • {r.name}: {r.probability:.2f}%  (w={w})  — {r.reasoning[:120]}{'...' if len(r.reasoning)>120 else ''}")
    print(f"Weighted Probability: {agg['weighted_probability']:.2f}%  | Disagreement≈{agg['disagreement']:.2f}pp | N={agg['count']}")

# ============ Main Integration Function ============
def run_meta_integrator(query: str) -> None:
    """
    Main function that coordinates both agents and integrates their responses.
    """
    print(">>> Meta-Integrator Agent (NVIDIA Llama-3.3 Nemotron Super 49B v1.5) <<<")
    print(f"Analyzing: {query}")
    print("\n" + "="*60)
    
    # Run both agents
    optimistic_report = run_optimistic_agent_direct(query)
    risk_control_report = run_risk_control_agent_direct(query)
    
    # Convert to AgentReport objects
    reports = [
        AgentReport(**optimistic_report),
        AgentReport(**risk_control_report)
    ]
    
    # Calculate aggregation
    agg = aggregate_probability(reports)
    
    # Show preview
    pretty_preview(query, reports, agg)
    
    # Stream final synthesis
    print("\n=== Final Integrated Analysis ===\n")
    llm_stream_synthesis(query, reports, agg)
    print("\n")

# ============ REPL ============
def run_meta_integrator_dialogue() -> None:
    print(">>> Meta-Integrator Agent (NVIDIA Llama-3.3 Nemotron Super 49B v1.5) <<<")
    print("Enter your investment question, and I'll coordinate both Optimistic and Risk-Control investors.")
    print("Examples:")
    print("  • Will NVDA outperform SPY over the next quarter?")
    print("  • What are the risks and opportunities for AAPL in the next 90 days?")
    print("  • Should I invest in TSLA given current market conditions?")
    print("\nType 'exit' to quit.\n")

    while True:
        try:
            user_q = input("Your investment question: ").strip()
        except KeyboardInterrupt:
            print("\nBye.")
            break
        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        try:
            run_meta_integrator(user_q)
        except Exception as e:
            print(f"Error during integration: {e}")
            print("Please try again with a different question.\n")

if __name__ == "__main__":
    run_meta_integrator_dialogue()
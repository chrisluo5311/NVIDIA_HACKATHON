# -*- coding: utf-8 -*-
"""
Launch script for Technical Expert Agent (fixes Chinese encoding issues)
"""
import sys
import io

# Set standard output encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

from technical_expert_agent import TechnicalExpertAgent

def main():
    """Test Technical Expert Agent"""
    # Create Agent
    agent = TechnicalExpertAgent()
    
    # Test question
    try:
        question = input("Enter a question to analyze (e.g., Will Apple release Vision Pro 2?): ")
        if not question.strip():
            question = "Will Apple release Vision Pro 2 in 2025?"
            print(f"Using default question: {question}")
    except:
        question = "Will Apple release Vision Pro 2 in 2025?"
        print(f"Using default question: {question}")
    
    # Perform analysis
    result = agent.analyze(question)
    
    # Display result summary
    if result:
        print(f"\n{'='*60}")
        print("📋 Analysis Result Summary")
        print(f"{'='*60}")
        print(f"Agent Type: {result['agent_type']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['probability']}%")
        print(f"Recommended Bet: ${result['bet_amount']}")
        print(f"Core Reasoning: {result['reasoning']}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Launch script for Hater Agent (fixes Chinese encoding issues)
"""
import sys
import io

# Set standard output encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

from hater_agent import HaterAgent

def main():
    """Test Hater Agent"""
    # Create Agent
    agent = HaterAgent()
    
    # Test question
    try:
        question = input("Enter a question for the hater to comment on (e.g., Will Apple release Vision Pro 2?): ")
        if not question.strip():
            question = "Will Apple release Vision Pro 2 in 2025?"
            print(f"Using default question: {question}")
    except:
        question = "Will Apple release Vision Pro 2 in 2025?"
        print(f"Using default question: {question}")
    
    # Perform "analysis"
    result = agent.analyze(question)
    
    # Display result summary
    if result:
        print(f"\n{'='*60}")
        print("📋 Hater Comment Summary")
        print(f"{'='*60}")
        print(f"Agent Type: {result['agent_type']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['probability']}%")
        print(f"Dare to Bet: ${result['bet_amount']}")
        print(f"Hater Summary: {result['reasoning']}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

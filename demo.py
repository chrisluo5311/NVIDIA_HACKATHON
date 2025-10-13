"""
Demo script - Shows how to use the two AI Agents
"""

from technical_expert_agent import TechnicalExpertAgent
from hater_agent import HaterAgent

def run_demo():
    """Run demo"""
    print("="*70)
    print("🎯 AI Betting Market Simulator - Dual Agent Demo")
    print("="*70)
    print()
    
    # Get question
    question = input("Enter a question to predict (press Enter for default): ")
    if not question.strip():
        question = "Will Apple release Apple Vision Pro 2 (MR headset generation 2) in 2025?"
        print(f"\nUsing default question: {question}\n")
    
    # Create two Agents
    print("Initializing AI Agents...\n")
    technical_expert = TechnicalExpertAgent()
    hater = HaterAgent()
    
    # Technical expert analysis
    print("\n" + "="*70)
    print("First Agent: Technical Expert")
    print("="*70)
    expert_result = technical_expert.analyze(question)
    
    # Hater comments
    print("\n" + "="*70)
    print("Second Agent: Hater")
    print("="*70)
    hater_result = hater.analyze(question)
    
    # Display comparison summary
    print("\n" + "="*70)
    print("📊 Agent Prediction Comparison")
    print("="*70)
    
    if expert_result and hater_result:
        print(f"\nQuestion: {question}\n")
        
        print("┌─────────────────────────────────────────────────────────────┐")
        print("│ Technical Expert 🔬                                         │")
        print("├─────────────────────────────────────────────────────────────┤")
        print(f"│ Prediction: {expert_result['prediction']:<20s}                   │")
        print(f"│ Probability: {expert_result['probability']:3d}%                                      │")
        print(f"│ Bet: ${expert_result['bet_amount']:4d} / $1000                                   │")
        print(f"│ Reason: {expert_result['reasoning'][:45]:45s}│")
        print("└─────────────────────────────────────────────────────────────┘")
        
        print()
        
        print("┌─────────────────────────────────────────────────────────────┐")
        print("│ Hater 😤                                                    │")
        print("├─────────────────────────────────────────────────────────────┤")
        print(f"│ Prediction: {hater_result['prediction']:<20s}                   │")
        print(f"│ Probability: {hater_result['probability']:3d}%                                      │")
        print(f"│ Bet: ${hater_result['bet_amount']:4d} / $1000                                   │")
        print(f"│ Reason: {hater_result['reasoning'][:45]:45s}│")
        print("└─────────────────────────────────────────────────────────────┘")
        
        # Calculate simple crowd prediction
        avg_probability = (expert_result['probability'] + hater_result['probability']) / 2
        total_bet = expert_result['bet_amount'] + hater_result['bet_amount']
        
        print("\n" + "="*70)
        print("🎲 Crowd Prediction (Simple Average)")
        print("="*70)
        print(f"Average Probability: {avg_probability:.1f}%")
        print(f"Total Bet: ${total_bet} / $2000")
        print(f"Market Confidence: {'🔥 High' if avg_probability > 60 else '❄️ Low' if avg_probability < 40 else '😐 Medium'}")
        print("="*70)
        
        print("\n💡 Tip: You can add more Agents (optimistic investor, risk manager, etc.) to improve prediction accuracy!")
    
    print("\n✅ Demo Complete!")


if __name__ == "__main__":
    run_demo()

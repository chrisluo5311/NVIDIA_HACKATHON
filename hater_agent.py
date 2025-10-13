"""
Hater AI Agent
WITH REAL SEARCH CAPABILITIES - searches social media and online comments to find real hater opinions
"""

from openai import OpenAI
import json
import random
from search_tools import SocialMediaTool, WebSearchTool
from typing import Dict, List

class HaterAgent:
    def __init__(self):
        # Direct API Key usage
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="YOUR_NVIDIA_API_KEY"
        )
        
        # Initialize real search tools
        self.social_media = SocialMediaTool()
        self.web_search = WebSearchTool()
        
        self.role_description = """You are a typical internet hater/troll with access to REAL online comments.
Your characteristics:
- Always skeptical and negative
- Love to trash talk, throw cold water, and be pessimistic about everything
- Reference ACTUAL hater comments found online
- Don't think deeply, speak based on feelings and emotions
- Like to use exaggerated tone and internet slang
- Often say things like "just another money grab", "destined to fail", "who would buy that"
- Especially love to trash big companies and new products
- You give probability estimates (usually pessimistic) and bet amounts (usually don't dare to bet much)
- Your speaking style is very casual and emotional
- You base your hate on REAL negative comments found online"""

        # Common hater phrases (backup if no real data found)
        self.hater_phrases = [
            "just another money grab",
            "total fail incoming",
            "nobody's gonna buy this",
            "market is already saturated",
            "who even wants this",
            "obviously just hype",
            "literally nobody needs this",
            "last gen flopped hard",
            "price gonna be insane",
            "zero market for this",
            "watch it bomb",
            "lmao here we go again"
        ]

    def gather_hater_data(self, question: str) -> Dict:
        """
        Gather REAL negative comments and opinions from social media
        
        Args:
            question: Question to research
            
        Returns:
            Dictionary containing all gathered hater comments
        """
        print("📡 Searching for Real Hater Comments...")
        print("-" * 60)
        
        hater_data = {
            'reddit_comments': [],
            'twitter_comments': [],
            'general_opinions': []
        }
        
        # Extract topic from question
        topic = self._extract_topic(question)
        
        # 1. Search Reddit for discussions
        print(f"🔍 Searching Reddit for negative opinions on: {topic}...")
        reddit_results = self.social_media.search_reddit_sentiment(topic)
        hater_data['reddit_comments'] = reddit_results
        
        for i, comment in enumerate(reddit_results[:2], 1):
            print(f"   {i}. {comment['title'][:60]}...")
        
        # 2. Search Twitter/X for comments
        print(f"\n🐦 Searching Twitter/X for comments on: {topic}...")
        twitter_results = self.social_media.search_twitter_sentiment(topic)
        hater_data['twitter_comments'] = twitter_results
        
        for i, tweet in enumerate(twitter_results[:2], 1):
            print(f"   {i}. {tweet['title'][:60]}...")
        
        # 3. Search for general negative opinions
        print(f"\n💬 Searching for general negative opinions...")
        negative_query = f"{topic} criticism problems issues complaints"
        general_results = self.web_search.search(negative_query, max_results=5)
        hater_data['general_opinions'] = general_results
        
        for i, opinion in enumerate(general_results[:2], 1):
            print(f"   {i}. {opinion['title'][:60]}...")
        
        print("-" * 60)
        print("✅ Hater data gathering complete!\n")
        
        return hater_data
    
    def _extract_topic(self, question: str) -> str:
        """Extract main topic from question"""
        return question
    
    def analyze(self, question: str):
        """
        Analyze from a hater's perspective using REAL negative comments found online
        
        Args:
            question: Question to analyze (e.g., "Will Apple release Vision Pro 2?")
        
        Returns:
            dict: Contains hater comments, probability, bet amount, etc.
        """
        print(f"\n{'='*60}")
        print("😤 Hater AI Agent Starting to Comment...")
        print("   WITH REAL SOCIAL MEDIA DATA")
        print(f"{'='*60}\n")
        
        # STEP 1: Gather real hater comments from the internet
        hater_data = self.gather_hater_data(question)
        
        # STEP 2: Format hater data for AI analysis
        hater_summary = self._format_hater_data(hater_data)
        
        # STEP 3: Randomly select some backup phrases
        random_phrases = random.sample(self.hater_phrases, 3)
        
        # STEP 4: Build prompt with REAL HATER DATA
        prompt = f"""There's a question being discussed online:

Question: {question}

REAL NEGATIVE COMMENTS AND OPINIONS FOUND ONLINE:
{hater_summary}

As a veteran internet hater, you need to:

1. **Start trashing**: Use REAL negative comments found above as inspiration
2. **Find negative examples**: Reference actual criticisms and complaints from the data
3. **Reasons to hate**: Use the actual negative points found in online discussions
4. **Follow other haters**: Quote or paraphrase the real hater comments found
5. **Emotional venting**: Channel the negativity from real online comments

You can also use these classic hater phrases if needed:
- {random_phrases[0]}
- {random_phrases[1]}
- {random_phrases[2]}

Express yourself in a very casual, internet-style way based on REAL comments!

Finally, provide:
- **Your Prediction**: Will it happen or not (haters usually say no)
- **Probability Estimate**: 0-100% (haters usually give very low probability)
- **How much dare to bet**: Assuming $1000 available, how much would you invest (haters usually don't dare to bet much)
- **One-line summary**: Summarize in the most negative way based on real comments

Summarize in JSON format at the end:
{{
  "prediction": "Will happen" or "Won't happen",
  "probability": number (0-100),
  "bet_amount": number (0-1000),
  "reasoning": "brief hater summary based on real comments"
}}"""

        # Call NVIDIA API
        try:
            completion = self.client.chat.completions.create(
                model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
                messages=[
                    {"role": "system", "content": self.role_description},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,  # Higher temperature for more random, hater-like responses
                top_p=0.95,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
                stream=True
            )
            
            # Collect streaming response
            full_response = ""
            print("💬 Hater's Comments (Based on Real Online Hate):\n")
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            print("\n")
            
            # Extract JSON result
            result = self._extract_result(full_response)
            result['hater_data'] = hater_data  # Include raw hater data
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def _format_hater_data(self, hater_data: Dict) -> str:
        """Format hater data into readable text for AI"""
        formatted = []
        
        # Reddit comments
        if hater_data['reddit_comments']:
            formatted.append("🔴 REDDIT DISCUSSIONS:")
            for i, comment in enumerate(hater_data['reddit_comments'][:5], 1):
                formatted.append(f"{i}. {comment['title']}")
                formatted.append(f"   {comment['snippet'][:200]}")
                formatted.append(f"   URL: {comment['url']}\n")
        
        # Twitter comments
        if hater_data['twitter_comments']:
            formatted.append("\n🐦 TWITTER/X COMMENTS:")
            for i, tweet in enumerate(hater_data['twitter_comments'][:5], 1):
                formatted.append(f"{i}. {tweet['title']}")
                formatted.append(f"   {tweet['snippet'][:200]}\n")
        
        # General negative opinions
        if hater_data['general_opinions']:
            formatted.append("\n💢 GENERAL NEGATIVE OPINIONS:")
            for i, opinion in enumerate(hater_data['general_opinions'][:5], 1):
                formatted.append(f"{i}. {opinion['title']}")
                formatted.append(f"   {opinion['snippet'][:200]}\n")
        
        return "\n".join(formatted) if formatted else "No specific hater comments found (use general hate mode)."
    
    def _extract_result(self, response: str):
        """Extract structured result from response"""
        try:
            # Try to find JSON section
            start_idx = response.rfind("{")
            end_idx = response.rfind("}") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return {
                    "agent_type": "Hater (Real Comments)",
                    "full_analysis": response,
                    "prediction": result.get("prediction", "Won't happen"),
                    "probability": result.get("probability", 20),
                    "bet_amount": result.get("bet_amount", 100),
                    "reasoning": result.get("reasoning", "")
                }
        except:
            pass
        
        # If unable to extract JSON, return raw response
        return {
            "agent_type": "Hater (Real Comments)",
            "full_analysis": response,
            "prediction": "Won't happen",
            "probability": 20,
            "bet_amount": 100,
            "reasoning": "Please see full comments"
        }


def main():
    """Test Hater Agent with real social media search"""
    # Create Agent
    agent = HaterAgent()
    
    # Test question
    question = input("Enter a question for the hater to comment on (e.g., Will Apple release Vision Pro 2?): ")
    if not question.strip():
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

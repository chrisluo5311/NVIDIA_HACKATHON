"""
Technical Expert AI Agent
WITH REAL SEARCH CAPABILITIES - searches internet, financial data, and performs professional analysis
"""

from openai import OpenAI
import json
from typing import Dict, List
from search_tools import WebSearchTool, FinancialDataTool

class TechnicalExpertAgent:
    def __init__(self):
        # Direct API Key usage
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="YOUR_NVIDIA_API_KEY"
        )
        
        # Initialize real search tools
        self.web_search = WebSearchTool()
        self.finance_tool = FinancialDataTool()
        
        self.role_description = """You are a senior technical analyst and financial advisor with access to REAL-TIME data.
Your characteristics:
- Deep technical analysis background
- Carefully research market trends, financial data, and industry reports
- Make rational judgments based on ACTUAL data and logic
- Consider technical feasibility, market demand, and competitive analysis
- Your predictions are always well-supported by REAL research
- You provide specific probability estimates (0-100%) and bet amounts (assuming a $1000 budget)"""

    def gather_research_data(self, question: str) -> Dict:
        """
        Gather real research data from the internet and financial sources
        
        Args:
            question: Question to research
            
        Returns:
            Dictionary containing all gathered research data
        """
        print("📡 Gathering Real-Time Data...")
        print("-" * 60)
        
        research_data = {
            'web_search_results': [],
            'financial_data': {},
            'company_news': []
        }
        
        # Extract company/topic from question
        keywords = self._extract_keywords(question)
        
        # 1. Web search for general information
        print(f"🔍 Searching web for: {keywords}...")
        web_results = self.web_search.search(keywords, max_results=5)
        research_data['web_search_results'] = web_results
        
        for i, result in enumerate(web_results[:3], 1):
            print(f"   {i}. {result['title'][:60]}...")
        
        # 2. Try to get financial data if it's about a company
        company_symbols = self._extract_stock_symbol(question)
        if company_symbols:
            print(f"\n💰 Fetching financial data for: {company_symbols}...")
            for symbol in company_symbols:
                stock_info = self.finance_tool.get_stock_info(symbol)
                research_data['financial_data'][symbol] = stock_info
                if 'current_price' in stock_info:
                    print(f"   {symbol}: ${stock_info.get('current_price', 'N/A')}")
        
        # 3. Search for company news
        if company_symbols:
            print(f"\n📰 Searching for company news...")
            news_results = self.finance_tool.search_company_news(keywords)
            research_data['company_news'] = news_results
            for i, news in enumerate(news_results[:2], 1):
                print(f"   {i}. {news['title'][:60]}...")
        
        print("-" * 60)
        print("✅ Data gathering complete!\n")
        
        return research_data

    def _extract_keywords(self, question: str) -> str:
        """Extract main keywords from question"""
        # Simple keyword extraction
        return question
    
    def _extract_stock_symbol(self, question: str) -> List[str]:
        """Try to extract stock symbols from question"""
        # Common company to symbol mapping
        company_map = {
            'apple': 'AAPL',
            'tesla': 'TSLA',
            'microsoft': 'MSFT',
            'google': 'GOOGL',
            'amazon': 'AMZN',
            'meta': 'META',
            'nvidia': 'NVDA',
            'tsmc': 'TSM',
            'amd': 'AMD',
            'intel': 'INTC'
        }
        
        question_lower = question.lower()
        symbols = []
        
        for company, symbol in company_map.items():
            if company in question_lower:
                symbols.append(symbol)
        
        return symbols

    def analyze(self, question: str):
        """
        Perform technical analysis with REAL DATA from the internet
        
        Args:
            question: Question to analyze (e.g., "Will Apple release Vision Pro 2?")
        
        Returns:
            dict: Contains analysis results, probability, bet amount, etc.
        """
        print(f"\n{'='*60}")
        print("🔬 Technical Expert AI Agent Starting Analysis...")
        print("   WITH REAL-TIME DATA SEARCH")
        print(f"{'='*60}\n")
        
        # STEP 1: Gather real research data
        research_data = self.gather_research_data(question)
        
        # STEP 2: Format research data for AI analysis
        research_summary = self._format_research_data(research_data)
        
        # STEP 3: Build prompt with REAL DATA
        prompt = f"""As a technical expert with access to real-time data, analyze the following question:

Question: {question}

REAL RESEARCH DATA GATHERED:
{research_summary}

Based on this ACTUAL data, please conduct an in-depth analysis:

1. **Market Research**: Analyze the real market trends from the data above
2. **Technical Assessment**: Evaluate technical feasibility based on current information
3. **Financial Analysis**: Use the actual financial data provided
4. **Competitive Analysis**: Analyze competitors based on real news and data
5. **Risk Assessment**: Identify potential risk factors from actual information

Finally, provide:
- **Prediction Result**: Will happen / Won't happen
- **Confidence Probability**: Number from 0-100%
- **Recommended Bet Amount**: Assuming a $1000 budget, how much would you invest
- **Core Reasoning**: Explain your judgment basis with 2-3 key points based on REAL DATA

Summarize in JSON format at the end:
{{
  "prediction": "Will happen" or "Won't happen",
  "probability": number (0-100),
  "bet_amount": number (0-1000),
  "reasoning": "brief reason based on real data"
}}"""

        # Call NVIDIA API
        try:
            completion = self.client.chat.completions.create(
                model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
                messages=[
                    {"role": "system", "content": self.role_description},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                top_p=0.95,
                max_tokens=4096,
                frequency_penalty=0,
                presence_penalty=0,
                stream=True
            )
            
            # Collect streaming response
            full_response = ""
            print("📊 Technical Expert's Analysis (Based on Real Data):\n")
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            print("\n")
            
            # Extract JSON result
            result = self._extract_result(full_response)
            result['research_data'] = research_data  # Include raw research data
            return result
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def _format_research_data(self, research_data: Dict) -> str:
        """Format research data into readable text for AI"""
        formatted = []
        
        # Web search results
        if research_data['web_search_results']:
            formatted.append("📰 WEB SEARCH RESULTS:")
            for i, result in enumerate(research_data['web_search_results'][:5], 1):
                formatted.append(f"{i}. {result['title']}")
                formatted.append(f"   {result['snippet'][:200]}")
                formatted.append(f"   Source: {result['url']}\n")
        
        # Financial data
        if research_data['financial_data']:
            formatted.append("\n💰 FINANCIAL DATA:")
            for symbol, data in research_data['financial_data'].items():
                if 'error' not in data:
                    formatted.append(f"\n{symbol}:")
                    formatted.append(f"  Current Price: ${data.get('current_price', 'N/A')}")
                    formatted.append(f"  Previous Close: ${data.get('previous_close', 'N/A')}")
                    formatted.append(f"  Day High: ${data.get('day_high', 'N/A')}")
                    formatted.append(f"  Day Low: ${data.get('day_low', 'N/A')}")
        
        # Company news
        if research_data['company_news']:
            formatted.append("\n\n📊 RECENT NEWS:")
            for i, news in enumerate(research_data['company_news'][:3], 1):
                formatted.append(f"{i}. {news['title']}")
                formatted.append(f"   {news['snippet'][:150]}\n")
        
        return "\n".join(formatted) if formatted else "No specific data found."
    
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
                    "agent_type": "Technical Expert (Real Data)",
                    "full_analysis": response,
                    "prediction": result.get("prediction", "Unknown"),
                    "probability": result.get("probability", 50),
                    "bet_amount": result.get("bet_amount", 0),
                    "reasoning": result.get("reasoning", "")
                }
        except:
            pass
        
        # If unable to extract JSON, return raw response
        return {
            "agent_type": "Technical Expert (Real Data)",
            "full_analysis": response,
            "prediction": "Unknown",
            "probability": 50,
            "bet_amount": 0,
            "reasoning": "Please see full analysis"
        }


def main():
    """Test Technical Expert Agent with real search"""
    # Create Agent
    agent = TechnicalExpertAgent()
    
    # Test question
    question = input("Enter a question to analyze (e.g., Will Apple release Vision Pro 2?): ")
    if not question.strip():
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

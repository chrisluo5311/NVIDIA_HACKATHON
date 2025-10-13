"""
Search Tools - Real web search, financial data, and social media scraping capabilities
"""

import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict
import time

class WebSearchTool:
    """Web search using DuckDuckGo (no API key required)"""
    
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search the web for information
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title, snippet, and URL
        """
        try:
            # Use DuckDuckGo HTML search
            url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.post(url, data=params, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.find_all('div', class_='result')[:max_results]:
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem:
                    results.append({
                        'title': title_elem.get_text(strip=True),
                        'url': title_elem.get('href', ''),
                        'snippet': snippet_elem.get_text(strip=True) if snippet_elem else ''
                    })
            
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []


class FinancialDataTool:
    """Financial data retrieval tool"""
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get stock information (using free Yahoo Finance alternative)
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
            
        Returns:
            Dictionary with stock information
        """
        try:
            # Use Yahoo Finance query API
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                meta = result.get('meta', {})
                
                return {
                    'symbol': symbol,
                    'currency': meta.get('currency', 'USD'),
                    'current_price': meta.get('regularMarketPrice', 'N/A'),
                    'previous_close': meta.get('previousClose', 'N/A'),
                    'market_cap': meta.get('marketCap', 'N/A'),
                    'day_high': meta.get('regularMarketDayHigh', 'N/A'),
                    'day_low': meta.get('regularMarketDayLow', 'N/A'),
                }
            else:
                return {'error': 'No data available'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def search_company_news(self, company_name: str) -> List[Dict]:
        """
        Search for company financial news
        
        Args:
            company_name: Company name
            
        Returns:
            List of news articles
        """
        web_search = WebSearchTool()
        query = f"{company_name} stock news financial"
        return web_search.search(query, max_results=5)


class SocialMediaTool:
    """Social media and forum scraping tool"""
    
    def search_reddit_sentiment(self, topic: str) -> List[Dict]:
        """
        Search Reddit for discussions (using web scraping)
        
        Args:
            topic: Topic to search for
            
        Returns:
            List of Reddit discussions
        """
        try:
            # Search Reddit using Google
            query = f"site:reddit.com {topic}"
            web_search = WebSearchTool()
            results = web_search.search(query, max_results=5)
            
            # Format results
            reddit_posts = []
            for result in results:
                reddit_posts.append({
                    'title': result['title'],
                    'url': result['url'],
                    'snippet': result['snippet'],
                    'platform': 'Reddit'
                })
            
            return reddit_posts
        except Exception as e:
            print(f"Reddit search error: {e}")
            return []
    
    def search_twitter_sentiment(self, topic: str) -> List[Dict]:
        """
        Search for Twitter/X discussions
        
        Args:
            topic: Topic to search for
            
        Returns:
            List of tweets/discussions
        """
        try:
            # Search Twitter using Google
            query = f"site:twitter.com OR site:x.com {topic}"
            web_search = WebSearchTool()
            results = web_search.search(query, max_results=5)
            
            # Format results
            tweets = []
            for result in results:
                tweets.append({
                    'title': result['title'],
                    'url': result['url'],
                    'snippet': result['snippet'],
                    'platform': 'Twitter/X'
                })
            
            return tweets
        except Exception as e:
            print(f"Twitter search error: {e}")
            return []
    
    def search_general_sentiment(self, topic: str) -> List[Dict]:
        """
        Search for general online sentiment (forums, blogs, etc.)
        
        Args:
            topic: Topic to search for
            
        Returns:
            List of online discussions
        """
        try:
            query = f"{topic} opinion review discussion"
            web_search = WebSearchTool()
            results = web_search.search(query, max_results=5)
            
            return results
        except Exception as e:
            print(f"Sentiment search error: {e}")
            return []


def test_search_tools():
    """Test all search tools"""
    print("="*70)
    print("Testing Search Tools")
    print("="*70)
    
    # Test web search
    print("\n1. Testing Web Search...")
    web_search = WebSearchTool()
    results = web_search.search("Apple Vision Pro 2 release date", max_results=3)
    for i, result in enumerate(results, 1):
        print(f"\n   Result {i}:")
        print(f"   Title: {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Snippet: {result['snippet'][:100]}...")
    
    # Test financial data
    print("\n2. Testing Financial Data...")
    finance = FinancialDataTool()
    stock_info = finance.get_stock_info("AAPL")
    print(f"\n   Apple (AAPL) Stock Info:")
    for key, value in stock_info.items():
        print(f"   {key}: {value}")
    
    # Test social media search
    print("\n3. Testing Social Media Search...")
    social = SocialMediaTool()
    reddit_results = social.search_reddit_sentiment("Apple Vision Pro")
    print(f"\n   Found {len(reddit_results)} Reddit discussions")
    if reddit_results:
        print(f"   Example: {reddit_results[0]['title']}")
    
    print("\n" + "="*70)
    print("Search Tools Test Complete!")
    print("="*70)


if __name__ == "__main__":
    test_search_tools()


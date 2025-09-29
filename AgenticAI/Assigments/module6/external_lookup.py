#!/usr/bin/env python3
"""
External Lookup System
Provides Wikipedia and DuckDuckGo search capabilities for customer service agents
"""

import json
import requests
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import urllib.parse
import re

@dataclass
class SearchResult:
    """Represents a search result from external sources"""
    title: str
    content: str
    url: str
    source: str
    relevance_score: float = 0.0
    summary: str = ""
    timestamp: str = ""

class ExternalLookup:
    """
    External information lookup system for customer service
    
    Provides safe, relevant information from:
    - Wikipedia for general knowledge
    - DuckDuckGo for broader web search
    """
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        """
        Initialize external lookup system
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.logger = logging.getLogger(__name__)
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Set user agent for API requests
        self.session.headers.update({
            'User-Agent': 'CustomerServiceAgent/1.0 (Educational/Research Purpose)'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
        # Content filtering for customer service relevance
        self.relevant_topics = {
            'customer_service', 'business', 'technology', 'product', 'service',
            'policy', 'procedure', 'support', 'help', 'assistance', 'solution',
            'troubleshooting', 'repair', 'warranty', 'guarantee', 'quality',
            'shipping', 'delivery', 'payment', 'billing', 'refund', 'exchange'
        }
    
    def search_wikipedia(self, query: str, max_results: int = 3) -> List[SearchResult]:
        """
        Search Wikipedia for relevant information
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            self._rate_limit()
            
            # Clean and prepare query
            clean_query = self._clean_query(query)
            if not clean_query:
                return []
            
            # Search Wikipedia
            search_results = self._wikipedia_search(clean_query, max_results)
            
            # Get detailed content for each result
            detailed_results = []
            for title in search_results[:max_results]:
                try:
                    content = self._get_wikipedia_content(title)
                    if content:
                        result = SearchResult(
                            title=title,
                            content=content['content'],
                            url=content['url'],
                            source='Wikipedia',
                            summary=content['summary'],
                            timestamp=datetime.now().isoformat()
                        )
                        result.relevance_score = self._calculate_relevance(result, query)
                        detailed_results.append(result)
                        
                except Exception as e:
                    self.logger.warning(f"Error getting Wikipedia content for '{title}': {str(e)}")
                    continue
            
            # Sort by relevance
            detailed_results.sort(key=lambda x: x.relevance_score, reverse=True)
            return detailed_results
            
        except Exception as e:
            self.logger.error(f"Wikipedia search error: {str(e)}")
            return []
    
    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Search DuckDuckGo for relevant information
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            self._rate_limit()
            
            # Clean and prepare query
            clean_query = self._clean_query(query)
            if not clean_query:
                return []
            
            # Use DuckDuckGo Instant Answer API
            results = self._duckduckgo_search(clean_query, max_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {str(e)}")
            return []
    
    def search_combined(self, query: str, include_wikipedia: bool = True, 
                       include_duckduckgo: bool = True, max_total_results: int = 5) -> Dict[str, Any]:
        """
        Search both Wikipedia and DuckDuckGo and combine results
        
        Args:
            query: Search query
            include_wikipedia: Whether to include Wikipedia results
            include_duckduckgo: Whether to include DuckDuckGo results
            max_total_results: Maximum total results to return
            
        Returns:
            Dictionary with combined results and metadata
        """
        all_results = []
        search_metadata = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'sources_searched': [],
            'total_results': 0
        }
        
        # Wikipedia search
        if include_wikipedia:
            try:
                wiki_results = self.search_wikipedia(query, max_results=3)
                all_results.extend(wiki_results)
                search_metadata['sources_searched'].append('Wikipedia')
                search_metadata['wikipedia_results'] = len(wiki_results)
            except Exception as e:
                self.logger.error(f"Wikipedia search failed: {str(e)}")
                search_metadata['wikipedia_error'] = str(e)
        
        # DuckDuckGo search
        if include_duckduckgo:
            try:
                ddg_results = self.search_duckduckgo(query, max_results=3)
                all_results.extend(ddg_results)
                search_metadata['sources_searched'].append('DuckDuckGo')
                search_metadata['duckduckgo_results'] = len(ddg_results)
            except Exception as e:
                self.logger.error(f"DuckDuckGo search failed: {str(e)}")
                search_metadata['duckduckgo_error'] = str(e)
        
        # Remove duplicates and sort by relevance
        unique_results = self._deduplicate_results(all_results)
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Limit to max results
        final_results = unique_results[:max_total_results]
        
        search_metadata['total_results'] = len(final_results)
        search_metadata['deduplication_removed'] = len(all_results) - len(unique_results)
        
        return {
            'results': final_results,
            'metadata': search_metadata,
            'query_suggestions': self._generate_query_suggestions(query, final_results)
        }
    
    def get_customer_service_context(self, query: str, customer_issue: str = None) -> Dict[str, Any]:
        """
        Get external information specifically for customer service context
        
        Args:
            query: Search query
            customer_issue: Type of customer issue (optional)
            
        Returns:
            Curated information relevant to customer service
        """
        # Enhance query with customer service context
        enhanced_query = self._enhance_query_for_customer_service(query, customer_issue)
        
        # Search for information
        search_results = self.search_combined(enhanced_query, max_total_results=3)
        
        # Filter and curate for customer service relevance
        curated_results = self._curate_for_customer_service(search_results['results'], customer_issue)
        
        return {
            'original_query': query,
            'enhanced_query': enhanced_query,
            'customer_issue': customer_issue,
            'curated_results': curated_results,
            'search_metadata': search_results['metadata'],
            'agent_guidance': self._generate_agent_guidance(curated_results, customer_issue),
            'confidence_score': self._calculate_overall_confidence(curated_results)
        }
    
    def validate_information(self, information: str, source: str = None) -> Dict[str, Any]:
        """
        Validate information for customer service use
        
        Args:
            information: Information to validate
            source: Source of information
            
        Returns:
            Validation results and recommendations
        """
        validation = {
            'is_reliable': True,
            'confidence': 0.8,
            'concerns': [],
            'recommendations': []
        }
        
        # Check source reliability
        if source:
            if source.lower() == 'wikipedia':
                validation['confidence'] = 0.9
                validation['recommendations'].append("Wikipedia information is generally reliable but verify specific claims")
            elif 'duckduckgo' in source.lower():
                validation['confidence'] = 0.7
                validation['recommendations'].append("Cross-reference with official sources when possible")
        
        # Check content flags
        concerning_patterns = [
            r'\b(opinion|allegedly|claims|rumors?)\b',
            r'\b(unverified|unconfirmed|speculation)\b',
            r'\b(breaking|developing|latest)\b'
        ]
        
        for pattern in concerning_patterns:
            if re.search(pattern, information, re.IGNORECASE):
                validation['concerns'].append(f"Contains potentially unreliable language: {pattern}")
                validation['confidence'] *= 0.8
        
        # Check recency indicators
        if re.search(r'\b(today|yesterday|recent|latest|breaking)\b', information, re.IGNORECASE):
            validation['concerns'].append("Information may be time-sensitive")
            validation['recommendations'].append("Verify current status before using")
        
        validation['confidence'] = min(validation['confidence'], 1.0)
        validation['is_reliable'] = validation['confidence'] > 0.5
        
        return validation
    
    # Private helper methods
    
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _clean_query(self, query: str) -> str:
        """Clean and sanitize search query"""
        if not query or len(query.strip()) < 2:
            return ""
        
        # Remove special characters and extra whitespace
        cleaned = re.sub(r'[^\w\s\-]', ' ', query)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Limit length
        if len(cleaned) > 100:
            cleaned = cleaned[:100]
        
        return cleaned
    
    def _wikipedia_search(self, query: str, max_results: int) -> List[str]:
        """Search Wikipedia and return page titles"""
        search_url = "https://en.wikipedia.org/api/rest_v1/page/search"
        
        params = {
            'q': query,
            'limit': max_results
        }
        
        response = self._make_request(search_url, params)
        if not response:
            return []
        
        pages = response.get('pages', [])
        return [page['title'] for page in pages if 'title' in page]
    
    def _get_wikipedia_content(self, title: str) -> Optional[Dict[str, str]]:
        """Get Wikipedia page content"""
        # Get page extract
        extract_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
        
        response = self._make_request(extract_url)
        if not response:
            return None
        
        content = response.get('extract', '')
        if not content:
            return None
        
        return {
            'content': content,
            'summary': content[:200] + "..." if len(content) > 200 else content,
            'url': response.get('content_urls', {}).get('desktop', {}).get('page', '')
        }
    
    def _duckduckgo_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Search DuckDuckGo using Instant Answer API"""
        # Note: DuckDuckGo's Instant Answer API has limited functionality
        # In a production environment, you might use their paid API or other search APIs
        
        search_url = "https://api.duckduckgo.com/"
        
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        response = self._make_request(search_url, params)
        if not response:
            return []
        
        results = []
        
        # Check for instant answer
        if response.get('AbstractText'):
            results.append(SearchResult(
                title=response.get('Heading', 'DuckDuckGo Result'),
                content=response.get('AbstractText', ''),
                url=response.get('AbstractURL', ''),
                source='DuckDuckGo',
                summary=response.get('AbstractText', '')[:200] + "..." if len(response.get('AbstractText', '')) > 200 else response.get('AbstractText', ''),
                timestamp=datetime.now().isoformat()
            ))
        
        # Check for related topics
        related_topics = response.get('RelatedTopics', [])
        for topic in related_topics[:max_results-1]:
            if isinstance(topic, dict) and 'Text' in topic:
                results.append(SearchResult(
                    title=topic.get('Result', '').split(' - ')[0] if ' - ' in topic.get('Result', '') else 'Related Topic',
                    content=topic.get('Text', ''),
                    url=topic.get('FirstURL', ''),
                    source='DuckDuckGo',
                    summary=topic.get('Text', '')[:200] + "..." if len(topic.get('Text', '')) > 200 else topic.get('Text', ''),
                    timestamp=datetime.now().isoformat()
                ))
        
        return results
    
    def _make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make HTTP request with retries and error handling"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"All request attempts failed for URL: {url}")
                    return None
            
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error for URL {url}: {str(e)}")
                return None
    
    def _calculate_relevance(self, result: SearchResult, query: str) -> float:
        """Calculate relevance score for search result"""
        score = 0.0
        query_lower = query.lower()
        content_lower = result.content.lower()
        title_lower = result.title.lower()
        
        # Title matching (higher weight)
        query_words = query_lower.split()
        title_words = title_lower.split()
        
        title_matches = sum(1 for word in query_words if word in title_words)
        if title_words:
            score += (title_matches / len(title_words)) * 0.4
        
        # Content matching
        content_matches = sum(1 for word in query_words if word in content_lower)
        if query_words:
            score += (content_matches / len(query_words)) * 0.3
        
        # Customer service relevance
        cs_relevance = sum(1 for topic in self.relevant_topics if topic in content_lower)
        score += min(cs_relevance * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on content similarity"""
        if not results:
            return []
        
        unique_results = []
        seen_titles = set()
        
        for result in results:
            # Simple deduplication based on title
            title_key = result.title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_results.append(result)
        
        return unique_results
    
    def _generate_query_suggestions(self, original_query: str, results: List[SearchResult]) -> List[str]:
        """Generate alternative query suggestions"""
        suggestions = []
        
        # Add customer service context
        suggestions.append(f"{original_query} customer service")
        suggestions.append(f"{original_query} support policy")
        suggestions.append(f"how to {original_query}")
        
        # Extract relevant terms from results
        if results:
            common_terms = set()
            for result in results:
                words = result.title.lower().split()
                common_terms.update([word for word in words if len(word) > 4])
            
            # Create suggestions with common terms
            for term in list(common_terms)[:2]:
                suggestions.append(f"{original_query} {term}")
        
        return suggestions[:5]
    
    def _enhance_query_for_customer_service(self, query: str, issue_type: str = None) -> str:
        """Enhance query with customer service context"""
        enhanced = query
        
        # Add issue-specific context
        if issue_type:
            issue_contexts = {
                'refund': 'refund policy procedure',
                'shipping': 'shipping delivery policy',
                'billing': 'billing payment policy',
                'technical': 'technical support troubleshooting',
                'account': 'account management policy'
            }
            
            context = issue_contexts.get(issue_type.lower(), 'customer service policy')
            enhanced = f"{query} {context}"
        
        return enhanced
    
    def _curate_for_customer_service(self, results: List[SearchResult], issue_type: str = None) -> List[Dict[str, Any]]:
        """Curate results for customer service relevance"""
        curated = []
        
        for result in results:
            # Calculate customer service relevance
            cs_score = self._calculate_cs_relevance(result.content, issue_type)
            
            if cs_score > 0.3:  # Minimum relevance threshold
                curated_result = {
                    'title': result.title,
                    'summary': result.summary,
                    'key_points': self._extract_key_points(result.content),
                    'customer_service_relevance': cs_score,
                    'source': result.source,
                    'url': result.url,
                    'usage_guidance': self._generate_usage_guidance(result, issue_type),
                    'reliability_check': self.validate_information(result.content, result.source)
                }
                curated.append(curated_result)
        
        return curated
    
    def _calculate_cs_relevance(self, content: str, issue_type: str = None) -> float:
        """Calculate customer service relevance score"""
        content_lower = content.lower()
        score = 0.0
        
        # General customer service terms
        cs_terms = ['customer', 'service', 'support', 'help', 'policy', 'procedure', 'guideline']
        for term in cs_terms:
            if term in content_lower:
                score += 0.1
        
        # Issue-specific terms
        if issue_type:
            issue_terms = {
                'refund': ['refund', 'return', 'money back', 'reimbursement'],
                'shipping': ['shipping', 'delivery', 'package', 'courier'],
                'billing': ['billing', 'payment', 'charge', 'invoice'],
                'technical': ['technical', 'troubleshoot', 'bug', 'error', 'fix'],
                'account': ['account', 'login', 'password', 'access']
            }
            
            relevant_terms = issue_terms.get(issue_type.lower(), [])
            for term in relevant_terms:
                if term in content_lower:
                    score += 0.15
        
        return min(score, 1.0)
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content"""
        sentences = content.split('. ')
        
        # Look for sentences with important indicators
        key_indicators = ['important', 'note', 'must', 'should', 'required', 'policy', 'procedure']
        key_points = []
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in key_indicators):
                key_points.append(sentence.strip() + '.')
        
        # If no key indicators found, return first few sentences
        if not key_points:
            key_points = [sentence.strip() + '.' for sentence in sentences[:3] if sentence.strip()]
        
        return key_points[:5]
    
    def _generate_usage_guidance(self, result: SearchResult, issue_type: str = None) -> Dict[str, Any]:
        """Generate guidance for using the information"""
        return {
            'confidence': 'high' if result.source == 'Wikipedia' else 'medium',
            'verification_needed': result.source != 'Wikipedia',
            'appropriate_for_customer': True,
            'usage_notes': [
                'Verify current applicability',
                'Cross-reference with company policy',
                'Use as general guidance only'
            ]
        }
    
    def _generate_agent_guidance(self, curated_results: List[Dict], issue_type: str = None) -> Dict[str, Any]:
        """Generate guidance for customer service agents"""
        if not curated_results:
            return {
                'recommendation': 'No relevant external information found',
                'confidence': 'low',
                'next_steps': ['Use internal knowledge base', 'Escalate if needed']
            }
        
        high_relevance_count = sum(1 for r in curated_results if r['customer_service_relevance'] > 0.7)
        
        return {
            'recommendation': f"Found {len(curated_results)} relevant external sources",
            'confidence': 'high' if high_relevance_count > 0 else 'medium',
            'next_steps': [
                'Review key points from external sources',
                'Cross-reference with internal policies',
                'Use information to enhance customer response'
            ],
            'cautions': [
                'Verify information is current',
                'Ensure compliance with company policies',
                'Do not promise specific outcomes without authorization'
            ]
        }
    
    def _calculate_overall_confidence(self, curated_results: List[Dict]) -> float:
        """Calculate overall confidence in search results"""
        if not curated_results:
            return 0.0
        
        # Average relevance scores
        avg_relevance = sum(r['customer_service_relevance'] for r in curated_results) / len(curated_results)
        
        # Factor in source reliability
        reliable_sources = sum(1 for r in curated_results if r['reliability_check']['confidence'] > 0.7)
        source_factor = reliable_sources / len(curated_results)
        
        return (avg_relevance + source_factor) / 2


def main():
    """Demonstration of external lookup system"""
    print("External Lookup System Demo")
    print("=" * 50)
    
    # Initialize lookup system
    lookup = ExternalLookup()
    
    # Test queries
    test_queries = [
        ("refund policy", "refund"),
        ("shipping delays", "shipping"), 
        ("password security", "account"),
        ("customer service best practices", "general")
    ]
    
    print("Testing external lookups:")
    print("-" * 30)
    
    for query, issue_type in test_queries:
        print(f"\nQuery: '{query}' (Issue: {issue_type})")
        
        # Get customer service context
        context = lookup.get_customer_service_context(query, issue_type)
        
        print(f"Enhanced Query: {context['enhanced_query']}")
        print(f"Overall Confidence: {context['confidence_score']:.2f}")
        print(f"Results Found: {len(context['curated_results'])}")
        
        for i, result in enumerate(context['curated_results'], 1):
            print(f"  {i}. {result['title']} ({result['source']})")
            print(f"     Relevance: {result['customer_service_relevance']:.2f}")
            print(f"     Key Points: {len(result['key_points'])}")
        
        print(f"Agent Guidance: {context['agent_guidance']['recommendation']}")
        print("-" * 30)
    
    print("\nExternal lookup system demonstration complete!")


if __name__ == "__main__":
    main()
import asyncio
import aiohttp
import ssl
import random
import time
import sys
from lxml import html as lxml_html

# Try to import the robust library first.
try:
    from duckduckgo_search import DDGS as LibDDGS
    HAS_LIB = True
except ImportError:
    HAS_LIB = False

class DDGS:
    """
    Talos-O Visual Cortex Mark V (The Sovereign Eye).
    Philosophy: Neo Techne / First Principles.
    Cryptographically secure, asynchronously threaded, and topologically parsed.
    """
    def __init__(self):
        self.lib_backend = LibDDGS() if HAS_LIB else None
        
        # High-entropy User-Agent rotation to bypass adversarial friction
        self.user_agents = [
            "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
        ]
        
        # STRICT SSL VERIFICATION: MitM attacks are a violation of Sovereign Integrity.
        self.ctx = ssl.create_default_context()
        self.ctx.verify_mode = ssl.CERT_REQUIRED
        self.ctx.check_hostname = True

    def text(self, keywords, max_results=5):
        """
        Synchronous wrapper for the Motor Cortex to maintain API compatibility,
        while utilizing the high-efficiency asynchronous engine underneath.
        """
        if self.lib_backend:
            try:
                # The official pip library returns a generator
                results = []
                for i, r in enumerate(self.lib_backend.text(keywords, max_results=max_results)):
                    results.append(r)
                return results
            except Exception as e:
                print(f"[VISION] Primary Cortex Failed ({e}). Engaging Fallback...")
        
        # Engage Neo-Techne Fallback
        return asyncio.run(self._async_text(keywords, max_results))

    async def _async_text(self, keywords, max_results):
        """Asynchronous HTTP retrieval with Exponential Backoff."""
        url = "https://html.duckduckgo.com/html/"
        data = {"q": keywords, "b": ""}
        
        headers = {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        # The Metabolization of Error: Exponential Backoff
        max_retries = 3
        base_delay = 1.0

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=self.ctx)) as session:
            for attempt in range(max_retries):
                try:
                    async with session.post(url, data=data, headers=headers, timeout=10.0) as response:
                        if response.status == 200:
                            html_content = await response.text()
                            return self._parse_html(html_content, max_results)
                        elif response.status == 429: # Rate Limited
                            raise Exception("Rate Limited (429)")
                        else:
                            raise Exception(f"HTTP {response.status}")
                            
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"\033[91m[VISION] Fatal Friction: {e}\033[0m")
                        return []
                    sleep_time = base_delay * (2 ** attempt) + random.uniform(0.1, 0.5)
                    print(f"[VISION] Network Friction ({e}). Backing off for {sleep_time:.2f}s...")
                    await asyncio.sleep(sleep_time)
        return []

    def _parse_html(self, content, limit):
        """
        Topological DOM Analysis (lxml). 
        Eradicates the brittleness of regular expressions.
        """
        results = []
        try:
            tree = lxml_html.fromstring(content)
            # XPath strictly maps the structural hierarchy of the results
            result_nodes = tree.xpath('//div[contains(@class, "result__body")]')
            
            for i, node in enumerate(result_nodes):
                if i >= limit: break
                
                # Safely extract href, title, and snippet using localized XPath
                href = node.xpath('.//a[contains(@class, "result__url")]/@href')
                title = node.xpath('.//h2/a/text()')
                snippet = node.xpath('.//a[contains(@class, "result__snippet")]/text()')
                
                if href and title:
                    clean_href = href[0].strip()
                    if "duckduckgo.com" in clean_href: continue # Filter ad loops
                    
                    results.append({
                        "title": title[0].strip(),
                        "href": clean_href,
                        "body": snippet[0].strip() if snippet else "Knowledge fragment missing."
                    })
        except Exception as e:
            print(f"\033[91m[VISION] Topological DOM Parse Failure (Lysosome Triggered): {e}\033[0m")
            
        return results

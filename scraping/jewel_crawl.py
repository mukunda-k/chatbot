import os
import sys
import psutil
import asyncio
import re
import logging
from typing import Set, List
from urllib.parse import urljoin, urlparse

__location__ = os.path.dirname(os.path.abspath(__file__))
# Save to data folder outside the scrapping folder (one level up from scrapping)
__output__ = os.path.join(os.path.dirname(__location__), "data")
os.makedirs(__output__, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(__output__, 'crawler.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Append parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

class MemoryMonitor:
    def __init__(self):
        self.peak_memory = 0
        self.process = psutil.Process(os.getpid())
    
    def log_memory(self, prefix: str = ""):
        current_mem = self.process.memory_info().rss
        if current_mem > self.peak_memory:
            self.peak_memory = current_mem
        current_mb = current_mem // (1024 * 1024)
        peak_mb = self.peak_memory // (1024 * 1024)
        logger.debug(f"{prefix} Memory: {current_mb} MB (Peak: {peak_mb} MB)")
        return current_mb, peak_mb

class DeepCrawler:
    def __init__(self, base_url: str, max_depth: int = 5, max_concurrent: int = 3):
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.max_depth = max_depth
        self.max_concurrent = max_concurrent
        self.visited_urls: Set[str] = set()
        self.crawled_count = 0
        self.failed_count = 0
        self.memory_monitor = MemoryMonitor()
        
        # Browser configuration optimized for memory efficiency
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            extra_args=[
                "--disable-gpu",
                "--disable-dev-shm-usage", 
                "--no-sandbox",
                "--disable-extensions",
                "--disable-plugins",
                "--disable-images",  # Skip images to save memory
                "--disable-javascript",  # Skip JS if not needed for content
                "--memory-pressure-off"
            ],
        )
        
        self.crawl_config = CrawlerRunConfig(
            keep_attrs=False,
            cache_mode=CacheMode.BYPASS,
            process_iframes=False,
            remove_overlay_elements=True
        )

    def url_to_filename(self, url: str) -> str:
        """Convert URL to a valid filename"""
        # Remove protocol
        filename = url.replace('https://', '').replace('http://', '')
        
        # Replace invalid characters with underscores
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Replace multiple underscores with single underscore
        filename = re.sub(r'_+', '_', filename)
        
        # Remove trailing slashes and replace with underscore
        filename = filename.rstrip('/')
        if filename.endswith('_'):
            filename = filename.rstrip('_')
        
        # If filename is empty or too long, create a hash-based name
        if not filename or len(filename) > 200:
            filename = f"url_{hash(url) % 100000}"
        
        # Ensure it ends with .html
        if not filename.endswith('.html'):
            filename += '.html'
            
        return filename

    def is_valid_url(self, url: str) -> bool:
        """Check if URL belongs to the same domain and hasn't been visited"""
        if not url or url in self.visited_urls:
            return False
        
        parsed = urlparse(url)
        return (
            parsed.netloc == self.base_domain and
            parsed.scheme in ['http', 'https'] and
            not any(ext in url.lower() for ext in ['.pdf', '.jpg', '.png', '.gif', '.css', '.js'])
        )

    def extract_links(self, html_content: str, base_url: str) -> Set[str]:
        """Extract valid links from HTML content"""
        # Simple regex to find href attributes
        href_pattern = r'href=["\']([^"\']+)["\']'
        links = set()
        
        for match in re.finditer(href_pattern, html_content, re.IGNORECASE):
            link = match.group(1)
            full_url = urljoin(base_url, link)
            if self.is_valid_url(full_url):
                links.add(full_url)
        
        return links

    async def crawl_batch(self, crawler: AsyncWebCrawler, urls: List[str], depth: int) -> Set[str]:
        """Crawl a batch of URLs and return discovered links"""
        if not urls:
            return set()

        logger.debug(f"Starting batch crawl at depth {depth} with {len(urls)} URLs")
        self.memory_monitor.log_memory(f"[Depth {depth}] Before batch: ")
        
        tasks = []
        for i, url in enumerate(urls):
            session_id = f"depth_{depth}_url_{i}_{hash(url) % 10000}"
            task = crawler.arun(url=url, config=self.crawl_config, session_id=session_id)
            tasks.append((url, session_id, task))

        # Execute tasks with timeout
        results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
        
        discovered_links = set()
        
        for (url, session_id, _), result in zip(tasks, results):
            self.visited_urls.add(url)
            
            if isinstance(result, Exception):
                logger.error(f"Error crawling {url}: {result}")
                self.failed_count += 1
            elif result.success and result.cleaned_html:
                self.crawled_count += 1
                logger.info(f"Successfully crawled: {url}")
                
                # Create filename from URL
                filename = self.url_to_filename(url)
                filepath = os.path.join(__output__, filename)
                
                # Save content with URL header and HTML content
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"# URL: {url}\n{result.cleaned_html}")
                
                logger.debug(f"Saved content to: {filename}")
                
                # Extract links for next depth level
                if depth < self.max_depth:
                    new_links = self.extract_links(result.cleaned_html, url)
                    discovered_links.update(new_links)
                    if new_links:
                        logger.debug(f"Found {len(new_links)} new links from {url}")
            else:
                logger.warning(f"Failed to crawl: {url}")
                self.failed_count += 1

        self.memory_monitor.log_memory(f"[Depth {depth}] After batch: ")
        return discovered_links

    async def deep_crawl(self):
        """Perform deep crawling starting from base URL"""
        logger.info(f"Starting deep crawl of {self.base_url}")
        logger.info(f"Saving files to: {__output__}")
        logger.info(f"Max depth: {self.max_depth}, Max concurrent: {self.max_concurrent}")
        
        crawler = AsyncWebCrawler(config=self.browser_config)
        await crawler.start()
        
        try:
            current_urls = {self.base_url}
            
            for depth in range(1, self.max_depth + 1):
                if not current_urls:
                    logger.info(f"No more URLs to crawl at depth {depth}")
                    break
                
                logger.info(f"Crawling depth {depth} with {len(current_urls)} URLs")
                
                # Process URLs in batches
                url_list = list(current_urls)
                next_level_urls = set()
                
                for i in range(0, len(url_list), self.max_concurrent):
                    batch = url_list[i:i + self.max_concurrent]
                    discovered = await self.crawl_batch(crawler, batch, depth)
                    next_level_urls.update(discovered)
                
                # Filter out already visited URLs for next depth
                current_urls = {url for url in next_level_urls if url not in self.visited_urls}
                
                logger.info(f"Depth {depth} complete. Found {len(current_urls)} new URLs for next level")
        
        finally:
            logger.info("Closing crawler...")
            await crawler.close()
            
            # Final summary
            logger.info("CRAWLING SUMMARY:")
            logger.info(f"Successfully crawled: {self.crawled_count}")
            logger.info(f"Failed: {self.failed_count}")
            logger.info(f"Total URLs discovered: {len(self.visited_urls)}")
            logger.info(f"Files saved to: {__output__}")
            
            final_mem, peak_mem = self.memory_monitor.log_memory("Final ")
            logger.info(f"Peak memory usage: {peak_mem} MB")

async def main():
    # Configuration
    BASE_URL = "https://www.jewelchangiairport.com"  # Change this to your target website
    MAX_DEPTH = 5
    MAX_CONCURRENT = 3  # Reduced for better memory management
    
    # Create and run deep crawler
    crawler = DeepCrawler(
        base_url=BASE_URL,
        max_depth=MAX_DEPTH,
        max_concurrent=MAX_CONCURRENT
    )
    
    await crawler.deep_crawl()

if __name__ == "__main__":
    logger.info("Deep Web Crawler Starting...")
    asyncio.run(main())
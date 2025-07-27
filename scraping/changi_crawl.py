import os
import sys
import psutil
import asyncio
import requests
import re
import logging
from xml.etree import ElementTree
from urllib.parse import urlparse

__location__ = os.path.dirname(os.path.abspath(__file__))
# Save to data folder outside the scrapping folder (one level up from scrapping)
__output__ = os.path.join(os.path.dirname(__location__), "data")
os.makedirs(__output__, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(__output__, 'sitemap_crawler.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Append parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

def url_to_filename(url: str) -> str:
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

async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    logger.info("Starting parallel crawling with browser reuse and memory monitoring")

    # We'll keep track of peak memory usage across all tasks
    peak_memory = 0
    process = psutil.Process(os.getpid())

    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss  # in bytes
        if current_mem > peak_memory:
            peak_memory = current_mem
        current_mb = current_mem // (1024 * 1024)
        peak_mb = peak_memory // (1024 * 1024)
        logger.debug(f"{prefix} Current Memory: {current_mb} MB, Peak: {peak_mb} MB")

    # Browser config with anti-detection measures
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=[
            "--disable-gpu", 
            "--disable-dev-shm-usage", 
            "--no-sandbox",
            "--disable-extensions",
            "--disable-plugins",
            "--disable-images",  # Skip images to save memory
            "--memory-pressure-off",
            "--disable-blink-features=AutomationControlled",  # Hide automation
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ],
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    )
    
    crawl_config = CrawlerRunConfig(
        keep_attrs=False,
        cache_mode=CacheMode.BYPASS,
        process_iframes=False,
        remove_overlay_elements=True,
        delay_before_return_html=2.0,  # Wait 2 seconds for page to load
        js_code=[
            "window.scrollTo(0, document.body.scrollHeight);",  # Scroll to simulate human behavior
        ]
    )

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # We'll chunk the URLs in batches of 'max_concurrent'
        success_count = 0
        fail_count = 0
        total_batches = (len(urls) + max_concurrent - 1) // max_concurrent
        
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            batch_num = i // max_concurrent + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} URLs")
            
            tasks = []
            for j, url in enumerate(batch):
                # Unique session_id per concurrent sub-task
                session_id = f"parallel_session_{i + j}"
                task = crawler.arun(url=url, config=crawl_config, session_id=session_id)
                tasks.append((url, session_id, task))

            # Check memory usage prior to launching tasks
            log_memory(prefix=f"Before batch {batch_num}: ")

            # Gather results with delay between requests
            results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
            
            # Add delay between batches to avoid rate limiting
            if i + max_concurrent < len(urls):  # Don't delay after the last batch
                delay = min(2.0, max_concurrent * 0.5)  # Scale delay with batch size
                logger.debug(f"Waiting {delay} seconds before next batch...")
                await asyncio.sleep(delay)

            # Check memory usage after tasks complete
            log_memory(prefix=f"After batch {batch_num}: ")

            # Evaluate results
            for (url, session_id, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"Error crawling {url}: {result}")
                    fail_count += 1
                elif result.success and result.cleaned_html:
                    success_count += 1
                    
                    # Create filename from URL
                    filename = url_to_filename(url)
                    filepath = os.path.join(__output__, filename)
                    
                    # Save content with URL header and HTML content
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"# URL: {url}\n{result.cleaned_html}")
                    
                    logger.info(f"Successfully crawled and saved: {url}")
                    logger.debug(f"Saved content to: {filename}")
                else:
                    logger.warning(f"Failed to crawl: {url}")
                    fail_count += 1

        logger.info("CRAWLING SUMMARY:")
        logger.info(f"Successfully crawled: {success_count}")
        logger.info(f"Failed: {fail_count}")
        logger.info(f"Files saved to: {__output__}")

    finally:
        logger.info("Closing crawler...")
        await crawler.close()
        # Final memory log
        log_memory(prefix="Final: ")
        peak_mb = peak_memory // (1024 * 1024)
        logger.info(f"Peak memory usage: {peak_mb} MB")

def get_sitemap_urls(sitemap_url: str):
    """
    Fetches all URLs from the sitemap.
    
    Args:
        sitemap_url (str): The sitemap URL
        
    Returns:
        List[str]: List of URLs
    """            
    try:
        logger.info(f"Fetching sitemap from: {sitemap_url}")
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        # The namespace is usually defined in the root element
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        logger.info(f"Found {len(urls)} URLs in sitemap")
        return urls
        
    except requests.RequestException as e:
        logger.error(f"Network error fetching sitemap: {e}")
        return []
    except ElementTree.ParseError as e:
        logger.error(f"XML parsing error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching sitemap: {e}")
        return []        

async def main():
    # Configuration
    SITEMAP_URL = "https://www.changiairport.com/sitemap.xml"  # Change this to your target sitemap
    MAX_CONCURRENT = 3  # Reduced to avoid rate limiting (was 10)
    
    logger.info("Sitemap Crawler Starting...")
    logger.info(f"Target sitemap: {SITEMAP_URL}")
    logger.info(f"Max concurrent requests: {MAX_CONCURRENT}")
    
    urls = get_sitemap_urls(SITEMAP_URL)
    if urls:
        logger.info(f"Starting crawl of {len(urls)} URLs")
        await crawl_parallel(urls, max_concurrent=MAX_CONCURRENT)
    else:
        logger.error("No URLs found to crawl")    

if __name__ == "__main__":
    asyncio.run(main())
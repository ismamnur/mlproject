import sys
import os
import crawl4ai
# Add the project root (one level above "src") to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.exception import CustomException
from src.logger import logging

import asyncio
import re
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

async def crawl_urls(urls, output_file="crawled_results.md"):
    """
    Crawl a list of URLs and save the extracted content to a file.

    Args:
        urls (list): List of URLs to crawl.
        output_file (str): Path to save the crawled content.
    """
    # Define the crawler configuration
    crawler_run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    all_content = ""

    async with AsyncWebCrawler() as crawler:
        for url in urls:
            try:
                logging.info(f"Starting crawl for: {url}")
                # Crawl the URL
                result = await crawler.arun(
                    url=url,
                    config=crawler_run_config
                )
                # Get the raw markdown content
                markdown_content = result.markdown_v2.raw_markdown

                # Convert Markdown links to plain text
                markdown_content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", markdown_content)

                # Append the cleaned content
                all_content += markdown_content + "\n\n"
                logging.info(f"Successfully crawled: {url}")
            except Exception as e:
                logging.error(f"Failed to crawl {url}")
                raise CustomException(e, sys)

    # Save the combined content to the output file
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(all_content)
        logging.info(f"All crawled content saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save crawled content to {output_file}")
        raise CustomException(e, sys)


# if __name__ == "__main__":
#     try:
#         # List of URLs to crawl
#         urls_to_crawl = [
#             "https://dermnetnz.org/topics/actinic-keratosis",
#             "https://dermnetnz.org/topics/atopic-dermatitis",
#             "https://dermnetnz.org/topics/dermatitis",
#             "https://dermnetnz.org/topics/lichen-planus",
#             "https://dermnetnz.org/topics/melanoma",
#             "https://dermnetnz.org/topics/psoriasis",
#             "https://dermnetnz.org/topics/rosacea",
#             "https://dermnetnz.org/topics/seborrhoeic-dermatitis",
#             "https://dermnetnz.org/topics/seborrhoeic-keratosis",
#             "https://dermnetnz.org/topics/basal-cell-carcinoma",
#         ]

#         # Run the crawler
#         asyncio.run(crawl_urls(urls_to_crawl))

#     except CustomException as ce:
#         logging.error(f"Application terminated due to an error: {ce}")


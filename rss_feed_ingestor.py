import feedparser
import re
import logging

logger = logging.getLogger(__name__)

def clean_html(raw_html):
    """Remove HTML tags from summary text."""
    clean_text = re.sub('<.*?>', '', raw_html)
    return clean_text.strip()

def get_rss_news(feed_url, max_articles=5):
    """Fetch and display news articles from an RSS feed with highlights."""
    try:
        feed = feedparser.parse(feed_url)
        articles = [] #return clean articles back in a list for further processing
        if 'entries' in feed:
            logger.info(f"Fetched {len(feed.entries)} articles from {feed.feed.title}")
            for entry in feed.entries[:max_articles]:
                summary = clean_html(entry.summary) if 'summary' in entry else 'No summary available'
                full_text = f"Title: {entry.title}\nSummary: {summary}\nLink: {entry.link}"
                articles.append(full_text)

        else:
            logger.warning(f"Failed to fetch news from {feed_url}. Check URL and permissions.")

        return articles

    except Exception as e:
        logger.error(f"Error during RSS feed processing: {e}", exc_info=True)
        return [] # Return an empty list on any error.

if __name__ == '__main__':
    # Simple example usage
    articles = get_rss_news("http://feeds.bbci.co.uk/news/rss.xml")
    if articles:
        for article in articles:
            print(article + "\n---\n")

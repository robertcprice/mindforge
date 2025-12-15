"""
Conch Web Tool

Provides web search and fetch capabilities.
"""

import json
import logging
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Optional
from pathlib import Path

from conch.tools.base import Tool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class HTMLTextExtractor(HTMLParser):
    """Extract text content from HTML."""

    def __init__(self):
        super().__init__()
        self.result = []
        self.skip_tags = {"script", "style", "head", "meta", "link", "noscript"}
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag.lower()

    def handle_endtag(self, tag):
        self.current_tag = None

    def handle_data(self, data):
        if self.current_tag not in self.skip_tags:
            text = data.strip()
            if text:
                self.result.append(text)

    def get_text(self) -> str:
        return "\n".join(self.result)


class WebTool(Tool):
    """Tool for web operations.

    Features:
    - Web page fetching
    - HTML to text conversion
    - Basic search simulation
    - URL validation
    - Content extraction
    """

    # User agent for requests
    USER_AGENT = "Conch/1.0 (AI Assistant; Educational Purpose)"

    # Blocked domains for safety
    BLOCKED_DOMAINS = [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "internal",
        "private",
        "intranet",
    ]

    def __init__(
        self,
        timeout: int = 30,
        max_content_size: int = 1024 * 1024,  # 1MB
        cache_dir: Optional[Path] = None,
    ):
        """Initialize web tool.

        Args:
            timeout: Request timeout in seconds
            max_content_size: Maximum content size to fetch
            cache_dir: Directory for caching responses
        """
        super().__init__(
            name="web",
            description="Web operations (fetch, search, extract)",
            requires_confirmation=False,
        )

        self.timeout = timeout
        self.max_content_size = max_content_size
        self.cache_dir = cache_dir

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute a web operation.

        Args:
            operation: Operation name (fetch, search, extract, validate)
            **kwargs: Operation-specific arguments

        Returns:
            ToolResult
        """
        operations = {
            "fetch": self._fetch,
            "search": self._search,
            "extract": self._extract,
            "validate": self._validate_url,
        }

        if operation not in operations:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Unknown operation: {operation}. Available: {list(operations.keys())}",
            )

        try:
            return operations[operation](**kwargs)
        except Exception as e:
            logger.error(f"Web {operation} failed: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
            )

    def _is_url_safe(self, url: str) -> tuple[bool, str]:
        """Check if a URL is safe to access."""
        try:
            parsed = urllib.parse.urlparse(url)

            # Check scheme
            if parsed.scheme not in ("http", "https"):
                return False, f"Invalid scheme: {parsed.scheme}"

            # Check blocked domains
            hostname = parsed.hostname or ""
            for blocked in self.BLOCKED_DOMAINS:
                if blocked in hostname.lower():
                    return False, f"Blocked domain: {hostname}"

            # Check for IP addresses (potential internal access)
            if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname):
                # Allow only public IP ranges (simplified check)
                parts = [int(p) for p in hostname.split(".")]
                if parts[0] in (10, 127) or (parts[0] == 192 and parts[1] == 168):
                    return False, "Private IP address not allowed"
                if parts[0] == 172 and 16 <= parts[1] <= 31:
                    return False, "Private IP address not allowed"

            return True, "OK"
        except Exception as e:
            return False, str(e)

    def _fetch(
        self,
        url: str,
        extract_text: bool = True,
        headers: Optional[dict] = None,
    ) -> ToolResult:
        """Fetch a web page.

        Args:
            url: URL to fetch
            extract_text: Whether to extract text from HTML
            headers: Additional headers
        """
        start_time = time.time()

        # Safety check
        is_safe, reason = self._is_url_safe(url)
        if not is_safe:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error=f"URL blocked: {reason}",
                execution_time=time.time() - start_time,
            )

        try:
            # Build request
            req_headers = {
                "User-Agent": self.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            if headers:
                req_headers.update(headers)

            request = urllib.request.Request(url, headers=req_headers)

            # Fetch content
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                content_length = int(response.headers.get("Content-Length", 0))

                # Check size
                if content_length > self.max_content_size:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        output="",
                        error=f"Content too large: {content_length} bytes",
                        execution_time=time.time() - start_time,
                    )

                # Read content
                content = response.read(self.max_content_size)

                # Decode
                encoding = "utf-8"
                if "charset=" in content_type:
                    encoding = content_type.split("charset=")[-1].split(";")[0].strip()

                try:
                    text = content.decode(encoding)
                except UnicodeDecodeError:
                    text = content.decode("utf-8", errors="ignore")

                # Extract text from HTML
                if extract_text and "html" in content_type.lower():
                    extractor = HTMLTextExtractor()
                    extractor.feed(text)
                    text = extractor.get_text()

                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=text,
                    execution_time=time.time() - start_time,
                    metadata={
                        "url": url,
                        "content_type": content_type,
                        "size": len(content),
                    },
                )

        except urllib.error.HTTPError as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"HTTP {e.code}: {e.reason}",
                execution_time=time.time() - start_time,
            )
        except urllib.error.URLError as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"URL error: {e.reason}",
                execution_time=time.time() - start_time,
            )
        except TimeoutError:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                output="",
                error=f"Request timed out after {self.timeout} seconds",
                execution_time=time.time() - start_time,
            )

    def _search(
        self,
        query: str,
        num_results: int = 5,
    ) -> ToolResult:
        """Simulate web search.

        Note: This is a placeholder. In production, integrate with
        a search API (DuckDuckGo, Tavily, etc.)

        Args:
            query: Search query
            num_results: Number of results to return
        """
        start_time = time.time()

        # For now, return a message about search capabilities
        # In production, integrate with actual search API
        output = f"""Search Query: {query}

Note: Web search requires API integration.

Recommended search APIs for production:
1. DuckDuckGo Instant Answer API (free, no key)
2. Tavily API (AI-optimized search)
3. SerpAPI (comprehensive but paid)
4. Brave Search API (privacy-focused)

To implement:
1. Install appropriate client library
2. Configure API keys in environment
3. Update this method to call actual API

For now, you can use the 'fetch' operation to retrieve
specific URLs directly.
"""

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=output,
            execution_time=time.time() - start_time,
            metadata={"query": query, "simulated": True},
        )

    def _extract(
        self,
        url: str,
        selector: Optional[str] = None,
        extract_links: bool = False,
        extract_images: bool = False,
    ) -> ToolResult:
        """Extract structured content from a web page.

        Args:
            url: URL to extract from
            selector: CSS-like selector (simplified)
            extract_links: Whether to extract all links
            extract_images: Whether to extract image URLs
        """
        start_time = time.time()

        # First fetch the page
        fetch_result = self._fetch(url, extract_text=False)
        if not fetch_result.success:
            return fetch_result

        html = fetch_result.output
        extracted = {"url": url}

        # Extract title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if title_match:
            extracted["title"] = title_match.group(1).strip()

        # Extract meta description
        desc_match = re.search(
            r'<meta[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']',
            html,
            re.IGNORECASE,
        )
        if desc_match:
            extracted["description"] = desc_match.group(1).strip()

        # Extract links
        if extract_links:
            links = re.findall(r'<a[^>]*href=["\'](https?://[^"\']+)["\']', html, re.IGNORECASE)
            extracted["links"] = list(set(links))[:50]  # Limit to 50 unique links

        # Extract images
        if extract_images:
            images = re.findall(r'<img[^>]*src=["\'](https?://[^"\']+)["\']', html, re.IGNORECASE)
            extracted["images"] = list(set(images))[:20]  # Limit to 20 unique images

        # Extract main text
        extractor = HTMLTextExtractor()
        extractor.feed(html)
        extracted["text"] = extractor.get_text()

        # Format output
        output_lines = [f"URL: {url}"]
        if "title" in extracted:
            output_lines.append(f"Title: {extracted['title']}")
        if "description" in extracted:
            output_lines.append(f"Description: {extracted['description']}")
        if "links" in extracted:
            output_lines.append(f"Links ({len(extracted['links'])}): {', '.join(extracted['links'][:5])}...")
        if "images" in extracted:
            output_lines.append(f"Images ({len(extracted['images'])}): {', '.join(extracted['images'][:3])}...")
        output_lines.append(f"\nContent:\n{extracted['text'][:2000]}...")

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output="\n".join(output_lines),
            execution_time=time.time() - start_time,
            metadata=extracted,
        )

    def _validate_url(self, url: str) -> ToolResult:
        """Validate a URL without fetching.

        Args:
            url: URL to validate
        """
        start_time = time.time()

        is_safe, reason = self._is_url_safe(url)

        if is_safe:
            parsed = urllib.parse.urlparse(url)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"URL is valid: {url}",
                execution_time=time.time() - start_time,
                metadata={
                    "valid": True,
                    "scheme": parsed.scheme,
                    "hostname": parsed.hostname,
                    "path": parsed.path,
                },
            )
        else:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Invalid URL: {reason}",
                execution_time=time.time() - start_time,
                metadata={"valid": False, "reason": reason},
            )


# Convenience functions
_web_tool: Optional[WebTool] = None


def get_web() -> WebTool:
    """Get the global web tool instance."""
    global _web_tool
    if _web_tool is None:
        _web_tool = WebTool()
    return _web_tool


def fetch_url(url: str, **kwargs) -> str:
    """Fetch a URL and return its content."""
    result = get_web().execute("fetch", url=url, **kwargs)
    if result.success:
        return result.output
    raise RuntimeError(result.error)


def search_web(query: str, **kwargs) -> str:
    """Search the web (requires API integration)."""
    result = get_web().execute("search", query=query, **kwargs)
    return result.output

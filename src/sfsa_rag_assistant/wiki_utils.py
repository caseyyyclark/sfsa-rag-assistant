# src/sfsa_rag_assistant/wiki_utils.py
"""
Utilities for constructing SFSA Wiki URLs from document metadata.

The SFSA Wiki uses MediaWiki's img_auth.php structure with MD5 hashing:
https://wiki.sfsa.org/img_auth.php/{first_char_of_md5}/{first_two_chars_of_md5}/{filename}

Examples:
- Rr008.pdf → https://wiki.sfsa.org/img_auth.php/c/c3/Rr008.pdf
- Handbook_4th._Ed._Chapter_25._Quality_Control.pdf → 
  https://wiki.sfsa.org/img_auth.php/b/b4/Handbook_4th._Ed._Chapter_25._Quality_Control.pdf
"""

import hashlib
import os
from typing import Optional, Dict, Any
from urllib.parse import quote


def construct_wiki_url(filename: str, page: Optional[int] = None) -> str:
    """
    Construct SFSA Wiki URL from filename using MediaWiki's MD5 hashing structure.
    
    Parameters
    ----------
    filename : str
        The PDF filename (e.g., "Rr008.pdf")
    page : Optional[int]
        Page number to link to (adds #page=N anchor)
    
    Returns
    -------
    str
        Full URL to the document on SFSA Wiki
    
    Examples
    --------
    >>> construct_wiki_url("Rr008.pdf")
    'https://wiki.sfsa.org/img_auth.php/c/c3/Rr008.pdf'
    
    >>> construct_wiki_url("Rr008.pdf", page=5)
    'https://wiki.sfsa.org/img_auth.php/c/c3/Rr008.pdf#page=5'
    """
    # Calculate MD5 hash of filename
    md5_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()
    
    # Extract first character and first two characters
    first_char = md5_hash[0]
    first_two = md5_hash[:2]
    
    # URL encode the filename (handles spaces and special characters)
    encoded_filename = quote(filename)
    
    # Construct base URL
    base_url = f"https://wiki.sfsa.org/img_auth.php/{first_char}/{first_two}/{encoded_filename}"
    
    # Add page anchor if specified
    if page is not None:
        return f"{base_url}#page={page}"
    
    return base_url


def extract_wiki_url_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    """
    Extract and construct wiki URL from document metadata.
    
    Parameters
    ----------
    metadata : Dict[str, Any]
        Document metadata dictionary containing 'source' and optionally 'page'
    
    Returns
    -------
    Optional[str]
        Wiki URL if source is available, None otherwise
    
    Examples
    --------
    >>> metadata = {"source": "/path/to/Rr008.pdf", "page": 5}
    >>> extract_wiki_url_from_metadata(metadata)
    'https://wiki.sfsa.org/img_auth.php/c/c3/Rr008.pdf#page=5'
    """
    if "source" not in metadata:
        return None
    
    # Extract filename from path (handle both / and \ separators)
    source = metadata["source"]
    # Use os.path.basename to handle both Unix and Windows paths properly
    # Also manually handle mixed separators
    filename = os.path.basename(source.replace('\\', '/'))
    
    # Get page number if available
    page = metadata.get("page")
    
    # Construct and return URL
    return construct_wiki_url(filename, page)


def format_terminal_link(url: str, display_text: str) -> str:
    """
    Format a clickable hyperlink for terminal output (OSC 8 standard).
    
    Many modern terminals (iTerm2, GNOME Terminal, Windows Terminal, etc.)
    support clickable links using the OSC 8 escape sequence.
    
    Parameters
    ----------
    url : str
        The URL to link to
    display_text : str
        The text to display (what user clicks on)
    
    Returns
    -------
    str
        Formatted string with terminal hyperlink escape codes
    
    Notes
    -----
    Format: \\033]8;;{url}\\007{display_text}\\033]8;;\\007
    If terminal doesn't support links, it will just show the display text.
    """
    # OSC 8 hyperlink format
    # Start: \033]8;;{url}\007
    # End: \033]8;;\007
    return f"\033]8;;{url}\007{display_text}\033]8;;\007"


if __name__ == "__main__":
    # Test the URL construction
    test_cases = [
        ("Rr008.pdf", None),
        ("Handbook_4th._Ed._Chapter_25._Quality_Control.pdf", None),
        ("Steel_Castings_Handbook_4th_Edition.pdf", 58),
        ("TO-2011-1.9_Monroe_-_SFSA.pdf", 6),
    ]
    
    print("Testing SFSA Wiki URL construction:")
    print("=" * 80)
    for filename, page in test_cases:
        url = construct_wiki_url(filename, page)
        print(f"\nFilename: {filename}")
        if page:
            print(f"Page: {page}")
        print(f"URL: {url}")
        print(f"Clickable: {format_terminal_link(url, filename)}")
    
    # Test extracting from various path formats
    print("\n\n" + "=" * 80)
    print("Testing path extraction:")
    print("=" * 80)
    
    path_test_cases = [
        {"source": "data\\raw\\wikidocs\\images\\7\\76\\TO-2011-1.9_Monroe_-_SFSA.pdf", "page": 6},
        {"source": "data/raw/Rr008.pdf", "page": 5},
        {"source": "/Users/user/Documents/Steel_Castings_Handbook_4th_Edition.pdf", "page": 58},
        {"source": "C:\\Users\\user\\Documents\\Handbook_4th._Ed._Chapter_25._Quality_Control.pdf"},
    ]
    
    for metadata in path_test_cases:
        url = extract_wiki_url_from_metadata(metadata)
        print(f"\nPath: {metadata['source']}")
        print(f"URL: {url}")

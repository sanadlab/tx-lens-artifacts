#!/usr/bin/env python3
"""
sota_eth_tx_labeler.py

Professional-grade TX labeler.
Improvements over version 7:
1.  **Robust Waiting:** Replaces the brittle `wait_for_timeout(10000)` with a 
    "poll-and-scroll" loop. This repeatedly checks the page, scrolls to 
    trigger lazy-loaded content, and waits, ensuring dynamic labels 
    (like late-loading security reports) are caught.
2.  **Anti-Bot Detection:** Implements "stealth" measures by setting a common
    User-Agent, Viewport, and disabling the `navigator.webdriver` flag to 
    appear more human and avoid CAPTCHAs or blocked content.
3.  **Expanded Keywords:** Adds more keywords to the list for better coverage.

Usage:
  pip install playwright
  python -m playwright install chromium
  python3 sota_eth_tx_labeler.py -i tx_day.csv -o sota_labeled_output.csv --delay 10.0 --skip-header
"""
import argparse
import asyncio
import csv
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# --- Sites ---
URL_TEMPLATES = {
    "etherscan": "https://etherscan.io/tx/{tx}",
    "blocksec": "https://app.blocksec.com/explorer/tx/eth/{tx}",
    "webacy": "https://dapp.webacy.com/dyor/tx/{tx}?chain=eth",
    "certik": "https://skylens.certik.com/tx/eth/{tx}"
}

# --- Anti-Bot/Stealth Constants ---
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
VIEWPORT = {'width': 1920, 'height': 1080}

# --- Keyword List (Expanded) ---
COMMON_KW = [
    "exploit", "exploiter", "exploited", "exploitation",
    "hack", "hacker", "hacked", "hacking",
    "attack", "attacker",
    "drain", "drained", "draining", "siphon",
    "compromised", "stolen",
    "rugpull", "rug-pull", "rugpull",
    "honeypot", "backdoor",
    "phishing", "fraud", "scam", "scammer",
    "vulnerability", "vulnerable", "critical",
    "malicious", "malware",
    "suspicious", "risky", "high-risk", # Added keywords
]

# Build a single combined regex: match token that contains any keyword,
# allowing letters/digits/underscore around the keyword (so "Flashloan_Attacker_Contract" matches "attack")
def build_single_combined_regex(keywords: List[str]) -> re.Pattern:
    # sort by length desc to prefer longer alternates
    kws = sorted({kw.strip() for kw in keywords if kw.strip()}, key=len, reverse=True)
    escaped = [re.escape(k) for k in kws]
    inner = "|".join(escaped)
    # match tokens that contain the keyword with optional alnum/_ around it
    pattern = rf"[A-Za-z0-9_]*?(?:{inner})[A-Za-z0-9_]*"
    return re.compile(pattern, flags=re.IGNORECASE)

SINGLE_KW_REGEX = build_single_combined_regex(COMMON_KW)

# shadow-DOM-aware extractor (same as original)
EXTRACT_JS = r"""
() => {
  function getText(node) {
    if (!node) return "";
    var text = "";
    if (node.nodeType === Node.TEXT_NODE) {
      return node.textContent + " ";
    }
    if (node.shadowRoot) {
      text += getText(node.shadowRoot);
    }
    var child = node.firstChild;
    while (child) {
      text += getText(child);
      child = child.nextSibling;
    }
    return text;
  }
  return getText(document.documentElement || document);
}
"""

def normalize_tx(tx: str) -> str:
    t = tx.strip()
    if not t:
        return t
    if t.startswith("0x"):
        return t.lower()
    if all(c in "0123456789abcdefABCDEF" for c in t) and len(t) in (64, 66):
        return "0x" + t.lower()
    return t

async def fetch_rendered_text_and_html(page) -> Tuple[str, str]:
    """
    Simpler extractor that *only* extracts text/HTML from the current state.
    The waiting logic is now handled by the caller.
    """
    text, html = "", ""
    try:
        text = await page.evaluate(EXTRACT_JS)
    except PlaywrightError as e:
        logging.debug(f"Could not evaluate EXTRACT_JS: {e}")
        pass  # text remains ""

    if not text:
        try:
            text = await page.inner_text("body")
        except Exception as e:
            logging.debug(f"Could not get inner_text('body'): {e}")
            pass  # text remains ""

    try:
        html = await page.content()
    except Exception as e:
        logging.debug(f"Could not get page.content(): {e}")
        pass  # html remains ""
    
    return (text or ""), (html or "")


async def visit_site_for_tx(browser_context, site: str, tx: str, args) -> Tuple[str, int, str]:
    """
    Visit a single site for a transaction.
    Returns tuple: (site, label (0/1), matches_str)
    
    IMPROVEMENT: This version uses a robust "poll-and-scroll" loop
    and waits for 'networkidle' to catch dynamic API calls.
    """
    url = URL_TEMPLATES[site].format(tx=tx)
    page = await browser_context.new_page()
    page.set_default_navigation_timeout(args.timeout_ms)
    
    # These headers are set in the context, but we can be explicit.
    await page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})
    
    matches_set = set()
    last_exc = None
    
    # --- Robust Poll-and-Scroll Loop ---
    # This gives dynamic content (like security reports) time to load.
    POLL_ATTEMPTS = 5       # Try 5 times (Increased)
    POLL_DELAY_MS = 3500  # Wait 3.5s between attempts (Increased)
    
    try:
        try:
            # *** KEY CHANGE ***
            # Wait for network to be idle, not just DOM.
            # This allows secondary API calls (for risk data) to complete.
            await page.goto(url, wait_until="networkidle", timeout=args.timeout_ms)
        except PlaywrightTimeoutError as e:
            last_exc = e
            logging.warning("Initial goto timeout (networkidle) for %s on tx %s", site, tx)
            # Don't try to poll if the page didn't even load
            raise

        # Give the page 1s to "settle" its JavaScript after network idle
        await page.wait_for_timeout(1000)

        for attempt in range(1, POLL_ATTEMPTS + 1):
            logging.debug("Poll attempt %d/%d for %s on %s", attempt, POLL_ATTEMPTS, tx, site)
            
            # On subsequent attempts, scroll to trigger lazy-load content
            if attempt > 1:
                try:
                    await page.evaluate('window.scrollTo(0, document.body.scrollHeight / 2)')
                    await page.wait_for_timeout(200) # small pause
                    await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                except Exception as scroll_e:
                    logging.debug("Scroll failed (page might be closed): %s", scroll_e)
                    break # Stop polling if page is broken
            
            text, html = await fetch_rendered_text_and_html(page)
            combined = (text or "") + "\n" + (html or "")
            
            # FAST prefilter (lowercased): cheap C-level substring checks
            lc = combined.lower()
            if not any(kw in lc for kw in COMMON_KW):
                # No keywords found yet.
                if attempt < POLL_ATTEMPTS:
                    # Wait and try again
                    await page.wait_for_timeout(POLL_DELAY_MS)
                    continue
                else:
                    # Last attempt, no keywords found.
                    logging.debug("No keywords found after %d polls for %s on %s", POLL_ATTEMPTS, tx, site)
                    break

            # Keywords *were* found in the pre-filter, now run the regex
            for m in SINGLE_KW_REGEX.finditer(combined):
                matches_set.add(m.group(0).lower())
            
            # Since we found matches, we can stop polling.
            if matches_set:
                logging.debug("Keywords found for %s on %s at attempt %d", tx, site, attempt)
                break
        
        # This replaces the original retry logic, which was less effective
        # than the polling loop.
        
    except Exception as e:
        if not last_exc:
            last_exc = e
        logging.warning("Error visiting %s for tx %s: %s", site, tx, e)
    finally:
        try:
            await page.close()
        except Exception:
            pass

    if matches_set:
        return (site, 1, ";".join(sorted(matches_set)))
    else:
        if last_exc:
            # Report the *first* error we encountered
            return (site, 0, f"ERROR:{type(last_exc).__name__}")
        return (site, 0, "")


async def process_single_tx(browser, tx: str, args, sites: List[str]) -> Dict[str, str]:
    row = {"tx": tx}
    
    # --- Apply Anti-Bot Detection Measures ---
    context = await browser.new_context(
        user_agent=USER_AGENT,
        viewport=VIEWPORT,
        locale='en-US',
        java_script_enabled=True,
    )
    try:
        # Hide the "webdriver" flag
        await context.add_init_script("() => { Object.defineProperty(navigator, 'webdriver', { get: () => undefined }) }")

        coros = [visit_site_for_tx(context, site, tx, args) for site in sites]
        results = await asyncio.gather(*coros, return_exceptions=True)
        
        for res in results:
            if isinstance(res, Exception):
                logging.exception("Unexpected exception while fetching for tx %s: %s", tx, res)
                for site in sites:
                    if f"{site}_label" not in row:
                        row[f"{site}_label"] = 0
                        row[f"{site}_matches"] = f"ERROR:Exception"
                break
            else:
                site, label, matches = res
                row[f"{site}_url"] = URL_TEMPLATES[site].format(tx=tx)
                row[f"{site}_label"] = label
                row[f"{site}_matches"] = matches
    finally:
        try:
            await context.close()
        except Exception:
            pass
    return row

# --- IO, runner, and CLI (Mostly unchanged) ---

def read_input(path: Path, skip_header: bool) -> List[str]:
    txs = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            if not row:
                continue
            if i == 0 and skip_header:
                continue
            tx = row[0].strip()
            if not tx:
                continue
            txs.append(normalize_tx(tx))
    return txs

def write_output(path: Path, rows: List[Dict], sites: List[str]):
    fieldnames = ["tx"]
    for s in sites:
        fieldnames += [f"{s}_url", f"{s}_label", f"{s}_matches"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            w.writerow(out)

def parse_args():
    p = argparse.ArgumentParser(description="Professional, robust Ethereum tx labeler")
    p.add_argument("--input", "-i", required=True, help="Input CSV (first column tx hash).")
    p.add_argument("--output", "-o", required=True, help="Output CSV") # <-- FIX was here
    p.add_argument("--delay", type=float, default=1.0, help="Delay between transactions in seconds (default 1.0)")
    p.add_argument("--timeout-ms", type=int, default=180000, help="Navigation timeout ms (default 180000)")
    p.add_argument("--headless", dest="headless", action="store_true", help="Run headless (default)")
    p.add_argument("--no-headless", dest="headless", action="store_false", help="Run with browser UI")
    p.add_argument("--skip-header", dest="skip_header", action="store_true", help="Skip first row (default True)")
    p.add_argument("--no-skip-header", dest="skip_header", action="store_false", help="Do not skip header")
    p.add_argument("--sites", nargs="+", default=["etherscan","blocksec","webacy","certik"], help="Sites to check")
    # Retries are now handled by the polling loop, so --retries is removed
    
    p.set_defaults(headless=True, skip_header=True)
    return p.parse_args()

async def run_all_sequential(txs: List[str], args):
    rows = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=args.headless,
            # Another anti-detection flag
            args=['--disable-blink-features=AutomationControlled']
        )
        try:
            for idx, tx in enumerate(txs, start=1):
                logging.info("Processing tx %d/%d : %s", idx, len(txs), tx)
                row = await process_single_tx(browser, tx, args, args.sites)
                rows.append(row)
                if args.delay and idx != len(txs):
                    await asyncio.sleep(args.delay)
        finally:
            try:
                await browser.close()
            except Exception:
                pass
    return rows

def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        logging.error("Input not found: %s", in_path)
        sys.exit(1)
    
    txs = read_input(in_path, skip_header=args.skip_header)
    
    if not txs:
        logging.error("No transactions found in input")
        sys.exit(1)
        
    logging.info("Loaded %d tx. Sites=%s Delay=%.2fs Timeout=%dms", 
                 len(txs), args.sites, args.delay, args.timeout_ms)
    
    rows = asyncio.run(run_all_sequential(txs, args))
    
    write_output(out_path, rows, args.sites)
    logging.info("Wrote %s rows=%d", out_path, len(rows))

if __name__ == "__main__":
    main()


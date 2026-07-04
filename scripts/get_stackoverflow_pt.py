#!/usr/bin/env python3
"""Download the pt.stackoverflow.com Stack Exchange dump, extract question titles+bodies
(strip HTML) as PT tech text for the RE-TRIM vocab. Output: data/stack_pt.parquet."""
import os, re, html, xml.etree.ElementTree as ET
import requests, py7zr, pandas as pd
URL = "https://archive.org/download/stackexchange/pt.stackoverflow.com.7z"
D = "/private/tmp/claude-501/-Users-tardelli-Workplace-embedding-vocab-trimmer/9252dfb3-8362-4061-94bf-f28e61c83f58/scratchpad/se_pt"
os.makedirs(D, exist_ok=True)
z = os.path.join(D, "pt.7z")
if not os.path.exists(z) or os.path.getsize(z) < 4e8:
    print("downloading dump...", flush=True)
    with requests.get(URL, stream=True) as r:
        r.raise_for_status()
        with open(z, "wb") as f:
            for c in r.iter_content(1 << 20):
                f.write(c)
    print("downloaded", os.path.getsize(z), flush=True)
print("extracting Posts.xml...", flush=True)
with py7zr.SevenZipFile(z, "r") as a:
    a.extract(path=D, targets=["Posts.xml"])
tag = re.compile(r"<[^>]+>")
rows = []
for _, el in ET.iterparse(os.path.join(D, "Posts.xml")):
    if el.tag == "row" and el.get("PostTypeId") == "1":
        title = el.get("Title") or ""
        body = html.unescape(tag.sub(" ", el.get("Body") or ""))
        t = re.sub(r"\s+", " ", (title + " " + body)).strip()
        if len(t) >= 80:
            rows.append(t)
    el.clear()
    if len(rows) >= 45000:
        break
pd.DataFrame({"text": rows, "domain": "stackoverflow-pt"}).to_parquet(
    "/Users/tardelli/Workplace/improving-gemma-test/data/stack_pt.parquet", index=False)
print("STACK_PT:", len(rows), "questions saved", flush=True)

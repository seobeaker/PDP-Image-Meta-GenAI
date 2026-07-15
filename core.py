"""
core.py — Image -> SEO generator (v5.2), ported from the Colab notebook.

All business logic lives here: brand tones, image resolution, the OpenAI
vision prompt, post-processing, and OpenAI Batch API helpers. Nothing in
this file imports Streamlit or ipywidgets, so it can be unit-tested or
reused outside the app.
"""

import re
import io
import json
import base64
import unicodedata
from io import BytesIO
from urllib.parse import urljoin
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import requests
import pandas as pd
from openai import OpenAI

try:
    from bs4 import BeautifulSoup
    _HAVE_BS4 = True
except Exception:
    _HAVE_BS4 = False

# ------------- CONFIG -------------
# NOTE: model IDs change over time. "gpt-5.4-mini" is the recommended default,
# but if a call fails with a "model not found" error, open your OpenAI
# dashboard (Models / Pricing) and paste the EXACT id into the model box.
DEFAULT_MODEL = "gpt-5.4-mini"
MODEL_CHOICES = ["gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "gpt-4o"]
TIMEOUT = 25
PAUSE = 0.6            # used only in "process now" mode
MAX_SIDE = 768          # downscale for the data-url fallback
IMAGE_DETAIL = "low"    # "low" keeps image-token cost down; SEO copy rarely needs more

# Output columns / headers (ID first; two renamed headers for upload)
OUTPUT_COLUMNS = ["ID", "url", "pageTitle__default", "pageDescription__default",
                  "h1", "short_copy", "alt_text", "used_image", "attrs_json", "warnings"]
# Input column names that may hold the product id (matched case-insensitively)
ID_KEYS = ["id", "product id", "product_id", "product-id", "productid", "sku"]

# ---- BRAND TONES ----
BRAND_TONES: Dict[str, dict] = {
    "Cotton On": {
        "persona": "Casual, inclusive, everyday style for all occasions.",
        "voice": {"sentence_length": "medium", "vocabulary": ["everyday", "style", "comfort", "casual", "essentials"],
                  "avoid": ["luxury", "exclusive", "premium", "guaranteed"], "emoji": "never"},
        "style": {"cta": ["Shop now", "Update your wardrobe"], "claims": "accessible, inclusive, no hype"}
    },
    "Cotton On Kids": {
        "persona": "Playful, bright, family-friendly; designed for kids and parents.",
        "voice": {"sentence_length": "short-to-medium", "vocabulary": ["play", "fun", "kids", "family", "everyday wear"],
                  "avoid": ["best", "No.1", "miracle", "luxury"], "emoji": "rare, playful use acceptable"},
        "style": {"cta": ["Shop for the little ones", "Make playtime fun"], "claims": "joyful, safe, practical"}
    },
    "Cotton On Body": {
        "persona": "Supportive, body-positive, feel-good everyday essentials.",
        "voice": {"sentence_length": "medium", "vocabulary": ["comfort", "support", "feel-good", "soft", "active"],
                  "avoid": ["perfect body", "flawless", "miracle", "best"], "emoji": "never"},
        "style": {"cta": ["Find your fit", "Feel the comfort"], "claims": "inclusive, confidence-building"}
    },
    "Factorie": {
        "persona": "Youthful, street-ready, casual essentials with an edge.",
        "voice": {"sentence_length": "short", "vocabulary": ["street", "laidback", "casual", "essentials", "everyday"],
                  "avoid": ["luxury", "premium", "heritage", "guaranteed"], "emoji": "never"},
        "style": {"cta": ["Add to your rotation", "Grab it now"], "claims": "casual, accessible, grounded"}
    },
    "Supre": {
        "persona": "Bold, expressive, fashion-forward for young women.",
        "voice": {"sentence_length": "short-to-medium", "vocabulary": ["bold", "statement", "trend", "style", "must-have"],
                  "avoid": ["luxury", "exclusive", "guaranteed"], "emoji": "rare, only for campaign emphasis"},
        "style": {"cta": ["Own your look", "Make it yours"], "claims": "trend-led, expressive, empowering"}
    },
    "Rubi": {
        "persona": "Affordable, stylish accessories and footwear for every day.",
        "voice": {"sentence_length": "short-to-medium", "vocabulary": ["shoes", "bag", "style", "accessories", "everyday"],
                  "avoid": ["luxury", "premium", "guaranteed"], "emoji": "never"},
        "style": {"cta": ["Step out in style", "Complete your look"], "claims": "value-driven, stylish, versatile"}
    },
    "Typo": {
        "persona": "Playful, cheeky, creative, anything but ordinary.",
        "voice": {"sentence_length": "short-to-medium", "vocabulary": ["play", "create", "fun", "gift", "desk", "study"],
                  "avoid": ["best", "No.1", "guaranteed", "medical"], "emoji": "rare, context-appropriate only"},
        "style": {"cta": ["Gift it", "Add to cart", "Make it yours"], "claims": "fun, creative, value-driven"}
    },
}
DEFAULT_BRAND = "Typo"


# ------------- TEXT HELPERS -------------
def clamp(s: str, hi: Optional[int] = None) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    if hi and len(s) > hi:
        s = s[:hi].rstrip(" ,.;-")
    return s


PUNCT_MAP = {
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00A0": " ",
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"', "\u2033": '"',
    "\u2026": "...",
}


def normalize_ascii(s: str) -> str:
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    for k, v in PUNCT_MAP.items():
        s = s.replace(k, v)
    return s


def strip_banned(text: str, banned) -> str:
    """Remove banned words/phrases as WHOLE words, case-insensitive.
    'bestseller' is preserved; 'best deal' -> 'deal'. Cleans up leftover spacing."""
    if not text:
        return text
    for b in banned:
        b = (b or "").strip()
        if not b:
            continue
        text = re.sub(rf"(?<!\w){re.escape(b)}(?!\w)", "", text, flags=re.I)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)          # kill space before punctuation
    text = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", text)   # collapse orphaned separators ",," -> ","
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.;:-")
    return text


# ------------- BRAND / TONE -------------
def resolve_tone_profile(brand: str, tone_str: str, default_brand: str) -> Tuple[dict, str, str]:
    """Returns (profile, brand_used, tone_override).
    A free-text tone that is NOT a brand key becomes a tone_override the model honours."""
    b = (brand or "").strip()
    t = (tone_str or "").strip()
    if b in BRAND_TONES:
        profile, brand_used = BRAND_TONES[b], b
    elif t in BRAND_TONES:
        profile, brand_used = BRAND_TONES[t], t
    elif default_brand in BRAND_TONES:
        profile, brand_used = BRAND_TONES[default_brand], default_brand
    else:
        profile, brand_used = BRAND_TONES[DEFAULT_BRAND], DEFAULT_BRAND
    tone_override = t if (t and t not in BRAND_TONES) else ""
    return profile, brand_used, tone_override


def banned_for(profile: Optional[dict]):
    banned = {"No.1", "best", "guaranteed", "clinically proven", "medical"}
    if profile and profile.get("voice", {}).get("avoid"):
        banned.update(profile["voice"]["avoid"])
    return banned


# ------------- IMAGE RESOLUTION -------------
def is_direct_image_url(url: str) -> bool:
    u = (url or "").lower().split("?")[0].split("#")[0]
    return u.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))


def extract_image_from_html(html: str, base_url: str) -> Optional[str]:
    """Robust og:image / twitter:image scrape (any attribute order, relative URLs)."""
    pref = ["og:image", "og:image:secure_url", "og:image:url", "twitter:image", "twitter:image:src"]
    if _HAVE_BS4:
        soup = BeautifulSoup(html, "html.parser")
        found = {}
        for tag in soup.find_all("meta"):
            key = (tag.get("property") or tag.get("name") or "").strip().lower()
            content = tag.get("content")
            if key in pref and content and key not in found:
                found[key] = content.strip()
        for key in pref:
            if key in found:
                return urljoin(base_url, found[key])
        return None
    for key in pref:
        for pat in (rf'(?:property|name)=["\']{re.escape(key)}["\']\s+content=["\']([^"\']+)["\']',
                    rf'content=["\']([^"\']+)["\']\s+(?:property|name)=["\']{re.escape(key)}["\']'):
            m = re.search(pat, html, re.I)
            if m:
                return urljoin(base_url, m.group(1).strip())
    return None


def fetch_og_image(page_url: str) -> Optional[str]:
    try:
        r = requests.get(page_url, timeout=TIMEOUT, headers={"User-Agent": "ImgSEO/2.0"})
        r.raise_for_status()
        return extract_image_from_html(r.text, page_url)
    except Exception:
        return None


def resolve_image_for_vision(url: str) -> Optional[str]:
    url = (url or "").strip()
    if not url:
        return None
    if is_direct_image_url(url):
        return url
    return fetch_og_image(url)


def download_and_resize_to_data_url(img_url: str):
    """Fallback for 'process now' mode when the model can't fetch a URL directly."""
    from PIL import Image
    try:
        r = requests.get(img_url, timeout=TIMEOUT, headers={"User-Agent": "ImgSEO/2.0"})
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        w_, h_ = img.size
        scale = min(1.0, MAX_SIDE / max(w_, h_))
        if scale < 1.0:
            img = img.resize((int(w_ * scale), int(h_ * scale)))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


# ------------- PROMPT -------------
SYSTEM_PROMPT = """
You generate SEO assets from a single product image.

Always follow the provided `tone_profile` JSON:
- persona: overall brand personality
- voice: guidance (sentence_length, vocabulary examples, words/claims to avoid, emoji policy)
- style: CTA list, claims policy

If `tone_override` is a non-empty string, treat it as an additional stylistic
instruction layered ON TOP of the brand voice (the brand's 'avoid' words and
claims policy still apply and win any conflict).

Rules:
- Return STRICT JSON with keys:
  title (<=70), meta_description (110-155), h1 (<=65), short_copy (<150 words),
  alt_text (<=125, plain literal description of the product for image alt text),
  attrs (object).
- 'avoid' words and claims are HARD constraints (never include).
- Vocabulary items are EXAMPLES, not a closed list.
- Titles should be natural, brand-appropriate, and keyword aligned.
- Keep copy compliant with the claims policy.
JSON only.
"""


def build_messages(hints: Dict[str, object], image_ref: str) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},  # stable prefix (cache-friendly)
        {"role": "user", "content": [
            {"type": "text", "text": json.dumps({"hints": hints}, ensure_ascii=False)},
            {"type": "image_url", "image_url": {"url": image_ref, "detail": IMAGE_DETAIL}},
        ]},
    ]


def make_hints(brand_used: str, profile: dict, tone_override: str, url: str, pid: str) -> Dict[str, object]:
    # _meta is recovered at collect time from the batch input file (no sidecar needed)
    return {
        "brand": brand_used,
        "tone_profile": profile,
        "tone_override": tone_override,
        "vocabulary_policy": "examples_not_constraints",
        "_meta": {"id": pid, "url": url, "brand": brand_used, "tone_override": tone_override},
    }


def call_openai_vision(client: "OpenAI", model: str, image_ref: str, hints: Dict[str, object]) -> Dict[str, object]:
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=build_messages(hints, image_ref),
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {}


# ------------- POST-PROCESS -------------
def postprocess(d: Dict[str, object], profile: Optional[dict] = None) -> Dict[str, str]:
    title = clamp(str(d.get("title", "")), hi=70)
    meta = clamp(str(d.get("meta_description", "")))
    if len(meta) > 155:
        meta = meta[:155].rstrip(" ,.;-")
    h1 = clamp(str(d.get("h1", "")), hi=65)
    short = clamp(str(d.get("short_copy", "")))
    alt = clamp(str(d.get("alt_text", "")), hi=125)

    banned = banned_for(profile)
    title = normalize_ascii(strip_banned(title, banned))
    meta = normalize_ascii(strip_banned(meta, banned))
    h1 = normalize_ascii(strip_banned(h1, banned))
    short = normalize_ascii(strip_banned(short, banned))
    alt = normalize_ascii(strip_banned(alt, banned))

    warn = []
    if meta and len(meta) < 110:
        warn.append("meta<110")
    if not title:
        warn.append("empty_title")
    return {"title": title, "meta_description": meta, "h1": h1,
            "short_copy": short, "alt_text": alt, "warnings": ";".join(warn)}


@dataclass
class RowOut:
    pid: str
    url: str
    used_image: str
    title: str
    meta_description: str
    h1: str
    short_copy: str
    alt_text: str
    attrs_json: str
    warnings: str


def row_record(o: "RowOut") -> dict:
    """Map a RowOut to the named output columns (ID first, renamed headers)."""
    return {
        "ID": o.pid,
        "url": o.url,
        "pageTitle__default": o.title,
        "pageDescription__default": o.meta_description,
        "h1": o.h1,
        "short_copy": o.short_copy,
        "alt_text": o.alt_text,
        "used_image": o.used_image,
        "attrs_json": o.attrs_json,
        "warnings": o.warnings,
    }


def process_row_sync(client: "OpenAI", model: str, pid: str, url: str, brand: str,
                      tone: str, default_brand: str) -> Optional[RowOut]:
    """Process (or reprocess) a single row synchronously — used by 'Process now'."""
    profile, brand_used, tone_override = resolve_tone_profile(brand, tone, default_brand)
    hints = make_hints(brand_used, profile, tone_override, url, pid)
    chosen = resolve_image_for_vision(url)
    if not chosen:
        return None
    data = call_openai_vision(client, model, chosen, hints)
    if not str(data.get("title", "")).strip():
        data_url = download_and_resize_to_data_url(chosen)
        if data_url:
            data = call_openai_vision(client, model, data_url, hints)
            chosen = f"(data-url from) {chosen}"
    clean = postprocess(data, profile)
    return RowOut(pid=pid, url=url, used_image=chosen, title=clean["title"],
                  meta_description=clean["meta_description"], h1=clean["h1"],
                  short_copy=clean["short_copy"], alt_text=clean["alt_text"],
                  attrs_json=json.dumps(data.get("attrs", {}), ensure_ascii=False),
                  warnings=clean["warnings"])


# ------------- BATCH HELPERS -------------
def build_batch_line(idx: int, pid: str, url: str, image_ref: str,
                      brand_used: str, profile: dict, tone_override: str, model: str) -> dict:
    hints = make_hints(brand_used, profile, tone_override, url, pid)
    return {
        "custom_id": f"row-{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": build_messages(hints, image_ref),
        },
    }


def parse_input_line_meta(line: dict) -> Tuple[str, dict, str]:
    """From a batch INPUT line, recover (custom_id, _meta, image_ref)."""
    cid = line["custom_id"]
    content = line["body"]["messages"][1]["content"]
    text = next(c["text"] for c in content if c.get("type") == "text")
    img = next(c["image_url"]["url"] for c in content if c.get("type") == "image_url")
    hints = json.loads(text)["hints"]
    return cid, hints.get("_meta", {}), img


def parse_output_line(line: dict) -> Tuple[str, dict, Optional[str]]:
    """From a batch OUTPUT line, recover (custom_id, parsed_json, error_str)."""
    cid = line.get("custom_id", "")
    if line.get("error"):
        return cid, {}, str(line["error"])
    try:
        body = line["response"]["body"]
        content = body["choices"][0]["message"]["content"]
        return cid, json.loads(content), None
    except Exception as e:
        return cid, {}, f"parse_error: {e}"


# ---- TABLE IO ----
def read_table(uploaded_file) -> pd.DataFrame:
    """Accept a Streamlit UploadedFile (.xlsx/.xls/.csv); return a cleaned df
    with columns: pid (product id, optional), url (required), brand, tone."""
    name = (uploaded_file.name or "").lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
    else:
        df = pd.read_excel(io.BytesIO(raw))

    cols = {str(c).strip().lower(): c for c in df.columns}
    if "url" not in cols:
        raise ValueError("Sheet must contain a 'url' column (case-insensitive).")
    df.rename(columns={cols["url"]: "url"}, inplace=True)

    # Optional product-id column under any of several names
    pid_src = next((cols[k] for k in ID_KEYS if k in cols), None)
    if pid_src and pid_src != "url":
        df.rename(columns={pid_src: "pid"}, inplace=True)
    else:
        df["pid"] = ""

    for opt in ("brand", "tone"):
        if opt in cols:
            df.rename(columns={cols[opt]: opt}, inplace=True)
        else:
            df[opt] = ""

    df["pid"] = df["pid"].fillna("").astype(str)
    df["brand"] = df["brand"].fillna("").astype(str)
    df["tone"] = df["tone"].fillna("").astype(str)
    df["url"] = df["url"].fillna("").astype(str)
    return df


def rows_to_frame(rows: List[RowOut]) -> pd.DataFrame:
    records = [row_record(o) for o in rows]
    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def frame_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def frame_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()

import re
from tokenizers import Regex
from tokenizers.normalizers import NFC, Replace, Sequence

# ============================================================
# TYPE B — lightweight, embeddable normalization
# ============================================================
def build_normalizer():

    return Sequence([
        NFC(),
        Replace(Regex( r"[\u064B-\u0652\u0640]"), "" ),
        Replace("أ", "ا"),
        Replace("إ", "ا"),
        Replace("آ", "ا"),
        Replace("ى", "ي"),   
        Replace(Regex(r"[ \t]+"), " "),  # collapse spaces/tabs (keep newlines)
    ])

_NORMALIZER = build_normalizer()

def normalize_text(text):
    """
    Lightweight normalization for a single string.
    """

    return _NORMALIZER.normalize_str(text)



# ============================================================
# TYPE A — heavy, one-time corpus preparation
# ============================================================

# English was cleaned properly upstream, but we cover its templates too.
_TEMPLATE_RE = re.compile(
    r"استشهاد\s*ب?(?:ويب|كتاب|دورية|خبر|مجلة|موسوعة|صحيفة)"   # AR cite templates
    r"|(?i:cite\s+(?:web|book|news|journal)|citation\s+needed)"  # EN cite templates
)

# These sections are link and citation dumps with little language value.
# Titles are compared after light normalization (see _norm_title).
_NAV_SECTION_TITLES = {
    # Arabic
    "مراجع", "المراجع", "مراجع ومصادر", "راجع", "مصادر", "المصادر",
    "وصلات خارجيه", "روابط خارجيه", "وصلات", "انظر ايضا", "طالع ايضا",
    "هوامش", "ملاحظات", "للاستزاده", "قائمه المراجع", "معرض الصور", "بوابات",
    # English
    "references", "reference", "see also", "external links", "notes",
    "further reading", "bibliography", "sources", "citations",
    "notes and references", "footnotes",
}

# Category / portal link lines like «تصنيف: ...» or «بوابة: ...».
_CATEGORY_LINE_RE = re.compile(r"(?m)^[ \t]*(?:تصنيف|بوابة|Category|Portal)\s*:.*$")

# MediaWiki heading «== title ==» (any level 2-6).
_MEDIAWIKI_HEADING_RE = re.compile(r"(?m)^[ \t]*=+[ \t]*(.+?)[ \t]*=+[ \t]*$")
# Markdown heading «## title».
_MARKDOWN_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]*(.+?)[ \t]*$")

MIN_SECTION_WORD = 8


def _norm_title(title):
    """Normalize a heading for blocklist comparison (AR + EN aware)."""
    t = title.strip().lower()
    t = re.sub(r"[\u064B-\u0652\u0640]", "", t)

    t = (t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
              .replace("ى", "ي").replace("ة", "ه"))
    return re.sub(r"\s+", " ", t).strip()

def split_into_sections(text):
    """
    Split an article chunk into (title, body) sections.

    Both heading styles are unified to markdown first, then we walk the lines.
    The text before the first heading is the intro section (title = "").
    """

    text = _MEDIAWIKI_HEADING_RE.sub(r"## \1", text) 

    sections = []
    title, body_lines = "", []

    for line in text.split("\n"):
        m = _MARKDOWN_HEADING_RE.match(line)
        if m:
            sections.append((title, "\n".join(body_lines)))
            title, body_lines = m.group(1).strip(), []
        else:
            body_lines.append(line)
    sections.append((title, "\n".join(body_lines)))
    return sections

def _dedupe_consecutive_lines(text):
    """Remove immediately-repeated lines (raw chunks duplicate titles/headers)."""
    out, prev = [], None
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and stripped == prev:
            continue
        out.append(line)
        prev = stripped
    return "\n".join(out)

def prepare_document(text , min_word = MIN_SECTION_WORD):
    
    text = _TEMPLATE_RE.sub("",text)
    text = _CATEGORY_LINE_RE.sub("",text)

    sections = split_into_sections(text = text)
    kept = []

    for title, body in sections:

        if _norm_title(title) in _NAV_SECTION_TITLES :
            continue 
        if len(body) < min_word:
            continue

        block = f"{title}\n{body}" if title else body 
        kept.append(block)

    cleaned_text = "\n\n".join(kept)
    cleaned_text = _dedupe_consecutive_lines(cleaned_text)

    return cleaned_text.strip()

def clean_dataframe(df, text_col="text", min_chars=40):

    new_df = df.copy()

    new_df["orig_len"] = new_df[text_col].str.len()

    new_df[text_col] = new_df[text_col].map(prepare_document)
    new_df = new_df[ new_df[text_col].str.len() >= min_chars ].reset_index(drop=True)

    return new_df
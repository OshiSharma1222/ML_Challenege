"""Text normalisation for business names and addresses.

Everything here is country-agnostic: any script is transliterated to ASCII with
anyascii, then cleaned. A phonetic "skeleton" (vowels dropped, similar consonants
merged) makes transliterated Indic names line up with their English spelling,
e.g. "praivet limitet" and "Private Limited" both become "prvt lnt".
"""
import re
from anyascii import anyascii

# ---------------------------------------------------------------- skeleton
_SK_SUBS = [
    (re.compile(r"ph"), "f"), (re.compile(r"sh"), "s"), (re.compile(r"ch"), "k"),
    (re.compile(r"ck"), "k"), (re.compile(r"[cq]"), "k"), (re.compile(r"x"), "ks"),
    (re.compile(r"z"), "s"), (re.compile(r"w"), "v"), (re.compile(r"j"), "g"),
    (re.compile(r"d"), "t"), (re.compile(r"m"), "n"), (re.compile(r"[aeiouyh]"), ""),
]
_REPEAT = re.compile(r"(.)\1+")


def skeleton(word: str) -> str:
    """Phonetic consonant key for one lowercase ascii word."""
    if word.isdigit():
        return word
    s = word
    for pat, rep in _SK_SUBS:
        s = pat.sub(rep, s)
    s = _REPEAT.sub(r"\1", s)
    return s or word[:2]


# ---------------------------------------------------------------- basic cleaning
_NONALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_NULLISH = re.compile(r"<null>|\bnull\b|\bnan\b|\bn/?a\b", re.I)


def to_ascii(s) -> str:
    if s is None:
        return ""
    s = anyascii(str(s)).lower()
    s = _NULLISH.sub(" ", s)
    return s


def clean(s: str) -> str:
    s = s.replace("&", " and ").replace("'", "").replace("`", "")
    s = _NONALNUM.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


# ---------------------------------------------------------------- names
LEGAL = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies", "llc", "l", "c",
    "ltd", "limited", "pvt", "private", "llp", "lp", "pllc", "plc", "pc", "pa", "na",
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "cie", "ets", "scop", "sca",
    "gmbh", "ag", "bv", "nv", "pty", "opc", "holdco",
}
# transliterated forms of legal words (compared through the skeleton)
LEGAL_SK = {skeleton(w) for w in ["private", "limited", "praivet", "limitet", "piraivet", "elelpi",
                                   "pvt", "ltd", "llp", "llc", "inc", "corp", "corporation",
                                   "incorporated", "company", "sarl", "sasu", "eurl"]}
STOP = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "et", "a", "an", "at", "for",
        "d", "l", "en", "by", "to", "in", "on", "et", "und"}
HONORIFIC = {"dr", "smt", "shri", "sri", "mr", "mrs", "ms", "m", "mme", "mlle", "the"}
_AKA = re.compile(r"\b(?:aka|dba|d b a|a k a|also known as|doing business as|t a|trading as)\b")
_WEB = re.compile(r"(?:https?\s*)?(?:www\s+)?([a-z0-9]+)\s+(?:com|net|org|in|co|fr|biz|info|us)\b")


def legal_kind(tokens):
    """Coarse legal-form code from the raw tokens (for a match/mismatch feature)."""
    t = set(tokens)
    if t & {"llc"}: return "llc"
    if t & {"llp"}: return "llp"
    if t & {"inc", "incorporated"}: return "inc"
    if t & {"corp", "corporation"}: return "corp"
    if t & {"pvt", "private"} or "prvt" in {skeleton(x) for x in t}: return "pvt"
    if t & {"ltd", "limited"}: return "ltd"
    if t & {"sarl", "eurl"}: return "sarl"
    if t & {"sas", "sasu"}: return "sas"
    if t & {"sa"}: return "sa"
    if t & {"lp", "pllc", "plc", "pc"}: return "lp"
    return ""


def name_parts(raw):
    """Return (full_clean, core, core_skeleton, legal_kind, n_alt, alt_core).

    core: name without legal suffixes/stop words/honorifics (word order preserved).
    alt_core: core of the part after "aka"/"dba" (or "" when absent).
    """
    s = to_ascii(raw)
    s = _WEB.sub(r" \1 ", s.replace(".", " "))
    s = clean(s)
    parts = [p.strip() for p in _AKA.split(s) if p.strip()]
    if not parts:
        parts = [""]
    toks_all = s.split()
    lk = legal_kind(toks_all)

    def core_of(p):
        out = []
        for w in p.split():
            if w in LEGAL or w in STOP or w in HONORIFIC:
                continue
            if len(w) > 3 and skeleton(w) in LEGAL_SK:
                continue
            out.append(w)
        if not out:  # name made only of legal words: keep them
            out = [w for w in p.split() if w not in STOP] or p.split()
        return " ".join(out)

    cores = [core_of(p) for p in parts]
    core = cores[0] if len(cores) == 1 else cores[-1]
    alt = cores[0] if len(cores) > 1 else ""
    sk = " ".join(skeleton(w) for w in core.split())
    return s, core, sk, lk, alt


# ---------------------------------------------------------------- addresses
ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "dr": "drive", "drv": "drive",
    "ln": "lane", "ct": "court", "crt": "court", "pl": "place", "plz": "plaza", "sq": "square",
    "pkwy": "parkway", "pky": "parkway", "hwy": "highway", "cir": "circle", "trl": "trail",
    "ter": "terrace", "terr": "terrace", "cres": "crescent", "expy": "expressway", "fwy": "freeway",
    "mt": "mount", "ft": "fort", "pt": "point", "apt": "apartment", "ste": "suite", "fl": "floor",
    "flr": "floor", "bldg": "building", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "r": "rue", "imp": "impasse", "all": "allee", "che": "chemin", "chem": "chemin", "rte": "route",
    "fbg": "faubourg", "qu": "quai", "sq": "square", "ch": "chemin",
    "ngr": "nagar", "mkt": "market", "opp": "opposite", "nr": "near", "sec": "sector",
    "hno": "house", "h": "house", "no": "number", "num": "number", "dist": "district",
    "vill": "village", "vpo": "village", "po": "post", "ps": "police", "tq": "taluka", "tal": "taluka",
    "extn": "extension", "ext": "extension", "colny": "colony", "col": "colony", "chs": "society",
    "chsl": "society", "soc": "society", "hsg": "housing", "indl": "industrial", "ind": "industrial",
    "saint": "saint", "ste.": "sainte",
}
# words carrying almost no locating signal (dropped from blocking tokens; kept in text)
ADDR_GENERIC = {"number", "house", "door", "flat", "plot", "unit", "suite", "apartment", "floor",
                "building", "near", "opposite", "post", "box", "po", "district", "the", "of",
                "de", "du", "des", "la", "le", "les", "and", "c", "o", "bis", "ter", "null", "shop",
                "office", "khasra", "survey", "ward", "block", "sector", "phase", "stage"}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "dc": "district of columbia", "pr": "puerto rico",
}
IN_STATES = {
    "ap": ["andhra pradesh"], "ar": ["arunachal pradesh"], "as": ["assam"], "br": ["bihar"],
    "cg": ["chhattisgarh", "chattisgarh", "ct"], "ga": ["goa"], "gj": ["gujarat", "gujrat"],
    "hr": ["haryana"], "hp": ["himachal pradesh"], "jh": ["jharkhand"], "ka": ["karnataka"],
    "kl": ["kerala", "keralam"], "mp": ["madhya pradesh"], "mh": ["maharashtra", "maharastra"],
    "mn": ["manipur"], "ml": ["meghalaya"], "mz": ["mizoram"], "nl": ["nagaland"],
    "od": ["odisha", "orissa", "or"], "pb": ["punjab"], "rj": ["rajasthan"], "sk": ["sikkim"],
    "tn": ["tamil nadu", "tamilnadu"], "ts": ["telangana", "tg"], "tr": ["tripura"],
    "uk": ["uttarakhand", "uttaranchal", "ut"], "up": ["uttar pradesh"],
    "wb": ["west bengal", "paschim banga", "paschimbanga", "pashchim banga", "paschimbanga"],
    "dl": ["delhi", "new delhi", "nct of delhi"], "jk": ["jammu and kashmir", "jammu kashmir"],
    "ch": ["chandigarh"], "py": ["puducherry", "pondicherry"], "ld": ["lakshadweep"],
    "an": ["andaman and nicobar islands"], "la": ["ladakh"], "dn": ["dadra and nagar haveli"],
}


def _state_tables():
    code = {}      # exact component text -> canonical state token
    sk = {}        # despaced skeleton of full state name -> token
    for c, n in US_STATES.items():
        code.setdefault(("us", c), "st_us_" + c)
        sk[skeleton(n.replace(" ", ""))] = "st_us_" + c
        code[("*", n)] = "st_us_" + c
    for c, names in IN_STATES.items():
        code[("india", c)] = "st_in_" + c
        for n in names:
            sk[skeleton(n.replace(" ", ""))] = "st_in_" + c
            code[("*", n)] = "st_in_" + c
    return code, sk


_STATE_CODE, _STATE_SK = _state_tables()
_NUM_ALPHA = re.compile(r"(\d+)(st|nd|rd|th)\b")
_DIGITS = re.compile(r"\d+")


def state_of(component: str, country: str):
    c = component.strip()
    if not c:
        return None
    ck = (country, c)
    if ck in _STATE_CODE:
        return _STATE_CODE[ck]
    if ("*", c) in _STATE_CODE:
        return _STATE_CODE[("*", c)]
    if len(c) > 3:
        return _STATE_SK.get(skeleton(c.replace(" ", "")))
    return None


def addr_parts(raw, country: str):
    """Return (addr_clean, addr_tokens(list), numbers(list), state_token or '')."""
    s = to_ascii(raw)
    if not s.strip():
        return "", [], [], ""
    cty = (country or "").strip().lower()
    comps = [clean(p) for p in re.split(r"[,;|]", s)]
    state = ""
    words = []
    for comp in comps:
        if not comp:
            continue
        st = state_of(comp, cty)
        if st:
            state = st
            continue
        for w in comp.split():
            words.append(w)
    toks, nums = [], []
    for w in words:
        m = _NUM_ALPHA.fullmatch(w)
        if m:
            w = m.group(1)
        if w.isdigit():
            w = w.lstrip("0") or "0"
            nums.append(w)
            toks.append(w)
            continue
        w = ADDR_ABBR.get(w, w)
        # mixed tokens like "9b", "a603", "12a" -> keep number part as a number too
        d = _DIGITS.findall(w)
        if d:
            for x in d:
                x = x.lstrip("0") or "0"
                nums.append(x)
        toks.append(w)
    text = " ".join(toks)
    return text, toks, nums, state

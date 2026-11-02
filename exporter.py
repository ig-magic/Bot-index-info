"""
================================================================================
 DATASET  ->  TELEGRAM  (private channel)   ·   BATCHED JSON  ·   FAST
================================================================================
 Ek message me JITNE RECORDS FIT HOTE HAIN utne bhejta hai (valid JSON array),
 isliye messages ki ginti 10-30x kam -> speed 10-30x zyada.

 Env (.env):
   HF_DATASET=username/dataset        (required)
   HF_TOKEN=                          (private dataset)
   HF_CONFIG=default  HF_SPLIT=train
   BOT_TOKEN=123456:ABC...
   CHANNEL_ID=-1001234567890123
   ADMIN_ID=123456789
   API_ID=1234567  API_HASH=abcd...   (MTProto -> 20 msg/min ka limit nahi)
   MODE=auto                          auto|user|bot
   BATCH_MODE=auto                    auto = message ko bhar bhar ke bhejo
                                      count = BATCH_SIZE records exactly
   BATCH_SIZE=20                      max records per message
   MSG_CHARS=3800                     message size budget (Telegram limit 4096)
   CONCURRENCY=8                      8 messages in parallel (order safe)
   DELAY=0                            per message gap (0 = full speed)
   PROGRESS_EVERY=10                  admin progress interval (seconds)
   STATE_SAVE_EVERY=10                state save interval (batches)
   START_OFFSET=0  END_OFFSET=0  STATE_FILE=state.json
================================================================================
"""

import os
import sys
import json
import time
import signal
import asyncio
import logging
import collections
import html as _html
from datetime import datetime, timezone

import httpx

# --------------------------------------------------------------------- logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telethon").setLevel(logging.WARNING)
log = logging.getLogger("exporter")


# ---------------------------------------------------------------- .env loader

def load_dotenv(path: str = ".env") -> None:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        target = next((p for p in (path, os.path.join(here, path)) if os.path.isfile(p)), None)
        if not target:
            return
        with open(target, "r", encoding="utf-8-sig") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name, value = name.strip(), value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                if name and name not in os.environ:
                    os.environ[name] = value
    except Exception:
        log.warning(".env read failed (ignored)")


load_dotenv()


def env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else str(value).strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(env(name, str(default)) or default)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(env(name, str(default)) or default)
    except ValueError:
        return default


# --------------------------------------------------------------------- config

HF_DATASET = env("HF_DATASET")
HF_TOKEN = env("HF_TOKEN")
HF_CONFIG = env("HF_CONFIG", "default") or "default"
HF_SPLIT = env("HF_SPLIT", "train") or "train"
HF_API = "https://datasets-server.huggingface.co"
HF_MAX = 100                       # HF /rows ka max per request

BOT_TOKEN = env("BOT_TOKEN") or env("TG_BOT_TOKEN")
CHANNEL_ID = env("CHANNEL_ID") or env("TG_CHANNEL") or env("TG_BOT_CHAT")
ADMIN_ID = env("ADMIN_ID") or env("TG_ADMIN_ID") or env("BOT_CHAT_ID")

API_ID = env_int("API_ID", env_int("TG_API_ID", 0))
API_HASH = env("API_HASH") or env("TG_API_HASH")

MODE_RAW = (env("MODE", "auto") or "auto").lower()
MODE = "bot" if MODE_RAW == "auto" else MODE_RAW

BATCH_MODE = (env("BATCH_MODE", "auto") or "auto").lower()
BATCH_SIZE = max(1, min(100, env_int("BATCH_SIZE", 20)))
MSG_CHARS = max(600, min(4000, env_int("MSG_CHARS", 3800)))
# itna bada record message me trim karke nahi bhejte -> .json file (data safe)
MAX_TRIM = max(4, env_int("MAX_TRIM_KB", 8)) * 1024

# ---- SPEED KNOBS (v8) ------------------------------------------------------
# KITNE MESSAGES ek hi account ko back-to-back bhejne se pehle next account?
#   1  = strict alternate (msg1->acc1, msg2->acc2 ...) — sabse slow
#   16 = burst mode: ek account 16 msg lagatar bhejta hai, phir agla.
#        Pattern phir bhi 1->2->3 hai, bas lehar (wave) chhoti nahi hoti.
#        Telegram ke per-bot ~20 msg/min limit se safely neeche rehta hai.
BURST = max(1, env_int("BURST", 1))
HF_PREFETCH = max(0, env_int("HF_PREFETCH", 6))     # aage ke kitne 100-row chunks pehle se
HF_PAGE = 100                                       # HF /rows max
# Bot API ka asli limit ~20 msg/min per bot. Isse upar jaate hi 429 (retry_after
# 1000s+ = 18 min barbaad). Sliding-window limiter limit ke NEECHE rakhta hai
# -> koi time waste nahi, throughput predictable. 0 = off.
RATE_PER_MIN = env_int("RATE_PER_MIN", 18)   # MODE=bot: 20/min per group ka safe margin
# Official: "single chat me 1 msg/sec se zyada mat bhejo" — ye limit CHAT ki hai, bot ki nahi.
# Saare accounts ek hi channel me bhejte hain -> poore export ka shared ceiling 60 msg/min.
# Ye throttle ussi ko enforce karta hai (0 = off).
CHAT_PER_SEC = env_float("CHAT_PER_SEC", 1.0)
DRY_LATENCY = env_float("DRY_RUN_LATENCY", 0.35)   # DRY_RUN me nakli send latency (benchmark)

PROGRESS_EVERY = env_float("PROGRESS_EVERY", 10.0)
STATE_SAVE_EVERY = env_int("STATE_SAVE_EVERY", 10)
START_OFFSET = env_int("START_OFFSET", 0)
END_OFFSET = env_int("END_OFFSET", 0)
STATE_FILE = env("STATE_FILE", "state.json") or "state.json"
MAX_RETRIES = env_int("MAX_RETRIES", 4)

# --------------------------------------------------------- multi-account pool
# 5 bots + 5 api_id/hash support:
#   ACCOUNTS=123:abc:TOKEN1, 456:def:TOKEN2, ...   (api_id:api_hash:bot_token)
#   ya lists: API_IDS=1,2,3  API_HASHES=a,b,c  BOT_TOKENS=t1,t2,t3
#   ya single: BOT_TOKEN= + API_ID= + API_HASH=

def _split_list(raw: str) -> list:
    items = []
    for part in (raw or "").replace("\n", ",").split(","):
        part = part.strip().strip("'\"").strip()
        if part:
            items.append(part)
    return items


def _classic_pairs() -> list:
    """
    Sabse common setup:
        API_ID=11        API_HASH=h1        BOT_TOKEN=111:AAA
        API_ID_2=22      API_HASH_2=h2      BOT_TOKEN_2=222:BBB
        ... _3, _4, _5 ...
    Har bot token apne hi api_id/hash ke saath pair hota hai (session = api_id+hash+token).
    Ye alag-alag cross-product nahi banata, isliye 13 accounts jaisa gadbad nahi hota.
    """
    import re as _re

    def group(regex_getter):
        found = {}
        for name, value in os.environ.items():
            if not value or not value.strip():
                continue
            match = _re.fullmatch(regex_getter, name)
            if not match:
                continue
            suffix = match.group(1) or "1"
            try:
                idx = int(suffix)
            except ValueError:
                idx = 1
            found[idx] = value.strip()
        return found

    ids = group(r"(?:API_ID|TG_API_ID)(?:_(\d+))?")
    hashes = group(r"(?:API_HASH|TG_API_HASH)(?:_(\d+))?")
    tokens = group(r"(?:BOT_TOKEN|TG_BOT_TOKEN)(?:_(\d+))?")

    # purane single naam (suffix nahi) ko index 1 maano
    pool = []
    for idx in sorted(set(list(ids) + list(hashes) + list(tokens))):
        api_id = ids.get(idx, 0)
        api_hash = hashes.get(idx, "")
        token = tokens.get(idx, "")
        try:
            api_id = int(api_id)
        except (TypeError, ValueError):
            api_id = 0
        if not token and not (api_id and api_hash):
            continue
        pool.append({"api_id": api_id, "api_hash": api_hash, "bot_token": token})
    return pool


def parse_pool() -> list:
    """Har account: {api_id, api_hash, bot_token}. Sab valid combos return karta hai."""
    pool = []

    # (0) classic pairing: API_ID_n / API_HASH_n / BOT_TOKEN_n  (same index)
    classic = _classic_pairs()
    if classic:
        return classic

    # (a) ACCOUNTS="api_id:api_hash:token"
    for entry in _split_list(env("ACCOUNTS")):
        parts = [p.strip() for p in entry.split(":")]
        api_id = api_hash = token = ""
        if len(parts) >= 3:
            api_id, api_hash, token = parts[0], parts[1], ":".join(parts[2:])
        elif len(parts) == 2:
            api_id, api_hash = parts[0], parts[1]
        elif len(parts) == 1:
            token = parts[0]
        try:
            api_id_int = int(api_id) if api_id else 0
        except ValueError:
            api_id_int = 0
        if api_id_int and api_hash:
            pool.append({"api_id": api_id_int, "api_hash": api_hash, "bot_token": token or BOT_TOKEN})
        elif token:
            pool.append({"api_id": 0, "api_hash": "", "bot_token": token})

    if pool:
        return pool

    # (b) lists  API_IDS / API_HASHES / BOT_TOKENS (index se zip)
    ids = _split_list(env("API_IDS"))
    hashes = _split_list(env("API_HASHES") or env("API_HASH"))
    tokens = _split_list(env("BOT_TOKENS") or env("BOT_TOKEN"))

    count = max(len(ids), len(hashes), len(tokens))
    for i in range(count):
        api_id = 0
        if i < len(ids):
            try:
                api_id = int(ids[i])
            except ValueError:
                api_id = 0
        # strict index pairing: token/ hash sirf apne index se
        api_hash = hashes[i] if i < len(hashes) else ""
        token = tokens[i] if i < len(tokens) else ""
        if api_id and api_hash:
            pool.append({"api_id": api_id, "api_hash": api_hash, "bot_token": token})
        elif token:
            pool.append({"api_id": 0, "api_hash": "", "bot_token": token})

    return pool


def dedupe_pool(pool: list) -> list:
    """
    SAME bot token ya SAME api_id+hash ko do jagah use karna = ek hi Telegram session
    do parallel senders me -> "Request was unsuccessful 6 time(s)" errors.
    Isliye duplicate hata dete hain (aur warn karte hain).
    """
    seen, out = set(), []
    for acc in pool:
        key = (acc["bot_token"] or "", acc["api_id"], acc["api_hash"] or "")
        token_key = acc["bot_token"] or ""
        if token_key and any(token_key == a["bot_token"] for a in out):
            log.warning("⚠️ duplicate BOT token mila — skip (wahi session parallel nahi chal sakta)")
            continue
        if not token_key and acc["api_id"] and any(
                a["api_id"] == acc["api_id"] and a["api_hash"] == acc["api_hash"] and not a["bot_token"]
                for a in out):
            log.warning("⚠️ duplicate api_id/api_hash mila (bina bot token) — skip")
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(acc)
    return out


POOL = parse_pool()

# duplicate accounts (same token / same session) hata do
_REDIRECT = []  # token -> [api_id, api_hash] (user mode me token aur id alag ho)
POOL = dedupe_pool(POOL)

# MODE=auto -> agar kisi account me api_id/hash hai to MTProto (fast), warna bot
if MODE_RAW == "auto":
    MODE = "user" if any(a["api_id"] and a["api_hash"] for a in POOL) else "bot"

# per-account parallel messages (total window = accounts x ye)
WAIT_FOR_ACCOUNT = env("WAIT_FOR_ACCOUNT", "1") not in ("0", "false", "no")
# MTProto logins ek saath na ho (FLOOD_WAIT se bachne ke liye) — chhota gap
LOGIN_STAGGER = env_float("LOGIN_STAGGER", 0.8)

# ---- Render web service ----
PORT = env_int("PORT", 0)                      # Render isko khud deta hai
WEB = env("WEB", "auto").lower()               # auto = PORT mile to server chalu
WEB_TOKEN = env("WEB_TOKEN")                   # control endpoints (/stop /start) ke liye
DRY_RUN = env("DRY_RUN", "0") in ("1", "true", "yes")   # Telegram ko chhode bina test
PER_ACCOUNT = env_int("PER_ACCOUNT_CONCURRENCY", env_int("CONCURRENCY", 4))
PER_ACCOUNT = max(1, min(16, PER_ACCOUNT))

if MODE == "user":
    DELAY = env_float("DELAY", 0.0)
else:
    DELAY = env_float("DELAY", 3.0)

DELAY = max(0.0, DELAY)
ACCOUNTS_N = max(1, len(POOL))
# total in-flight messages = accounts x per-account (max 32)
CONCURRENCY = max(1, min(512, PER_ACCOUNT * ACCOUNTS_N))



def as_chat_id(value: str):
    value = (value or "").strip()
    if not value:
        return None
    if value.lstrip("-").isdigit():
        return int(value)
    return value


# ---------------------------------------------------------------------- state

DEFAULT_STATE = {
    "dataset": "", "config": "", "split": "",
    "offset": 0, "sent": 0, "messages": 0, "total": 0,
    "finished": False, "started_at": "", "updated_at": "", "last_error": "",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state() -> dict:
    state = dict(DEFAULT_STATE)
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        if isinstance(saved, dict):
            state.update(saved)
        log.info("STATE loaded -> offset=%s sent=%s messages=%s finished=%s",
                 state["offset"], state["sent"], state["messages"], state["finished"])
    except FileNotFoundError:
        log.info("STATE nahi mila -> fresh start (offset %s)", START_OFFSET)
        state["offset"] = START_OFFSET
    except Exception as exc:
        log.warning("STATE read failed (%s) -> fresh start", exc)

    if state.get("dataset") and state["dataset"] != HF_DATASET:
        log.warning("STATE dusre dataset ka hai -> reset")
        state = dict(DEFAULT_STATE)
        state["offset"] = START_OFFSET

    state["dataset"] = HF_DATASET
    state["config"] = HF_CONFIG
    state["split"] = HF_SPLIT
    if not state.get("started_at"):
        state["started_at"] = now_iso()
    if START_OFFSET and not state.get("offset"):
        state["offset"] = START_OFFSET
    return state


def save_state(state: dict) -> None:
    state["updated_at"] = now_iso()
    tmp = STATE_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, STATE_FILE)
    except Exception as exc:
        log.error("STATE save failed: %s", exc)


# ------------------------------------------------------------------ HF reader

def hf_headers() -> dict:
    headers = {"User-Agent": "Dataset-TG-Exporter/4.0"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return headers


class HfReader:
    """100-row chunk + background prefetch (fetch kabhi bhi bottleneck nahi)."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.total = 0
        self.chunks: dict = {}
        self._prefetch: dict = {}

    async def fetch(self, offset: int, length: int) -> dict:
        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await self.client.get(
                    f"{HF_API}/rows",
                    params={
                        "dataset": HF_DATASET,
                        "config": HF_CONFIG,
                        "split": HF_SPLIT,
                        "offset": offset,
                        "length": min(HF_MAX, max(1, length)),
                    },
                    timeout=45.0,
                )
                if response.status_code == 429:
                    await asyncio.sleep(min(30, 3 * attempt))
                    last_error = "HTTP 429"
                    continue
                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}"
                    await asyncio.sleep(min(20, 2 * attempt))
                    continue
                data = response.json()
                if data.get("error"):
                    last_error = str(data["error"])
                    await asyncio.sleep(min(20, 2 * attempt))
                    continue
                return data
            except Exception as exc:
                last_error = str(exc)
                await asyncio.sleep(min(20, 2 * attempt))
        raise RuntimeError(f"HF rows fetch failed: {last_error}")

    async def _load_chunk(self, start: int) -> list:
        if start in self.chunks:
            return self.chunks[start]
        data = await self.fetch(start, HF_MAX)
        self.total = int(data.get("num_rows_total") or self.total or 0)
        records = []
        for i, item in enumerate(data.get("rows", []) or []):
            row = item.get("row") if isinstance(item, dict) else None
            if not isinstance(row, dict):
                row = {"value": item}
            record = clean(row)
            record["_index"] = item.get("row_idx", start + i) if isinstance(item, dict) else start + i
            records.append(record)
        self.chunks[start] = records
        for key in [k for k in self.chunks if k < start - HF_MAX * 2]:
            self.chunks.pop(key, None)
        return records

    def prefetch(self, offset: int) -> None:
        """v8 TRICK: aage ke HF_PREFETCH chunks EK SAATH (parallel) maango.
        Pehle ek-ek karke aata tha (6 x 0.2s = 1.2s stall). Ab ~0.25s me 600 rows."""
        start = offset - (offset % HF_MAX)
        for step in range(1, HF_PREFETCH + 1):
            nxt = start + HF_MAX * step
            if self.total and nxt >= self.total:
                break
            if nxt in self.chunks:
                continue
            task = self._prefetch.get(nxt)
            if task and not task.done():
                continue
            self._prefetch[nxt] = asyncio.create_task(self._safe_load(nxt))

    async def _safe_load(self, start: int) -> None:
        try:
            await self._load_chunk(start)
        except Exception as exc:
            log.debug("prefetch fail (%s) — ignore", exc)

    async def get(self, offset: int):
        start = offset - (offset % HF_MAX)
        if start not in self.chunks:
            await self._load_chunk(start)
            log.info("HF chunk %s ready (total %s)", start, self.total)
        records = self.chunks[start]
        position = offset - start
        record = records[position] if 0 <= position < len(records) else None
        self.prefetch(offset)
        return record, self.total


def clean(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return str(value)


# ------------------------------------------------------------------ messages --

def _truncated(record: dict, limit: int) -> bool:
    """Sach me kuch trim hua? (nahi hua to 'trimmed' label galat hoga)"""
    def walked(value):
        if isinstance(value, str):
            return len(value) > limit
        if isinstance(value, dict):
            return any(walked(v) for v in value.values())
        if isinstance(value, list):
            return any(walked(v) for v in value)
        return False
    return walked(record)


def _shrink(record: dict, limit: int = 1200) -> dict:
    def cut(value):
        if isinstance(value, str):
            return value if len(value) <= limit else value[:limit] + f"…[+{len(value) - limit} chars]"
        if isinstance(value, dict):
            return {k: cut(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cut(v) for v in value]
        return value
    return {k: cut(v) for k, v in record.items()}


def html_escape(text: str) -> str:
    return _html.escape(str(text), quote=False)


def _pretty_block(record: dict, step: int = 2) -> str:
    """Record ka pretty JSON (indent 2), array ke andar 2 space indent ke saath."""
    dumped = json.dumps(record, ensure_ascii=False, indent=step)
    return "\n".join("  " + line for line in dumped.split("\n"))


def _header(start: int, count: int, total: int) -> str:
    if count <= 1:
        return f"#{start + 1}/{total or '?'}"
    last = start + count                      # 1-based: 0..5 -> "#1–6"
    return f"#{start + 1}–{last}/{total or '?'}"


PRE_OPEN = '<pre><code class="language-json">'
PRE_CLOSE = "</code></pre>"


def _render(header: str, body: str) -> str:
    """Telegram HTML: header + code block (pretty JSON andar)."""
    return (
        html_escape(header)
        + "\n"
        + PRE_OPEN
        + html_escape(body)
        + PRE_CLOSE
    )


def build_batch(items: list, total: int):
    """
    items: [(offset, record), ...]
    Ek message me JITNE RECORDS PRETTY-PRINT (indent 2) me fit hote hain utne.
    Returns (text, caption, file_or_None, count).
    """
    start = items[0][0]
    # header ki worst-case length reserve (count ke digits badh sakte hain)
    reserve = len(html_escape(_header(start, BATCH_SIZE, total))) + 2 \
        + len(PRE_OPEN) + len(PRE_CLOSE)
    budget = MSG_CHARS - reserve

    blocks = [_pretty_block(record) for _, record in items[:BATCH_SIZE]]

    # ---- array / object me kitne fit honge (pretty, indent 2) ----
    chosen = 0
    used = len(html_escape("[\n\n]"))          # brackets
    for block in blocks:
        add = len(html_escape(block)) + (2 if chosen else 0)   # ",\n"
        if used + add > budget:
            break
        used += add
        chosen += 1

    if chosen >= 2:
        body = "[\n" + ",\n".join(blocks[:chosen]) + "\n]"
        header = _header(start, chosen, total)
        text = _render(header, body)
        if len(text) <= MSG_CHARS:
            return text, header, None, chosen

    # ---- ek hi record (object form, pretty) ----
    if chosen >= 1:
        offset, record = items[0]
        body = json.dumps(record, ensure_ascii=False, indent=2)
        header = _header(offset, 1, total)
        text = _render(header, body)
        if len(text) <= MSG_CHARS:
            return text, header, None, 1

    # ---- ek single record jo message me fit nahi: sirf chhoti trim (<=100 chars) ----
    # zyada trim = data barbaad, isliye limit ke hisaab se chhota cut, warna file
    offset, record = items[0]
    if len(_pretty_block(record)) > MAX_TRIM:
        limit = 400
        body = json.dumps(_shrink(record, limit), ensure_ascii=False, indent=2)
        header = _header(offset, 1, total)
        text = _render(header, body)
        if len(text) <= MSG_CHARS and not _truncated(record, limit):
            return text, header + " (trimmed)", None, 1

    # ---- last resort: poora pretty JSON file ke roop me ----
    payload = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
    header = _header(offset, 1, total)
    return (
        f"{header} — bada record, .json file me bheja",
        f"{header} (bada record -> .json file)",
        (f"record_{offset}.json", payload),
        1,
    )


class FloodWait(Exception):
    def __init__(self, seconds: int, detail=""):
        super().__init__(f"FloodWait {seconds}s")
        self.seconds = max(1, int(seconds))
        self.detail = detail


class SendGate:
    """Per-account flood control: ek account flood-wait me -> sirf wahi rukta hai."""

    def __init__(self, label: str = ""):
        self.label = label
        self.pause_until = 0.0
        self.lock = asyncio.Lock()
        # v8: ek hi account ke messages EK SAATH nahi jate (order + 429 safety),
        # par har account apni alag "lane" me chalta hai -> 13 accounts = 13 parallel lanes
        self.lane = asyncio.Semaphore(1)   # size AccountPool set karta hai
        self.sent = 0
        self.errors = 0
        self.hits = collections.deque()    # rate limiter: pichle 60s ke send timestamps

    async def wait(self):
        while True:
            left = self.pause_until - time.monotonic()
            if left <= 0:
                break
            await asyncio.sleep(min(left, 5.0))
        if not RATE_PER_MIN:
            return
        while True:                       # sliding 60s window
            now = time.monotonic()
            while self.hits and now - self.hits[0] > 60.0:
                self.hits.popleft()
            if len(self.hits) < RATE_PER_MIN:
                self.hits.append(now)
                return
            await asyncio.sleep(max(0.05, 60.0 - (now - self.hits[0])))

    async def pause(self, seconds: int):
        async with self.lock:
            target = time.monotonic() + min(seconds, 900) + 1
            self.pause_until = max(self.pause_until, target)
        log.warning("⏳ [%s] flood-wait %ss -> ye account %ss ruk gaya",
                    self.label, seconds, seconds)


class AccountPool:
    """
    STRICT ALTERNATE: message 1 -> acc1, 2 -> acc2, 3 -> acc3, ... 6 -> acc6,
    7 -> acc1 (phir se). Har record ka account pehle se tay hota hai (deterministic),
    isliye koi record miss/mismatch nahi hota.

    v8: har account ki apni "lane" (PER_ACCOUNT_CONCURRENCY msg parallel) hai —
    isliye 13 accounts ek saath chalte hain aur pattern bhi 1->2->...->13->1 rehta hai.

    Agar kisi account ka flood-wait chal raha ho:
      • WAIT_FOR_ACCOUNT=1 (default) -> usi ka number ban raha hai to uska wait hoga
        (order aur alternation dono intact, dhakka-mukki nahi)
      • WAIT_FOR_ACCOUNT=0 -> agla free account (speed zyada, pattern thoda tootega)
    """

    def __init__(self, accounts: list, wait_for_account: bool = True):
        self.accounts = accounts                     # [(label, sender)]
        per = max(1, env_int("PER_ACCOUNT_CONCURRENCY", 4))
        self.gates = {}
        for label, _ in accounts:
            gate = SendGate(label)
            gate.lane = asyncio.Semaphore(per)      # ek account X msg parallel (order server par)
            self.gates[label] = gate
        self.senders = {label: sender for label, sender in accounts}
        self.wait_for_account = wait_for_account
        self._assignment = 0                         # record-message number (0-based)
        self._admin_i = 0                            # admin/progress ke liye alag

    def __len__(self):
        return len(self.accounts)

    @property
    def labels(self) -> list:
        return [label for label, _ in self.accounts]

    def live_labels(self, horizon: int = 0) -> list:
        """Flood-wait me nahi hain aise accounts (agla wave inhi ko bhejega)."""
        now = time.monotonic()
        return [l for l, _ in self.accounts if self.gates[l].pause_until <= now]

    def assign(self):
        """Strict number-wise account (no await -> order safe)."""
        idx = self._assignment % len(self.accounts)
        self._assignment += 1
        label, sender = self.accounts[idx]
        return label, sender, self.gates[label], idx

    def assign_admin(self):
        """Admin/progress messages — record pattern ko disturb nahi karte."""
        label, sender = self.accounts[self._admin_i % len(self.accounts)]
        self._admin_i += 1
        return label, sender, self.gates[label], self._admin_i % len(self.accounts)

    def free_account(self, skip_idx: int):
        """Flood me ho aur wait na karna ho to agla free account."""
        for offset in range(1, len(self.accounts)):
            idx = (skip_idx + offset) % len(self.accounts)
            label, sender = self.accounts[idx]
            if self.gates[label].pause_until <= time.monotonic():
                return label, sender, self.gates[label], idx
        return None

    def stats(self) -> str:
        return " · ".join(f"{label}:{self.gates[label].sent}" for label in self.labels)

    def flood_list(self) -> list:
        now = time.monotonic()
        return [
            f"{label}({int(self.gates[label].pause_until - now)}s)"
            for label in self.labels
            if self.gates[label].pause_until > now
        ]

    async def close(self):
        for label, sender in self.accounts:
            try:
                await sender.close()
            except Exception:
                pass


class BotAccount:
    """Ek bot token = ek account (apna flood window). Pool inhe round-robin use karta hai."""

    def __init__(self, token: str, channel, admin, label: str = "bot"):
        self.base = f"https://api.telegram.org/bot{token}"
        self.channel = channel
        self.admin = admin
        self.label = label

    async def _call(self, client, method: str, payload: dict):
        response = await client.post(f"{self.base}/{method}", json=payload, timeout=40.0)
        try:
            data = response.json()
        except Exception:
            raise RuntimeError(f"Telegram invalid response: {response.text[:150]}")
        if data.get("ok"):
            return data
        retry_after = (data.get("parameters") or {}).get("retry_after")
        if retry_after or data.get("error_code") == 429:
            raise FloodWait(int(retry_after or 30), str(data.get("description")))
        raise RuntimeError(f"Telegram {data.get('error_code')}: {data.get('description')}")

    async def send_message(self, client, chat, text: str, parse_mode: str = "HTML") -> None:
        payload = {
            "chat_id": str(chat),
            "text": text,
            "disable_web_page_preview": True,
            "disable_notification": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        await self._call(client, "sendMessage", payload)

    async def send_record(self, client, text: str, file=None, caption: str = None) -> None:
        if file:
            files = {"document": (file[0], file[1], "application/json")}
            payload = {"chat_id": str(self.channel), "caption": (caption or text)[:1000],
                       "parse_mode": "HTML",
                       "disable_notification": "true"}
            response = await client.post(f"{self.base}/sendDocument",
                                         data=payload, files=files, timeout=60.0)
            body = response.json()
            if not body.get("ok"):
                retry_after = (body.get("parameters") or {}).get("retry_after")
                if retry_after or body.get("error_code") == 429:
                    raise FloodWait(int(retry_after or 30), str(body.get("description")))
                raise RuntimeError(f"Telegram {body.get('error_code')}: {body.get('description')}")
        else:
            await self.send_message(client, self.channel, text)

    async def send_admin(self, client, text: str) -> None:
        if self.admin is not None:
            await self.send_message(client, self.admin, text)


class MtprotoAccount:
    """Ek api_id/api_hash (+ bot token) = ek account."""

    def __init__(self, api_id: int, api_hash: str, bot_token: str, channel, admin,
                 label: str = "mtproto"):
        from telethon import TelegramClient
        from telethon.sessions import StringSession

        # receive_updates=False -> GetChannelDifference/update loop band
        #   (PersistentTimestampOutdatedError aana band ho jata hai)
        # request_retries=1 -> "Request was unsuccessful 6 time(s)" ki jagah asli error
        self.client = TelegramClient(
            StringSession(), api_id, api_hash,
            receive_updates=False,
            request_retries=1,
            connection_retries=5,
            retry_delay=2,
            timeout=30,
            flood_sleep_threshold=60,
            auto_reconnect=True,
        )
        self.bot_token = bot_token
        self.channel = channel
        self.admin = admin
        self.label = label

    async def start(self):
        if self.bot_token:
            await self.client.start(bot_token=self.bot_token)
        else:
            await self.client.start()
        me = await self.client.get_me()
        # channel/peer pehle resolve karo -> send me "Could not find the input entity" nahi aayega
        try:
            peer = await self.client.get_input_entity(self.channel)
            log.info("%s connected as @%s (id %s) · channel resolve ✓",
                     self.label, getattr(me, "username", None), me.id)
        except Exception as exc:
            log.warning("%s: channel resolve fail (%s) — check karo bot channel me admin hai",
                        self.label, exc)

    async def _send(self, chat, text=None, file=None):
        from telethon.errors import FloodWaitError, SlowModeWaitError
        try:
            if file:
                from io import BytesIO
                from telethon.tl.types import DocumentAttributeFilename
                await self.client.send_file(
                    chat, BytesIO(file[1]),
                    caption=(text or "")[:900],
                    force_document=True,
                    attributes=[DocumentAttributeFilename(file[0])],
                )
            else:
                await asyncio.wait_for(
                    self.client.send_message(chat, text, link_preview=False, parse_mode="html"),
                    timeout=60,
                )
        except asyncio.TimeoutError:
            raise RuntimeError("send timeout (60s) — network/DC issue")
        except (FloodWaitError, SlowModeWaitError) as exc:
            raise FloodWait(getattr(exc, "seconds", 30), str(exc))

    async def send_record(self, text: str, file=None) -> None:
        await self._send(self.channel, text, file)

    async def send_admin(self, text: str) -> None:
        if self.admin is not None:
            await self._send(self.admin, text)

    async def close(self):
        try:
            await self.client.disconnect()
        except Exception:
            pass


# -------------------------------------------------------------------- progress

STATS_HOLDER: dict = {}

def progress_text(state: dict, elapsed: float, speed: float) -> str:
    total = int(state.get("total") or 0)
    sent = int(state.get("sent") or 0)
    offset = int(state.get("offset") or 0)
    messages = int(state.get("messages") or 0)
    remaining = max(0, total - offset) if total else None
    pct = (offset / total * 100) if total else 0.0
    eta_min = (remaining / speed / 60) if (speed > 0 and remaining) else 0
    msg_speed = (messages / elapsed) if elapsed > 0 else 0.0

    if total and remaining == 0:
        tail = "🎉 Poora ho gaya (100%)"
    elif remaining is None:
        tail = "🕒 ETA: total pata chalte hi"
    else:
        tail = f"🕒 ETA: {eta_min:.1f} min"

    return "\n".join([
        "📊 EXPORT PROGRESS",
        f"🗂 {HF_DATASET} · {HF_CONFIG}/{HF_SPLIT}",
        f"✅ Records sent: {sent}",
        f"📨 Messages: {messages} ({msg_speed:.2f}/s · {msg_speed * 60:.0f}/min)",
        f"📈 Position: {offset}/{total or '?'} ({pct:.1f}%)",
        f"⏳ Baaki: {remaining if remaining is not None else '?'}",
        f"⚡ Speed: {speed:.1f} rec/s · {speed * 60:.0f} rec/min · ⏱ {elapsed / 60:.1f} min",
        tail,
        f"🔧 {MODE} · {BATCH_MODE} batch(max {BATCH_SIZE}) · in-flight {CONCURRENCY} · delay {DELAY}s",
    ] + (
        [f"👥 Accounts: {STATS_HOLDER['pool'].stats()}"]
        + ([f"⏳ Flood: {' '.join(STATS_HOLDER['pool'].flood_list())}"]
           if STATS_HOLDER["pool"].flood_list() else [])
        if STATS_HOLDER.get("pool") else []
    ))


# ------------------------------------------------------------------------ run

async def run() -> None:
    problems = []
    if not HF_DATASET:
        problems.append("HF_DATASET missing (username/dataset)")
    if not CHANNEL_ID:
        problems.append("CHANNEL_ID missing (-100... private channel)")
    if not POOL:
        problems.append("koi account nahi mila — BOT_TOKEN ya API_IDS/API_HASHES/BOT_TOKENS "
                        "ya ACCOUNTS=api_id:api_hash:token")
    if problems:
        for item in problems:
            log.error("❌ %s", item)
        return

    global STOP_EVENT, CONCURRENCY

    channel = as_chat_id(CHANNEL_ID)
    admin = as_chat_id(ADMIN_ID) if ADMIN_ID else None

    log.info("MODE=%s | accounts=%s | batch=%s(max %s, %s chars) | in-flight=%s | delay=%ss",
             MODE, len(POOL), BATCH_MODE, BATCH_SIZE, MSG_CHARS, CONCURRENCY, DELAY)
    if MODE == "bot":
        log.info("ℹ️  MODE=bot → api_id/api_hash Telegram ko NAHI jaate (Bot API me unki "
                 "zaroorat hi nahi). Speed sirf unique bot tokens se aati hai. "
                 "Per-bot limit: 1 msg/sec per chat, 20/min per group.")
        if RATE_PER_MIN == 18:
            log.info("   RATE_PER_MIN=18 (Bot API default). MTProto chahiye to MODE=user karo.")
    else:
        log.warning("⚠️  MODE=user → api_id/api_hash session banane ke liye use ho rahe hain "
                    "(MTProto). Ye as a BOT login karta hai (bot token ke saath), "
                    "isliye bot ke rights/limit hi rehte hain — sirf transport badalta hai.")
        log.info("   MTProto + per-chat 1 msg/sec = shared channel par ~60 msg/min. "
                 "Bot API (MODE=bot) me per-bot 20/min = 13 bots par ~260 msg/min.")

    if STOP_EVENT is None:
        STOP_EVENT = asyncio.Event()
    stop = STOP_EVENT
    state = load_state()

    def _signal(*_):
        log.warning("🛑 stop signal -> pending messages ke baad rukunga")
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            asyncio.get_event_loop().add_signal_handler(sig, _signal)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, _signal)

    # ---------------- accounts setup (1..N) ----------------
    accounts = []
    http_client = None

    async def login_mtproto(idx, acc, label):
        """Ek MTProto account login (parallel chalta hai)."""
        api_id, api_hash, token = acc["api_id"], acc["api_hash"], acc["bot_token"]
        if not (api_id and api_hash):
            log.warning("%s skip — api_id/api_hash missing", label)
            return None
        if not token:
            log.warning("%s skip — BOT_TOKEN nahi (MTProto login ke liye bot token chahiye; "
                        "ya MODE=bot karo)", label)
            return None
        acc_obj = MtprotoAccount(api_id, api_hash, token, channel, admin, label=label)
        try:
            await acc_obj.start()
            return label, acc_obj
        except Exception as exc:
            msg = str(exc) or exc.__class__.__name__
            log.error("❌ %s login FAIL: %s", label, msg[:200])
            low = msg.lower()
            if "api_id" in low or "api_hash" in low:
                log.error("   → api_id/api_hash pair GALAT hai (my.telegram.org se dobara lo). "
                          "Ye account skip, baaki chalenge.")
            elif any(k in low for k in ("unauthorized", "token")):
                log.error("   → bot token galat/revoked hai (BotFather se dobara lo)")
            elif any(k in low for k in ("auth_key", "session", "revoked")):
                log.error("   → session problem: naya string session / token check karo")
            elif any(k in low for k in ("connect", "timeout", "network")):
                log.error("   → network: Render ka outbound / DC reachable nahi")
            JOB["last_error"] = f"{label}: {msg[:160]}"
            try:
                await acc_obj.close()
            except Exception:
                pass
            return None

    if MODE == "user":
        tasks = []
        for idx, acc in enumerate(POOL, start=1):
            label = f"acc{idx}"
            tasks.append(asyncio.create_task(login_mtproto(idx, acc, label)))
            if LOGIN_STAGGER > 0:
                await asyncio.sleep(min(LOGIN_STAGGER, 0.15))   # sirf socket burst bachane ke liye
        log.info("🔐 %s MTProto account parallel login ho rahe hain…", len(tasks))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        accounts = [r for r in results if isinstance(r, tuple)]
    else:
        for idx, acc in enumerate(POOL, start=1):
            label = f"acc{idx}"
            token = acc["bot_token"]
            if not token:
                log.warning("%s skip — bot_token missing", label)
                continue
            if http_client is None:
                http_client = httpx.AsyncClient(
                    timeout=60.0,
                    limits=httpx.Limits(max_connections=64, max_keepalive_connections=32,
                                        keepalive_expiry=300.0),
                    http2=False,
                )
            accounts.append((label, BotAccount(token, channel, admin, label=label)))
    if not accounts:
        JOB["last_error"] = "koi account login nahi ho paya (BOT_TOKEN/API_ID/API_HASH check karo)"
        log.error("❌ koi valid account nahi mila — Render env me BOT_TOKEN / API_ID+API_HASH check karo. "
                  "Web service phir bhi chalu rahegi (/health, /status).")
        return

    pool = AccountPool(accounts, wait_for_account=WAIT_FOR_ACCOUNT)
    STATS_HOLDER["pool"] = pool

    # actual in-flight = per-account × valid accounts (skip hue accounts count na hon)
    CONCURRENCY = max(1, min(512, PER_ACCOUNT * len(pool)))
    pattern = (" → ".join(pool.labels + [pool.labels[0]]) if len(pool) > 1 else pool.labels[0])
    log.info("ACCOUNTS: %s (total %s, in-flight %s) | pattern: %s | flood par: %s",
             ", ".join(pool.labels), len(pool), CONCURRENCY, pattern,
             "wait (order safe)" if WAIT_FOR_ACCOUNT else "agla free account")

    # ---- SHARED CHAT throttle: sab accounts ek hi channel me bhejte hain ----
    # Official: "single chat me 1 msg/sec se zyada nahi" — limit CHAT ki hai.
    # Isliye 13 bots ka combined ceiling ~60 msg/min hai, 13x20 nahi.
    chat_lock = asyncio.Lock()
    chat_next = [0.0]

    async def chat_gate():
        if CHAT_PER_SEC <= 0:
            return
        gap = 1.0 / CHAT_PER_SEC
        async with chat_lock:
            now = time.monotonic()
            if chat_next[0] > now:
                await asyncio.sleep(chat_next[0] - now)
                now = time.monotonic()
            chat_next[0] = now + gap

    async def raw_send(label, text, file=None, caption=None):
        if DRY_RUN:
            # DRY_RUN me bhi realistic timing (0.35s/msg) taaki speed naap sako
            await asyncio.sleep(DRY_LATENCY)
            return
        sender = pool.senders[label]
        gate = pool.gates[label]
        await chat_gate()                     # channel-level 1 msg/sec (official rule)
        async with gate.lane:                 # ek account ke msg order me, par dusre accounts parallel
            if isinstance(sender, MtprotoAccount):
                await sender.send_record(text, file)
            else:
                await sender.send_record(http_client, text, file, caption)

    async def send_admin(text):
        if DRY_RUN:
            log.info("[DRY_RUN admin] %s", text.split("\n")[0])
            return
        label, sender, gate, _idx = pool.assign_admin()
        try:
            if isinstance(sender, MtprotoAccount):
                await sender.send_admin(text)
            else:
                await sender.send_admin(http_client, text)
        except Exception as exc:
            log.warning("admin message fail (%s)", exc)

    async def close_sender():
        await pool.close()
        if http_client is not None:
            await http_client.aclose()

    async def send_one(text, file=None, caption=None, slot=None):
        """
        slot = is message ka account number (strict alternate). Flood-wait par:
          WAIT_FOR_ACCOUNT=1 -> usi slot ka intezaar (order + pattern intact)
          WAIT_FOR_ACCOUNT=0 -> agla free account (fast, pattern thoda tootega)
        Record kabhi skip nahi hota — fail hone par exception (offset aage nahi badhta).
        """
        if slot is None:
            label, sender, gate, idx = pool.assign()
        else:
            label, sender, gate, idx = slot

        attempts = 0
        flood_hits = 0
        while True:
            # v8: agar is account ka flood-wait chal raha hai to bhejne se PEHLE hi
            # agla zinda account le lo (3s/18min wait barbaad na ho). Record safe.
            if gate.pause_until > time.monotonic() and len(pool) > 1:
                alt = pool.free_account(idx)
                if alt:
                    log.debug("[%s] flood me hai -> pre-switch [%s]", label, alt[0][:0] or alt[0])
                    label, sender, gate, idx = alt
            await gate.wait()
            try:
                await raw_send(label, text, file, caption)
                gate.sent += 1
                return label
            except FloodWait as exc:
                gate.errors += 1
                flood_hits += 1
                await gate.pause(exc.seconds)
                # v8: 3 flood/ya 5-min se lamba flood = ye account bekaar ho gaya.
                # Us message ko zinda account par shift karo (record bachta hai,
                # warna 18 min tak poora export ruka rehta tha).
                want_switch = (not pool.wait_for_account) or flood_hits >= 3 or exc.seconds > 300
                if want_switch:
                    alt = pool.free_account(idx)
                    if alt:
                        old_label = label
                        label, sender, gate, idx = alt
                        flood_hits = 0
                        log.warning("↪️ [%s] flood (%ss) -> ye message [%s] se bhej raha hoon "
                                    "(record safe, order safe)", old_label, exc.seconds, label)
                    elif flood_hits > MAX_RETRIES * 4:
                        raise RuntimeError(f"[{label}] flood-wait khatam nahi ho raha ({exc.seconds}s)")
            except Exception as exc:
                gate.errors += 1
                attempts += 1
                msg = str(exc) or exc.__class__.__name__
                if "unsuccessful" in msg or "timeout" in msg.lower():
                    # network/DC issue -> us account ko 5s rest, dusra account try
                    log.warning("[%s] connection issue (%s) -> account 5s rest", label, msg)
                    gate.pause(5)
                if attempts >= MAX_RETRIES * max(1, len(pool)):
                    raise RuntimeError(f"[{label}] {msg}")
                wait = min(15, 1.5 ** min(attempts, 5))
                log.warning("[%s] send error (%s) -> %.1fs baad retry", label, exc, wait)
                await asyncio.sleep(wait)

    hf_client = httpx.AsyncClient(headers=hf_headers(), timeout=60.0)
    reader = HfReader(hf_client)

    started_at = time.monotonic()
    last_progress = 0.0
    session_sent = 0
    session_msgs = 0
    saved = 0
    started_offset = int(state["offset"])

    async def push_progress(force: bool = False):
        nonlocal last_progress
        now = time.monotonic()
        if not force and (now - last_progress) < PROGRESS_EVERY:
            return
        last_progress = now
        elapsed = now - started_at
        try:
            await send_admin(progress_text(state, elapsed,
                                          (session_sent / elapsed) if elapsed else 0.0))
        except Exception as exc:
            log.warning("progress send fail: %s", exc)

    def commit(start_offset: int, count: int):
        nonlocal session_sent, session_msgs, saved
        state["offset"] = start_offset + count
        state["sent"] = int(state.get("sent") or 0) + count
        state["messages"] = int(state.get("messages") or 0) + 1
        session_sent += count
        session_msgs += 1
        saved += 1

    async def collect_items(start_offset):
        items = []
        for i in range(BATCH_SIZE):
            try:
                record, total = await reader.get(start_offset + i)
            except Exception as exc:
                raise RuntimeError(f"HF read fail: {exc}") from exc
            if total:
                state["total"] = total
            if record is None:
                break
            items.append((start_offset + i, record))
        return items

    # ================== WAVE ENGINE (v8 - asli speed yahin hai) ==================
    # PEHLE (v7): har message ke baad "await window.popleft()" -> pipeline block.
    #             Ek 0.5-5s wala send = baaki 12 accounts idle. 13 tokens hone par bhi slow.
    # AB (v8):    ek "wave" = (live accounts x BURST) messages, sab PARALLEL dispatch,
    #             har account ki apni lane (SendGate.lane) -> uske andar order safe.
    #             13 accounts = 13 messages ek saath = 13x throughput.
    #             Commits strict order me hote hain, isliye record kabhi miss nahi hota.
    wave_span = max(1, BURST * len(pool))
    log.info("⚡ SPEED MODE: %s accounts x BURST %s · dispatch depth %s msg · per-account lane 1 (order safe)",
             len(pool), BURST, CONCURRENCY)

    try:
        await reader.get(int(state["offset"]))     # HF prefetch garam karo
    except Exception:
        pass

    # ------------------------------------------------------------------
    # v8 PIPELINE: dispatch bharta rehta hai, commit FIFO order me hota hai.
    #  • Ek account flood-wait me chala jaye -> BAQI 12 accounts rukte NAHI
    #    (pehle v7 me poora pipeline usi ek account par atka rehta tha = "slow 🦥")
    #  • Commit sirf contiguous prefix (offset, offset+count, ...) par hota hai
    #    -> record kabhi skip/mismatch nahi hota, state hamesha sahi.
    # ------------------------------------------------------------------
    pending: collections.deque = collections.deque()   # [(task, offset, count)]
    committed_upto = int(state["offset"])
    ahead_limit = max(CONCURRENCY, wave_span * 2)      # kitne msg aage tak dispatch

    async def dispatch_one(offset: int) -> bool:
        """Ek message banao + bhejo. False = dataset khatam."""
        if END_OFFSET and offset >= END_OFFSET:
            return False
        try:
            items = await collect_items(offset)
        except Exception as exc:
            state["last_error"] = str(exc)
            save_state(state)
            log.error("%s -> 20s baad retry", exc)
            await asyncio.sleep(20)
            return True                                 # retry
        if not items:
            return False
        text, caption, file, count = build_batch(items, int(state["total"]))
        task = asyncio.create_task(send_one(text, file, caption, pool.assign()))
        pending.append((task, offset, count))
        return True

    failure = None
    planned = committed_upto
    try:
        while not stop.is_set() and not state.get("finished"):
            # ---- 1) dispatch: depth bharo (slots strict order me assign hote hain) ----
            while len(pending) < ahead_limit and (planned - committed_upto) < ahead_limit * BATCH_SIZE:
                if not await dispatch_one(planned):
                    break
                planned += pending[-1][2]
            # ---- agar kuch bhi pending nahi aur aage data nahi -> khatam ----
            if not pending:
                if int(state["total"]) and committed_upto >= int(state["total"]):
                    state["finished"] = True
                elif planned >= (END_OFFSET or 0) > 0:
                    break
                else:
                    state["finished"] = True
                break
            # ---- 2) sabse purane message ka intezaar, phir commit ----
            task, off, cnt = pending[0]
            try:
                await task
            except Exception as exc:
                failure = exc
                break
            pending.popleft()
            commit(off, cnt)
            committed_upto = int(state["offset"])
            if saved >= STATE_SAVE_EVERY:
                save_state(state)
                saved = 0
            await push_progress()
            if DELAY:
                await asyncio.sleep(DELAY)

        # ---- baaki pending flush (order me) ----
        while pending and not failure:
            task, off, cnt = pending.popleft()
            try:
                await task
            except Exception as exc:
                failure = exc
                break
            commit(off, cnt)

        if failure:
            state["last_error"] = str(failure)
            save_state(state)
            log.error("❌ send fail: %s — offset %s par saved (resume possible)",
                      failure, state["offset"])
            try:
                await send_admin(f"⚠️ Error: {failure}\n📍 offset {state['offset']} par ruka — "
                                 f"restart karne par yahin se chalega.")
            except Exception:
                pass

        if state["total"] and state["offset"] >= int(state["total"]):
            state["finished"] = True

    finally:
        save_state(state)
        elapsed = time.monotonic() - started_at
        speed = (session_sent / elapsed) if elapsed > 0 else 0.0
        try:
            await send_admin("⏹ STOPPED\n" + progress_text(state, elapsed, speed) +
                             f"\n💾 state: {STATE_FILE}")
        except Exception:
            pass
        await close_sender()
        await hf_client.aclose()
        log.info("STOP · sent %s · messages(is session) %s · offset %s · %.1f rec/s",
                 state.get("sent"), session_msgs, state.get("offset"), speed)


# ==============================================================================
#  RENDER WEB SERVICE  (health check + dashboard + start/stop control)
# ==============================================================================

STOP_EVENT: "asyncio.Event | None" = None
JOB: dict = {"running": False, "started_at": "", "stopped_at": "", "last_error": "",
             "restarts": 0, "task": None}
WEB_STATE = {"listeners": 0, "peak": 0, "boot": now_iso()}


def status_snapshot() -> dict:
    """Live status (dashboard + /status) — state file se padhta hai."""
    data = dict(DEFAULT_STATE)
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as handle:
            data.update(json.load(handle) or {})
    except Exception:
        pass

    total = int(data.get("total") or 0)
    offset = int(data.get("offset") or 0)
    sent = int(data.get("sent") or 0)
    remaining = max(0, total - offset) if total else None

    started = JOB.get("started_at") or data.get("started_at") or ""
    elapsed = 0.0
    if started:
        try:
            elapsed = (datetime.now(timezone.utc)
                       - datetime.fromisoformat(started)).total_seconds()
        except Exception:
            elapsed = 0.0
    speed = (sent / elapsed) if elapsed > 0 else 0.0

    return {
        "exporter": ("running" if JOB["running"]
                     else ("finished" if data.get("finished") else "stopped")),
        "pid_started": JOB.get("started_at"),
        "restarts": JOB.get("restarts", 0),
        "last_error": JOB.get("last_error") or data.get("last_error") or "",
        "dataset": data.get("dataset") or HF_DATASET,
        "config": data.get("config") or HF_CONFIG,
        "split": data.get("split") or HF_SPLIT,
        "total": total,
        "offset": offset,
        "sent": sent,
        "messages": int(data.get("messages") or 0),
        "remaining": remaining,
        "percent": round((offset / total * 100), 2) if total else 0.0,
        "finished": bool(data.get("finished")),
        "elapsed_sec": int(elapsed),
        "speed_rec_per_sec": round(speed, 2),
        "eta_sec": int(remaining / speed) if (speed and remaining) else None,
        "accounts": {"count": len(POOL), "mode": MODE,
                     "labels": [f"acc{i}" for i in range(1, len(POOL) + 1)],
                     "in_flight": CONCURRENCY, "wait_for_account": WAIT_FOR_ACCOUNT},
        "config_used": {
            "batch_mode": BATCH_MODE, "batch_size": BATCH_SIZE, "msg_chars": MSG_CHARS,
            "per_account": PER_ACCOUNT, "delay": DELAY, "dry_run": DRY_RUN,
        },
        "state_file": STATE_FILE,
        "updated_at": data.get("updated_at"),
        "web": {"listeners": WEB_STATE["listeners"], "boot": WEB_STATE["boot"]},
    }


DASHBOARD_HTML = """<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exporter · Live</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;padding:18px;background:#0a0e18;color:#e6e9f2;
   font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial}
 .wrap{max-width:760px;margin:0 auto}
 h1{font-size:17px;margin:0 0 4px}
 .sub{color:#8b93a7;font-size:12.5px;margin-bottom:16px}
 .card{background:#111827;border:1px solid #1f2740;border-radius:14px;padding:16px;margin-bottom:12px}
 .row{display:flex;justify-content:space-between;gap:12px;padding:7px 0;border-bottom:1px solid #1b2237;font-size:13.5px}
 .row:last-child{border-bottom:0}
 .k{color:#8b93a7}
 .v{font-weight:600;font-family:ui-monospace,Menlo,monospace;text-align:right;word-break:break-all}
 .bar{height:10px;border-radius:99px;background:#1b2237;overflow:hidden;margin:12px 0 6px}
 .fill{height:100%;width:0;background:linear-gradient(90deg,#2b6ef6,#5ee1a0);transition:width .4s}
 .pill{display:inline-block;font-size:11px;padding:3px 10px;border-radius:99px;
   background:#16203a;border:1px solid #1f2740;color:#9fb4dd;font-family:ui-monospace,Menlo,monospace}
 .ok{color:#5ee1a0}.bad{color:#ff9ba8}
</style></head><body><div class="wrap">
<h1>Exporter <span class="pill" id="mode">—</span> <span class="pill" id="live">live</span></h1>
<div class="sub" id="sub">loading…</div>
<div class="card">
  <div class="row"><span class="k">Records sent</span><span class="v" id="sent">—</span></div>
  <div class="bar"><div class="fill" id="fill"></div></div>
  <div class="row"><span class="k">Position</span><span class="v" id="pos">—</span></div>
  <div class="row"><span class="k">Remaining</span><span class="v" id="rem">—</span></div>
  <div class="row"><span class="k">Messages</span><span class="v" id="msgs">—</span></div>
  <div class="row"><span class="k">Speed</span><span class="v" id="speed">—</span></div>
  <div class="row"><span class="k">ETA</span><span class="v" id="eta">—</span></div>
</div>
<div class="card">
  <div class="row"><span class="k">Accounts</span><span class="v" id="acc">—</span></div>
  <div class="row"><span class="k">Batch</span><span class="v" id="batch">—</span></div>
  <div class="row"><span class="k">State file</span><span class="v" id="sf">—</span></div>
  <div class="row"><span class="k">Last error</span><span class="v" id="err">—</span></div>
</div>
</div>
<script>
const $=id=>document.getElementById(id);
const fmt=s=>s==null?"—":(s>=3600?`${(s/3600).toFixed(1)}h`:(s>=60?`${(s/60).toFixed(1)}m`:`${s}s`));
async function tick(){
  try{
    const r=await fetch("/status");const d=await r.json();
    $("mode").textContent=d.exporter.toUpperCase()+" · "+d.config_used.batch_mode;
    $("mode").className="pill "+(d.exporter==="running"?"ok":"bad");
    $("sub").textContent=`${d.dataset} · ${d.config}/${d.split} · restarts ${d.restarts} · updated ${d.updated_at||"—"}`;
    $("sent").textContent=d.sent;
    $("fill").style.width=(d.percent||0)+"%";
    $("pos").textContent=`${d.offset} / ${d.total||"?"} (${d.percent}%)`;
    $("rem").textContent=d.remaining==null?"?":d.remaining;
    $("msgs").textContent=d.messages;
    $("speed").textContent=d.speed_rec_per_sec+" rec/s";
    $("eta").textContent=d.eta_sec==null?"—":fmt(d.eta_sec);
    $("acc").textContent=`${d.accounts.count} × ${d.accounts.mode} · in-flight ${d.accounts.in_flight}`;
    $("batch").textContent=`${d.config_used.batch_size}/msg · ${d.config_used.msg_chars} chars${d.config_used.dry_run?" · DRY RUN":""}`;
    $("sf").textContent=d.state_file;
    $("err").textContent=d.last_error||"—";
    $("live").textContent="live"; $("live").className="pill ok";
  }catch(e){ $("live").textContent="offline"; $("live").className="pill bad"; }
}
tick(); setInterval(tick, 3000);
</script></body></html>"""


def build_web_app():
    """Chhota stdlib HTTP server — koi extra dependency nahi."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import urllib.parse

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):        # Render logs clean rakho
            return

        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass

        def _json(self, code: int, payload: dict):
            self._send(code, json.dumps(payload, ensure_ascii=False, indent=2).encode(),
                       "application/json; charset=utf-8")

        def _auth_ok(self) -> bool:
            if not WEB_TOKEN:
                return True
            token = self.headers.get("x-token") or ""
            query = urllib.parse.urlparse(self.path).query
            if not token:
                token = urllib.parse.parse_qs(query).get("token", [""])[0]
            return token == WEB_TOKEN

        def _loop(self):
            return getattr(self.server, "loop", None)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
            WEB_STATE["listeners"] += 1
            WEB_STATE["peak"] = max(WEB_STATE["peak"], WEB_STATE["listeners"])
            try:
                if path in ("/", "/dashboard", "/ui"):
                    self._send(200, DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
                elif path in ("/health", "/healthz", "/api/health"):
                    snap = status_snapshot()
                    self._json(200, {"status": "ok", "service": "dataset-tg-exporter",
                                     "exporter": snap["exporter"],
                                     "sent": snap["sent"], "total": snap["total"],
                                     "percent": snap["percent"],
                                     "boot": WEB_STATE["boot"]})
                elif path in ("/status", "/api/status", "/progress"):
                    self._json(200, status_snapshot())
                elif path in ("/stop", "/pause", "/api/stop"):
                    if not self._auth_ok():
                        return self._json(401, {"ok": False, "error": "unauthorized (x-token)"})
                    loop, event = self._loop(), STOP_EVENT
                    if loop and event and not event.is_set():
                        loop.call_soon_threadsafe(event.set)
                    JOB["stopped_at"] = now_iso()
                    self._json(200, {"ok": True, "action": "stop",
                                     "note": "current message ke baad rukega, state saved"})
                elif path in ("/state", "/api/state"):
                    try:
                        with open(STATE_FILE, "r", encoding="utf-8") as fh:
                            self._json(200, {"ok": True, "state_file": STATE_FILE,
                                             "state": json.load(fh)})
                    except Exception as exc:
                        self._json(200, {"ok": False, "error": str(exc),
                                         "state_file": STATE_FILE})
                elif path in ("/reset", "/api/reset"):
                    if not self._auth_ok():
                        return self._json(401, {"ok": False, "error": "unauthorized (x-token)"})
                    if JOB["running"]:
                        return self._json(409, {"ok": False,
                                                "error": "pehle /stop karo, phir /reset"})
                    try:
                        if os.path.exists(STATE_FILE):
                            os.replace(STATE_FILE, STATE_FILE + ".bak")
                        reset = dict(DEFAULT_STATE)
                        reset["dataset"] = HF_DATASET
                        reset["config"] = HF_CONFIG
                        reset["split"] = HF_SPLIT
                        reset["started_at"] = now_iso()
                        with open(STATE_FILE, "w", encoding="utf-8") as fh:
                            json.dump(reset, fh, separators=(",", ":"))
                        self._json(200, {"ok": True, "action": "reset",
                                         "note": "state reset (purana state.json.bak me)",
                                         "state_file": STATE_FILE})
                    except Exception as exc:
                        self._json(500, {"ok": False, "error": str(exc)})
                elif path in ("/start", "/resume", "/api/start"):
                    if not self._auth_ok():
                        return self._json(401, {"ok": False, "error": "unauthorized (x-token)"})
                    loop = self._loop()
                    if loop and not JOB["running"]:
                        loop.call_soon_threadsafe(start_job)
                        self._json(200, {"ok": True, "action": "start",
                                         "note": "state file se resume hoga"})
                    else:
                        self._json(200, {"ok": True, "action": "start",
                                         "note": "pehle se chal raha hai"})
                else:
                    self._json(404, {"error": "not found",
                                     "routes": ["/", "/health", "/status", "/stop", "/start"]})
            finally:
                WEB_STATE["listeners"] -= 1

        do_POST = do_GET

    port = PORT or 8080
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    return server


async def start_web_server(loop) -> None:
    """HTTP server ko background thread me chalao (asyncio loop block na ho)."""
    import threading

    server = build_web_app()
    server.loop = loop
    thread = threading.Thread(target=server.serve_forever, name="web", daemon=True)
    thread.start()
    log.info("🌐 WEB SERVICE live on 0.0.0.0:%s  (health: /health · dashboard: / · control: /stop,/start)",
             PORT or 8080)


def start_job() -> None:
    """Exporter ko naye task ke roop me chalao (restart safe)."""
    if JOB["running"]:
        return
    JOB["running"] = True
    JOB["started_at"] = now_iso()
    JOB["restarts"] = JOB.get("restarts", 0) + 1
    if asyncio.get_event_loop().is_closed():
        return
    task = asyncio.ensure_future(supervise())
    JOB["task"] = task


async def supervise() -> None:
    """
    run() ko chalata hai; crash hone par 15s baad WAHIN se restart karta hai
    (state.json se resume) — Render web service 24x7 healthy rehti hai.
    """
    global STOP_EVENT
    while True:
        if STOP_EVENT is None or STOP_EVENT.is_set():
            STOP_EVENT = asyncio.Event()
        try:
            await run()
            JOB["last_error"] = ""
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            msg = str(exc) or exc.__class__.__name__
            JOB["last_error"] = msg[:200]
            log.error("run() ruk gaya: %s -> 60s baad dobara koshish", msg[:200])
            await asyncio.sleep(60)
            continue
        finally:
            JOB["stopped_at"] = now_iso()

        # normal exit (finished / stop) -> web service chalta rahega, dobara start nahi
        JOB["running"] = False
        log.info("exporter ruk gaya (finished/stop). Web service live hai — /start se dobara chala sakte ho.")
        return


def main() -> None:
    """
    Render WEB SERVICE entry point:
      • HTTP server (PORT) chalta rehta hai -> Render health check pass
      • exporter usi process me background task ke roop me chalta hai
      • kaam khatam hone par bhi process zinda rehta hai (Render crash nahi dikhata)
      • /stop, /start, /reset endpoints se control
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    global STOP_EVENT
    STOP_EVENT = asyncio.Event()

    web_enabled = bool(PORT) or WEB in ("1", "true", "yes")
    if web_enabled:
        try:
            loop.run_until_complete(start_web_server(loop))
        except Exception as exc:
            log.error("web server start fail (%s) — sirf exporter chalega", exc)

    if DRY_RUN:
        log.warning("🧪 DRY_RUN=1 — Telegram par kuch nahi jayega (sirf log)")

    if not POOL:
        log.error("❌ koi account nahi mila — Render env me API_IDS/API_HASHES/BOT_TOKENS daalo")
    else:
        start_job()

    # graceful shutdown (Render deploy/stop par SIGTERM bhejta hai)
    def _shutdown(*_):
        log.warning("SIGTERM/SIGINT -> state save karke band ho raha hoon")
        if STOP_EVENT and not STOP_EVENT.is_set():
            loop.call_soon_threadsafe(STOP_EVENT.set)
        loop.call_later(3.0, loop.stop)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except (NotImplementedError, RuntimeError):
            pass

    if web_enabled:
        log.info("web service ke saath exporter chal raha hai — loop forever")
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                log.info("shutdown · sent %s · state %s",
                         status_snapshot().get("sent"), STATE_FILE)
            except Exception:
                pass
    else:
        # sirf worker (Render background worker / VPS)
        try:
            loop.run_until_complete(supervise())
        except KeyboardInterrupt:
            log.warning("KeyboardInterrupt")


if __name__ == "__main__":
    main()

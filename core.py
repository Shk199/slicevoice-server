"""منطق نقي (بدون torch) لتجهيز النص وتجميع المقاطع — قابل للاختبار منفصلاً."""
import re
import unicodedata
from typing import Dict, List, Sequence, Set, Tuple

FRAME_SEC = 0.02  # كل إطار من wav2vec2 = 20 ميلي ثانية

# التشكيل + التطويل + علامات القرآن + محارف الاتجاه
_MARKS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640\u200c-\u200f]")


def to_bool(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def parse_lines(raw) -> List[str]:
    """يقسم النص إلى أسطر غير فارغة. كل سطر = مقطع واحد في الناتج."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig", errors="replace")
    lines = [ln.strip() for ln in raw.replace("\r", "").split("\n")]
    return [ln for ln in lines if ln]


def normalize_word(word: str, allowed: Set[str]) -> str:
    """يزيل التشكيل ويبقي فقط الحروف التي يعرفها النموذج."""
    word = unicodedata.normalize("NFKC", word)
    word = _MARKS.sub("", word)
    word = word.replace("\u0671", "\u0627")  # ألف الوصل ٱ -> ا
    return "".join(ch for ch in word if ch in allowed)


def build_targets(
    lines: Sequence[str], char_ids: Dict[str, int], delim_id: int
) -> Tuple[List[int], List[Tuple[int, int, int]], List[List[str]]]:
    """
    يرجع:
      tokens       : معرفات الحروف مع فاصل الكلمات
      words        : (رقم السطر، أول توكن، آخر توكن) لكل كلمة
      line_letters : حروف كل سطر بعد التطبيع
    """
    allowed = set(char_ids)
    tokens: List[int] = []
    words: List[Tuple[int, int, int]] = []
    line_letters: List[List[str]] = []
    for li, line in enumerate(lines):
        letters: List[str] = []
        for w in line.split():
            norm = normalize_word(w, allowed)
            if not norm:
                continue
            if tokens:
                tokens.append(delim_id)
            first = len(tokens)
            tokens.extend(char_ids[c] for c in norm)
            words.append((li, first, len(tokens) - 1))
            letters.extend(norm)
        line_letters.append(letters)
    return tokens, words, line_letters


def assemble_segments(
    lines: Sequence[str],
    words: Sequence[Tuple[int, int, int]],
    spans: Sequence,
    line_letters: Sequence[Sequence[str]],
    trim_silence: bool = True,
    include_letters: bool = True,
) -> List[dict]:
    """يحوّل مدى كل حرف (start/end/score بالإطارات) إلى مقاطع زمنية لكل سطر."""
    per_line: Dict[int, List[Tuple[int, int]]] = {}
    for li, first, last in words:
        per_line.setdefault(li, []).append((first, last))

    segs: List[dict] = []
    prev_end = 0.0
    for li, line in enumerate(lines):
        ranges = per_line.get(li)
        if not ranges:  # سطر بلا حروف عربية قابلة للمطابقة
            start = end = prev_end
            conf = 0.0
        else:
            start = spans[ranges[0][0]].start * FRAME_SEC
            end = max(spans[ranges[-1][1]].end * FRAME_SEC, start)
            scores = [spans[i].score for a, b in ranges for i in range(a, b + 1)]
            conf = sum(scores) / len(scores)
        prev_end = end
        segs.append(
            {
                "id": li + 1,
                "start": start,
                "end": end,
                "text": line,
                "confidence": min(max(conf, 0.0), 1.0),
                "phonemes": list(line_letters[li]) if include_letters else [],
            }
        )

    if not trim_silence:  # مقاطع متصلة: نهاية كل مقطع = بداية التالي
        for a, b in zip(segs, segs[1:]):
            if b["start"] > a["end"]:
                a["end"] = b["start"]

    for s in segs:
        s["start"] = round(s["start"], 3)
        s["end"] = round(s["end"], 3)
        s["confidence"] = round(s["confidence"], 3)
    return segs

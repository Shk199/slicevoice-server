"""سيرفر مطابقة SliceVoice: يطابق الصوت مع النص سطراً بسطر عبر CTC forced alignment.

POST /api/align  (multipart): audio, text, engine, phoneme_slices, silence_trimming
GET  /           فحص الحالة
"""
import logging
import os
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
import torchaudio.functional as taf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from core import FRAME_SEC, assemble_segments, build_targets, parse_lines, to_bool

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("slicevoice")

MODEL_ID = os.getenv("MODEL_ID", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
SR = 16000
STRIDE = 320                      # عينات لكل إطار
CHUNK_S, CTX_S = 20, 2            # نافذة المعالجة وسياقها بالثواني
MAX_MINUTES = float(os.getenv("MAX_AUDIO_MINUTES", "20"))

STATE: dict = {}
LOCK = threading.Lock()           # معالجة طلب واحد في الوقت (CPU)


@asynccontextmanager
async def lifespan(_: FastAPI):
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    t0 = time.time()
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).eval()
    vocab = processor.tokenizer.get_vocab()
    STATE.update(
        processor=processor,
        model=model,
        char_ids={t: i for t, i in vocab.items() if len(t) == 1 and t != "|"},
        delim_id=vocab["|"],
        blank_id=processor.tokenizer.pad_token_id,
    )
    log.info("model ready in %.1fs", time.time() - t0)
    yield


app = FastAPI(title="SliceVoice Align", lifespan=lifespan)


@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "ready": "model" in STATE, "model": MODEL_ID}


def decode_audio(path: str) -> np.ndarray:
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True,
    )
    if p.returncode != 0 or not p.stdout:
        raise HTTPException(415, "تعذر قراءة الملف الصوتي. استخدم MP3 أو WAV سليماً.")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def probe_sample_rate(path: str) -> int:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate", "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=30,
        )
        return int(r.stdout.strip())
    except Exception:
        return SR


@torch.inference_mode()
def log_probs(audio: np.ndarray) -> torch.Tensor:
    """احتمالات الحروف لكل إطار، بمعالجة على نوافذ حتى لا تنفد الذاكرة."""
    proc, model = STATE["processor"], STATE["model"]
    n, chunk, ctx = len(audio), CHUNK_S * SR, CTX_S * SR
    out = []
    for s in range(0, n, chunk):
        a, b = max(0, s - ctx), min(n, s + chunk + ctx)
        seg = audio[a:b]
        if len(seg) < SR // 2:
            seg = np.pad(seg, (0, SR // 2 - len(seg)))
        x = proc(seg, sampling_rate=SR, return_tensors="pt").input_values
        logits = model(x).logits[0]
        left = (s - a) // STRIDE
        out.append(logits[left:left + chunk // STRIDE])
    return torch.log_softmax(torch.cat(out), dim=-1)


@app.post("/api/align")
def align(
    audio: UploadFile = File(...),
    text: UploadFile = File(...),
    engine: str = Form("wav2vec2-ctc-ar"),
    phoneme_slices: str = Form("true"),
    silence_trimming: str = Form("true"),
):
    if "model" not in STATE:
        raise HTTPException(503, "النموذج ما زال يُحمَّل، أعد المحاولة بعد دقيقة.")

    lines = parse_lines(text.file.read())
    if not lines:
        raise HTTPException(422, "النص المرجعي فارغ.")
    tokens, words, line_letters = build_targets(lines, STATE["char_ids"], STATE["delim_id"])
    if not tokens:
        raise HTTPException(422, "النص لا يحتوي حروفاً عربية يمكن مطابقتها.")

    suffix = os.path.splitext(audio.filename or "")[1] or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio.file.read())
        tmp.flush()
        wav = decode_audio(tmp.name)
        sample_rate = probe_sample_rate(tmp.name)

    duration = len(wav) / SR
    if duration < 0.5:
        raise HTTPException(422, "الملف الصوتي قصير جداً.")
    if duration > MAX_MINUTES * 60:
        raise HTTPException(413, f"الحد الأقصى للصوت {MAX_MINUTES:g} دقيقة.")

    t0 = time.time()
    with LOCK:
        emission = log_probs(wav).unsqueeze(0).float().cpu()
        targets = torch.tensor([tokens], dtype=torch.int32)
        blank = STATE["blank_id"]
        try:
            aligned, scores = taf.forced_align(emission, targets, blank=blank)
        except (RuntimeError, ValueError) as e:
            log.warning("forced_align failed: %s", e)
            raise HTTPException(422, "تعذرت المطابقة: الصوت أقصر من النص أو لا يوافقه.")
        spans = taf.merge_tokens(aligned[0], scores[0].exp(), blank=blank)

    if len(spans) != len(tokens):
        raise HTTPException(500, "تعذر ربط الحروف بالصوت. جرّب مقطعاً أقصر.")

    segments = assemble_segments(
        lines, words, spans, line_letters,
        trim_silence=to_bool(silence_trimming),
        include_letters=to_bool(phoneme_slices),
    )
    log.info("aligned %d lines, %.0fs audio in %.1fs", len(lines), duration, time.time() - t0)
    return {
        "status": "aligned",
        "engine": "wav2vec2-ctc-ar",
        "sample_rate": sample_rate,
        "audio_file": audio.filename,
        "duration_seconds": round(duration, 3),
        "segments": segments,
    }

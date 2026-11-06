"""يُشغَّل أثناء بناء الصورة لتحميل النموذج مسبقاً فيبدأ السيرفر بسرعة."""
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_ID = os.getenv("MODEL_ID", "jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
Wav2Vec2Processor.from_pretrained(MODEL_ID)
Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
print("model cached")

---
title: SliceVoice Align
emoji: ✂️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# سيرفر مطابقة SliceVoice

يطابق ملفاً صوتياً مع نص عربي **سطراً بسطر**، ويرجع توقيت بداية ونهاية كل سطر.
النموذج: `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (محاذاة CTC). التشكيل لا يؤثر، فيُزال داخلياً ويبقى في الناتج كما كتبته.

## الرفع على Hugging Face
1. huggingface.co ← New Space ← اختر SDK: **Docker** (Blank) والعتاد **CPU basic (مجاني)**.
2. من Files ← Add file ← Upload files: ارفع `app.py` و`core.py` و`download_model.py` و`requirements.txt` و`Dockerfile` و`README.md`.
3. انتظر انتهاء البناء (نحو 10 دقائق في المرة الأولى) حتى تصبح الحالة Running.
4. افتح رابط الـSpace؛ يجب أن يظهر `"ready": true`.
5. في التطبيق ← الإعدادات ← رابط السيرفر: `https://اسم-المستخدم-اسم-الـSpace.hf.space/`

## ملاحظات
- كل **سطر** في النص = مقطع واحد. اكتب كل بيت أو جملة في سطر.
- `silence_trimming=true`: توقيت ضيق على الكلام. `false`: كل مقطع يمتد إلى بداية التالي.
- حقل `phonemes` يحمل **حروف** السطر (بدون تشكيل)، وليس فونيمات حقيقية.
- الحد الأقصى للصوت 20 دقيقة (غيّره بالمتغير `MAX_AUDIO_MINUTES`).
- الـSpace المجاني ينام بعد فترة خمول، وأول طلب بعد النوم يتأخر دقيقة أو أكثر.
- الرابط عام: أي شخص يعرفه يستطيع استخدامه.

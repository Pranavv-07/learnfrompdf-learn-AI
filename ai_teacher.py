import json
import google.generativeai as genai
from pypdf import PdfReader

from gtts import gTTS

from moviepy import *

# =========================
# GEMINI SETUP
# =========================

genai.configure(
    api_key="AIzaSyD_9UFdDKfW_DF6UX9uUQ-FPDc_QBQoNks"
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# =========================
# EDUCATIONAL CONTENT
# =========================
reader = PdfReader("Notes.pdf")
content = ""
for page in reader.pages:
    text= page.extract_text()

    if text:
        content += text
""

# =========================
# PROMPT
# =========================

prompt = f"""
You are an AI educational teacher.

Analyze the educational content.

Return ONLY valid JSON.

JSON format:

{{
  "title": "lesson title",

  "bullets": [
  "exactly 3 bullet points"
],

  "narration": " Indian Professor style explanation of less than 50 seconds in English India with mix of formal and informal tones , with correct punctuation and grammar so that when we convert it to speech it is clear and easy to understand"
}}

Educational Content:
{content}
"""

# =========================
# GEMINI RESPONSE
# =========================

response = model.generate_content(prompt)

clean_text = response.text.replace("```json", "")
clean_text = clean_text.replace("```", "")

data = json.loads(clean_text)

# =========================
# EXTRACT AI DATA
# =========================

title_text = data["title"]

bullets = data["bullets"]

narration = data["narration"]

# =========================
# GENERATE VOICE
# =========================

tts = gTTS(
    text=narration,
    lang='te',
    slow=False
)

tts.save("voice.mp3")

# =========================
# CREATE VIDEO
# =========================

background = ColorClip(
    size=(1280, 720),
    color=(30, 60, 90),
    duration=12
)

# TITLE

title = TextClip(
    text=title_text,
    font_size=40,
    color="white",
    margin=(10,10),
    method="caption",
    size=(1100, None)
)

title = title.with_position(("center", 80))
title = title.with_duration(12)

# BULLET 1

Text1 = TextClip(
    text="> " + bullets[0],
    font_size=20,
    color="white",
    margin=(10,10),
    method="caption",
    size=(1000, None)
)
Text1 = Text1.with_position((90, 250))
Text1 = Text1.with_start(2)
Text1 = Text1.with_duration(10)

# BULLET 2

Text2 = TextClip(
    text="> " + bullets[1],
    font_size=20,
    color="white",
    margin=(10,10),
    method="caption",
    size=(1000, None)
)

Text2 = Text2.with_position((90,400))
Text2 = Text2.with_start(4)
Text2 = Text2.with_duration(8)

# BULLET 3

Text3 = TextClip(
    text="> " + bullets[2],
    font_size=20,
    color="white",
    margin=(10,10),
    method="caption",
    size=(1000, None)

)

Text3 = Text3.with_position((90, 500))
Text3 = Text3.with_start(6)
Text3 = Text3.with_duration(6)

# =========================
# COMBINE CLIPS
# =========================

final_video = CompositeVideoClip([
    background,
    title,
    Text1,
    Text2,
    Text3
])

# =========================
# ADD AUDIO
# =========================

audio = AudioFileClip("voice.mp3")

final_video = final_video.with_audio(audio)

# =========================
# EXPORT VIDEO
# =========================

final_video.write_videofile(
    "final_video.mp4",
    fps=24
)
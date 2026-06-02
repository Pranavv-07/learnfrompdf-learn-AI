import streamlit as st
import json
import os
import glob
import subprocess
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google import genai
from gtts import gTTS
from moviepy import *
from pypdf import PdfReader

# =========================================
# PAGE TITLE
# =========================================

st.set_page_config(page_title="AI PDF Teacher", layout="centered")
st.title("Learn AI")
st.write("Upload a PDF and generate an AI teaching video.")

# =========================================
# FONT DETECTION
# =========================================

@st.cache_resource
def find_system_font():
    search_dirs = [
        "/System/Library/Fonts",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "C:\\Windows\\Fonts",
    ]
    preferred = [
        "Arial.ttf", "Helvetica.ttf", "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf", "FreeSans.ttf",
        "NotoSans-Regular.ttf", "Roboto-Regular.ttf",
        "Arial Unicode.ttf",
    ]
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        for pref in preferred:
            matches = glob.glob(os.path.join(search_dir, "**", pref), recursive=True)
            if matches:
                return matches[0]
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        matches = glob.glob(os.path.join(search_dir, "**", "*.ttf"), recursive=True)
        if matches:
            return matches[0]
    try:
        result = subprocess.run(
            ["fc-list", "--format", "%{file}\n"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split("\n"):
            if line.endswith(".ttf"):
                return line
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


FONT_PATH = find_system_font()

if FONT_PATH is None:
    st.error(
        "No TrueType font (.ttf) found on your system. "
        "Please install fonts (e.g., `sudo apt install fonts-dejavu` on Linux)."
    )
    st.stop()


# =========================================
# HELPER: render text → RGBA numpy array
# =========================================

def make_text_image(text, font_size, color=(255, 255, 255),
                    max_width=1100, padding=(20, 10)):
    font = ImageFont.truetype(FONT_PATH, font_size)
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current_line = words[0]
        for word in words[1:]:
            test_line = current_line + " " + word
            bbox = font.getbbox(test_line)
            if bbox[2] - bbox[0] <= max_width - 2 * padding[0]:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

    line_height = font_size + 6
    img_width   = max_width
    img_height  = len(lines) * line_height + 2 * padding[1]

    img  = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = padding[1]
    for line in lines:
        draw.text((padding[0], y), line, font=font, fill=(*color, 255))
        y += line_height
    return np.array(img)


# =========================================
# HELPER: pure-Python animated decoration
#
#  Draws a 200×200 frame for a given time t:
#  • Outer slowly-spinning hexagon ring  (accent colour)
#  • Three orbiting dots                 (white)
#  • Central glowing circle             (pulsing blue-white)
#  • Small "book" icon drawn with lines  (always visible)
#
#  Returns RGBA numpy array (200, 200, 4)
# =========================================

def make_animation_frame(t, total_duration):
    SIZE  = 200
    CX    = SIZE // 2   # 100
    CY    = SIZE // 2   # 100

    img  = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # --- pulsing glow: radius oscillates between 28 and 38 ---
    pulse = 28 + 10 * (0.5 + 0.5 * math.sin(2 * math.pi * t / 1.4))
    r = int(pulse)
    # soft outer halo
    for halo in range(6, 0, -1):
        alpha = int(30 + halo * 10)
        hr    = r + halo * 3
        draw.ellipse(
            [CX - hr, CY - hr, CX + hr, CY + hr],
            fill=(80, 140, 255, alpha)
        )
    # solid centre circle
    draw.ellipse(
        [CX - r, CY - r, CX + r, CY + r],
        fill=(120, 180, 255, 230)
    )

    # --- spinning hexagon ring ---
    hex_r    = 72
    hex_rot  = (2 * math.pi * t / 6.0)          # full rotation every 6 s
    hex_pts  = []
    for k in range(6):
        angle = hex_rot + k * math.pi / 3
        hex_pts.append((
            CX + hex_r * math.cos(angle),
            CY + hex_r * math.sin(angle)
        ))
    for k in range(6):
        x1, y1 = hex_pts[k]
        x2, y2 = hex_pts[(k + 1) % 6]
        draw.line([x1, y1, x2, y2], fill=(100, 160, 255, 180), width=2)

    # --- three orbiting dots ---
    dot_r     = 58
    dot_speed = 2 * math.pi * t / 2.5           # full orbit every 2.5 s
    for k in range(3):
        angle = dot_speed + k * 2 * math.pi / 3
        dx    = CX + dot_r * math.cos(angle)
        dy    = CY + dot_r * math.sin(angle)
        draw.ellipse([dx - 5, dy - 5, dx + 5, dy + 5],
                     fill=(255, 255, 255, 220))

    # --- book icon in the centre (always static) ---
    bx, by = CX, CY
    # book body
    draw.rounded_rectangle(
        [bx - 14, by - 16, bx + 14, by + 16],
        radius=3, outline=(255, 255, 255, 240), width=2
    )
    # spine line
    draw.line([bx, by - 16, bx, by + 16], fill=(255, 255, 255, 200), width=1)
    # two page lines
    draw.line([bx + 4, by - 8, bx + 11, by - 8], fill=(255, 255, 255, 180), width=1)
    draw.line([bx + 4, by,     bx + 11, by],     fill=(255, 255, 255, 180), width=1)
    draw.line([bx + 4, by + 8, bx + 11, by + 8], fill=(255, 255, 255, 180), width=1)

    return np.array(img)


# =========================================
# GEMINI SETUP
# =========================================

api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
if not api_key:
    st.error("GEMINI_API_KEY not found. Set it in Streamlit secrets or as an environment variable..")
    st.stop()

client     = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"

# =========================================
# SESSION STATE
# =========================================

for key, default in [("generated", False), ("data", None), ("video_path", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# =========================================
# PDF UPLOAD
# =========================================

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# =========================================
# GENERATE VIDEO
# =========================================

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF uploaded successfully!")

    if st.button("Generate AI Video"):

        st.session_state.generated  = False
        st.session_state.data       = None
        st.session_state.video_path = None

        # --- READ PDF ---
        with st.spinner("Reading PDF..."):
            reader  = PdfReader("uploaded.pdf")
            content = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text
        content = content[:8000]

        # --- GEMINI PROMPT ---
        prompt = f"""
You are an AI educational teacher.

Analyze the educational content below. And generate the summary of the document, incase it consists of multiple topics, then generate the summary for each topic separately as three points and incase there are any questions in the document then generate answers for them, incase it is a question bank just generate the summary of all the questions as a group in those three points. Based on the type of document you analyze it and generate the required output. If it is a coding question explain the approach and one example testcase with output, if it is a theoretical question then explain the concept in detail and give one example, if it is a question bank then give the summary of all the questions in three points. Understand the document and generate the output accordingly.

Return ONLY valid JSON — no markdown, no explanation, no extra text.

Return exactly 3 bullet points, 3 summary points, and 4 quiz questions.

The JSON must follow this exact structure:

{{
  "title": "lesson title",
  "bullets": [
    "point 1",
    "point 2",
    "point 3"
  ],
  "summary": [
    "summary point 1",
    "summary point 2",
    "summary point 3"
  ],
  "narration": "A detailed explanation of the topic in a university professor style, within 45 seconds when spoken. Use proper punctuation, natural pauses, and paragraph breaks so the speech sounds clear. The tone should be English, friendly, slightly informal, and student-friendly. Do NOT simply read the bullet points — give an engaging explanation of the concepts.",
  "summary_narration": "A quick 10-second revision summary of the key points.",
  "quiz": [
    {{
      "question": "What is ...?",
      "options": ["First option text", "Second option text", "Third option text", "Fourth option text"],
      "answer": "First option text"
    }}
  ]
}}

Quiz rules:
- Each question must have exactly 4 options.
- Only one option should be correct.
- The "answer" field MUST be the COMPLETE, EXACT text of the correct option — NOT a letter like "A" or "B".
- Questions should test understanding, not memorization.
- Questions should be suitable for Indian university students and should be basic level important questions.

Educational Content:
{content}
"""

        # --- GEMINI GENERATION ---
        with st.spinner("Generating lesson using Gemini AI..."):
            response   = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            clean_text = response.text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            try:
                data = json.loads(clean_text)
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse Gemini response as JSON: {e}")
                st.code(clean_text, language="json")
                st.stop()

        # --- EXTRACT ---
        title_text        = data.get("title", "Untitled Lesson")
        bullets           = data.get("bullets", [])
        narration         = data.get("narration", "")
        summary           = data.get("summary", [])
        summary_narration = data.get("summary_narration", "")
        quiz              = data.get("quiz", [])

        st.subheader("Generated Lesson")
        st.write("### Title")
        st.write(title_text)

        # --- VOICE ---
        with st.spinner("Generating narration audio..."):
            tts = gTTS(text=narration, lang='en', slow=False)
            tts.save("voice.mp3")

        # --- RENDER VIDEO ---
        with st.spinner("Rendering AI video..."):

            audio          = AudioFileClip("voice.mp3")
            video_duration = audio.duration

            # ── Background ──────────────────────────────────────────────────
            background = ColorClip(size=(1280, 720), color=(20, 20, 40),
                                   duration=video_duration)

            # ── Title  (font 38, y=28 — well above divider at y=108) ────────
            title_img = make_text_image(title_text, font_size=38, max_width=1180,
                                        color=(255, 255, 255))
            title_clip = ImageClip(title_img, transparent=True)
            title_clip = title_clip.with_position(("center", 28))
            title_clip = title_clip.with_duration(video_duration)

            # ── Divider line (y=108, clearly below title) ───────────────────
            div_img = Image.new("RGBA", (1180, 3), (100, 100, 200, 200))
            divider = ImageClip(np.array(div_img), transparent=True)
            divider = divider.with_position((50, 108))
            divider = divider.with_duration(video_duration)

            # ── Bullets  (left column x=90, max_width=760) ──────────────────
            #   Layout (1280×720):
            #     Title          y=28–100
            #     Divider        y=108
            #     Bullet 1       y=130
            #     Bullet 2       y=270
            #     Bullet 3       y=410
            #     Summary        y=555
            #     Animation      bottom-right: x=1050, y=490  (200×200)
            bullet_clips     = []
            bullet_y         = [130, 270, 410]
            bullet_starts    = [0.10, 0.30, 0.50]
            bullet_durations = [0.90, 0.70, 0.50]

            for idx in range(min(3, len(bullets))):
                b_img = make_text_image(
                    "• " + bullets[idx],
                    font_size=22, max_width=760, color=(220, 220, 255)
                )
                bc = ImageClip(b_img, transparent=True)
                bc = bc.with_position((90, bullet_y[idx]))
                bc = bc.with_start(video_duration * bullet_starts[idx])
                bc = bc.with_duration(video_duration * bullet_durations[idx])
                bc = bc.with_effects([vfx.FadeIn(0.8)])
                bullet_clips.append(bc)

            # ── Summary  (y=555, fades in at 80%) ───────────────────────────
            summary_display = "Summary:\n" + "\n".join(
                f"  {i+1}. {s}" for i, s in enumerate(summary)
            )
            s_img = make_text_image(summary_display, font_size=18,
                                    max_width=900, color=(180, 255, 180))
            summary_clip = ImageClip(s_img, transparent=True)
            summary_clip = summary_clip.with_position((90, 555))
            summary_clip = summary_clip.with_start(video_duration * 0.80)
            summary_clip = summary_clip.with_duration(video_duration * 0.20)
            summary_clip = summary_clip.with_effects([vfx.FadeIn(0.5)])

            # ── Pure-Python animation  ───────────────────────────────────────
            #   Position: bottom-right corner, x=1050, y=490
            #   Size: 200×200
            #   Visible: from 15% of video to end (so it's always on screen)
            #   Built entirely with Pillow — NO external API, NO errors
            ANIM_X     = 1050
            ANIM_Y     = 490
            anim_start = video_duration * 0.15
            anim_dur   = video_duration * 0.85   # 15% → 100%

            def animation_make_frame(t):
                """Return a 200×200 RGBA numpy array for time t."""
                return make_animation_frame(t, anim_dur)

            anim_clip = VideoClip(animation_make_frame, duration=anim_dur)
            anim_clip = anim_clip.with_position((ANIM_X, ANIM_Y))
            anim_clip = anim_clip.with_start(anim_start)
            anim_clip = anim_clip.with_effects([vfx.FadeIn(1.0)])

            # ── Compose ──────────────────────────────────────────────────────
            all_clips  = [background, title_clip, divider] + bullet_clips + [summary_clip, anim_clip]
            final      = CompositeVideoClip(all_clips)
            final      = final.with_duration(video_duration)
            final      = final.with_audio(audio)

            output_path = "final_video.mp4"
            final.write_videofile(
                output_path,
                fps=10,
                preset="ultrafast",
                threads=4,
                logger=None,
            )
            audio.close()

        st.session_state.generated  = True
        st.session_state.data       = data
        st.session_state.video_path = output_path
        st.rerun()

# =========================================
# DISPLAY RESULTS
# =========================================

if st.session_state.generated and st.session_state.data is not None:

    data       = st.session_state.data
    title_text = data.get("title", "Untitled Lesson")
    bullets    = data.get("bullets", [])
    quiz       = data.get("quiz", [])

    st.subheader("Generated Lesson")
    st.write(f"**{title_text}**")
    for b in bullets:
        st.write(f"- {b}")

    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        st.success("AI Video Generated Successfully!")
        st.video(st.session_state.video_path)
    else:
        st.warning("Video file not found. Please regenerate.")

    # ── Quiz ─────────────────────────────────────────────────────────────────
    if quiz:
        st.subheader("Knowledge Check")
        user_answers = []

        for i, q in enumerate(quiz):
            st.write(f"**Question {i + 1}:** {q['question']}")
            user_answer = st.radio(
                f"Choose your answer for Q{i + 1}:",
                q["options"],
                key=f"quiz_{i}",
                index=None
            )
            user_answers.append(user_answer)

        if st.button("Submit Quiz"):
            unanswered = [i + 1 for i, ans in enumerate(user_answers) if ans is None]
            if unanswered:
                st.warning(f"Please answer all questions. Unanswered: {', '.join(map(str, unanswered))}")
            else:
                score   = 0
                results = []

                for i, q in enumerate(quiz):
                    correct_answer   = q["answer"]
                    user_ans         = user_answers[i]
                    options          = q.get("options", [])
                    is_correct       = False
                    resolved_correct = correct_answer

                    if user_ans == correct_answer:
                        is_correct = True
                    elif user_ans.strip().lower() == correct_answer.strip().lower():
                        is_correct = True
                    elif correct_answer.strip().upper() in ("A", "B", "C", "D") and options:
                        letter_index = ord(correct_answer.strip().upper()) - ord("A")
                        if 0 <= letter_index < len(options):
                            resolved_correct = options[letter_index]
                            if user_ans == resolved_correct:
                                is_correct = True
                            elif user_ans.strip().lower() == resolved_correct.strip().lower():
                                is_correct = True

                    if is_correct:
                        score += 1
                    results.append((is_correct, resolved_correct, user_ans))

                st.divider()
                st.success(f"Your Score: {score}/{len(quiz)}")

                for i, (is_correct, correct_ans, user_ans) in enumerate(results):
                    icon = "✅" if is_correct else "❌"
                    st.write(
                        f"{icon} **Q{i + 1}:** Correct Answer: **{correct_ans}** "
                        f"| Your Answer: **{user_ans}**"
                    )

                if score == len(quiz):
                    st.balloons()
                    st.success("Excellent! You understood the lesson very well.")
                elif score >= len(quiz) // 2:
                    st.info("Good job! A quick revision will make it even better.")
                else:
                    st.warning("Consider watching the lesson once again.")

import streamlit as st
import json
import os
import glob
import subprocess
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from google import genai
from google.genai import types
from gtts import gTTS
from moviepy import *
from pypdf import PdfReader

# =========================================
# PAGE TITLE
# =========================================

st.set_page_config(
    page_title="AI PDF Teacher",
    layout="centered"
)

st.title("Learn AI")
st.write("Upload a PDF and generate an AI teaching video.")

# =========================================
# FONT DETECTION FOR MOVIEPY
# =========================================


@st.cache_resource
def find_system_font():
    """Find a usable TrueType font on the system."""
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
            matches = glob.glob(
                os.path.join(search_dir, "**", pref), recursive=True
            )
            if matches:
                return matches[0]

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        matches = glob.glob(
            os.path.join(search_dir, "**", "*.ttf"), recursive=True
        )
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
        "moviepy requires a font file for TextClip. "
        "Please install fonts (e.g., `brew install font-dejavu` on macOS "
        "or `sudo apt install fonts-dejavu` on Linux)."
    )
    st.stop()


def make_text_image(text, font_size, color=(255, 255, 255),
                    max_width=1100, padding=(20, 10)):
    """
    Render text to a transparent RGBA numpy array using Pillow.
    """
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
    img_width = max_width
    img_height = len(lines) * line_height + 2 * padding[1]

    img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = padding[1]
    for line in lines:
        draw.text((padding[0], y), line, font=font, fill=(*color, 255))
        y += line_height

    return np.array(img)


# =========================================
# GEMINI SETUP
# =========================================

api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))

if not api_key:
    st.error("GEMINI_API_KEY not found. Set it in Streamlit secrets or as an environment variable.")
    st.stop()

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"

# =========================================
# INITIALIZE SESSION STATE
# =========================================

if "generated" not in st.session_state:
    st.session_state.generated = False
if "data" not in st.session_state:
    st.session_state.data = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None

# =========================================
# PDF UPLOAD
# =========================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

# =========================================
# GENERATE VIDEO BUTTON
# =========================================

if uploaded_file:

    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    if st.button("Generate AI Video"):

        # Reset state for a fresh generation
        st.session_state.generated = False
        st.session_state.data = None
        st.session_state.video_path = None

        # =========================================
        # READ PDF
        # =========================================

        with st.spinner("Reading PDF..."):
            reader = PdfReader("uploaded.pdf")
            content = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text

        content = content[:8000]

        # =========================================
        # GEMINI PROMPT
        # =========================================

        prompt = f"""
You are an AI educational teacher.

Analyze the educational content below. And generate the summary of the document, incase it consists of multiple topics, then generate the summary for each topic separately. as three points and incase there are any questions in the document then generate answers for them, incase it is a question bank just generate the summary of the all the questions as a group in those three points. Based on the type of document you analyze it and generate the required output. if it is a coding question explain the approach and one example testcase with output, if it is a theoretical question then explain the concept in detail and give one example, if it is a question bank then give the summary of all the questions in three points. understand the document and generate the output accordingly.

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
  "image_prompt": "A short, vivid description (1-2 sentences) of an educational illustration that visually represents the main concept of this lesson. Describe a clean, modern, flat-style illustration suitable for a teaching video. Do NOT include any text in the image description.",
  "narration": "A detailed explanation of the topic in an university professor style, within 45 seconds when spoken. Use proper punctuation, natural pauses, and paragraph breaks so the speech sounds clear. The tone should be English, friendly, slightly informal, and student-friendly. Do NOT simply read the bullet points — give an engaging explanation of the concepts.",
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
  For example, if options are ["Photosynthesis", "Respiration", "Osmosis", "Diffusion"] and the correct answer is Photosynthesis, set "answer": "Photosynthesis".
- Questions should test understanding, not memorization.
- Questions should be suitable for Indian university students and should be basic level important questions.

Educational Content:
{content}
"""

        # =========================================
        # GEMINI GENERATION
        # =========================================

        with st.spinner("Generating lesson using Gemini AI..."):

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

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

        # =========================================
        # EXTRACT DATA
        # =========================================

        title_text        = data.get("title", "Untitled Lesson")
        bullets           = data.get("bullets", [])
        narration         = data.get("narration", "")
        summary           = data.get("summary", [])
        summary_narration = data.get("summary_narration", "")
        quiz              = data.get("quiz", [])
        image_prompt      = data.get("image_prompt", "")

        # =========================================
        # DISPLAY GENERATED CONTENT
        # =========================================

        st.subheader("Generated Lesson")
        st.write("### Title")
        st.write(title_text)

        # =========================================
        # GENERATE AI IMAGE
        # FIX: correct model  →  gemini-3.1-flash-image-preview
        #      no response_modalities needed
        #      check part.inline_data directly (not inline_data.data attribute)
        # =========================================

        generated_image_path = None

        if image_prompt:
            with st.spinner("Generating AI illustration..."):
                try:
                    img_response = client.models.generate_content(
                        model="gemini-3.1-flash-image-preview",   # ← FIXED model
                        contents=image_prompt,
                        # No response_modalities config needed for this model
                    )

                    pil_img = None
                    if (img_response.candidates
                            and img_response.candidates[0].content
                            and img_response.candidates[0].content.parts):
                        for part in img_response.candidates[0].content.parts:
                            if hasattr(part, "inline_data") and part.inline_data is not None:
                                pil_img = Image.open(
                                    BytesIO(part.inline_data.data)
                                )
                                break

                    if pil_img is not None:
                        pil_img = pil_img.convert("RGBA")

                        # Constrain to 200×200 — fits safely in bottom-right corner
                        pil_img.thumbnail((200, 200), Image.LANCZOS)

                        # Rounded corners
                        w, h = pil_img.size
                        corner_radius = 15
                        mask = Image.new("L", (w, h), 0)
                        mask_draw = ImageDraw.Draw(mask)
                        mask_draw.rounded_rectangle(
                            [(0, 0), (w, h)],
                            radius=corner_radius,
                            fill=255
                        )
                        pil_img.putalpha(mask)

                        pil_img.save("ai_illustration.png")
                        generated_image_path = "ai_illustration.png"
                        st.success("AI illustration generated!")
                    else:
                        st.warning("Image generation returned no image data.")

                except Exception as e:
                    st.warning(
                        f"Could not generate AI image (will continue without it): {e}"
                    )

        # =========================================
        # GENERATE VOICE
        # =========================================

        with st.spinner("Generating narration audio..."):
            tts = gTTS(text=narration, lang='en', slow=False)
            tts.save("voice.mp3")

        # =========================================
        # CREATE VIDEO
        # =========================================

        with st.spinner("Rendering AI video..."):

            audio = AudioFileClip("voice.mp3")
            video_duration = audio.duration

            # --------------------------------------------------
            # VIDEO LAYOUT  (1280 × 720)
            #
            #  ┌──────────────────────────────────────────────┐
            #  │            TITLE  (y = 30)                   │
            #  │          ─────────────── (divider y=115)     │
            #  ├──────────────────────────────────────────────┤
            #  │  • Bullet 1  (y = 175)                       │
            #  │  • Bullet 2  (y = 300)   [max_width=760]     │
            #  │  • Bullet 3  (y = 420)                       │
            #  ├──────────────────────────────────────────────┤
            #  │  📌 Summary  (y = 555)                       │
            #  ├────────────────────────────┬─────────────────┤
            #  │                            │  AI IMAGE       │
            #  │                            │  bottom-right   │
            #  │                            │  x=1050, y=490  │
            #  └────────────────────────────┴─────────────────┘
            # --------------------------------------------------

            background = ColorClip(
                size=(1280, 720),
                color=(20, 20, 40),
                duration=video_duration
            )

            # FIX 1: title font_size reduced from 45 → 40
            #         and moved up to y=30 so divider at y=115 no longer clips it
            title_img = make_text_image(title_text, font_size=40, max_width=1180)
            title = ImageClip(title_img, transparent=True)
            title = title.with_position(("center", 30))
            title = title.with_duration(video_duration)

            # Thin horizontal divider — sits clearly BELOW title text
            divider_img = Image.new("RGBA", (1180, 3), (100, 100, 180, 200))
            divider = ImageClip(np.array(divider_img), transparent=True)
            divider = divider.with_position((50, 115))
            divider = divider.with_duration(video_duration)

            # BULLET CLIPS  (left column only, max_width=760)
            bullet_clips = []
            bullet_y_positions = [155, 290, 420]
            bullet_start_fracs = [0.10, 0.30, 0.50]
            bullet_dur_fracs   = [0.90, 0.70, 0.50]

            for idx in range(min(3, len(bullets))):
                bullet_text = "• " + bullets[idx]
                bullet_img  = make_text_image(
                    bullet_text,
                    font_size=22,
                    max_width=760,
                    color=(220, 220, 255)
                )
                clip = ImageClip(bullet_img, transparent=True)
                clip = clip.with_position((90, bullet_y_positions[idx]))
                clip = clip.with_start(video_duration * bullet_start_fracs[idx])
                clip = clip.with_duration(video_duration * bullet_dur_fracs[idx])
                clip = clip.with_effects([vfx.FadeIn(0.8)])
                bullet_clips.append(clip)

            # SUMMARY TEXT  (left column, y=555)
            summary_display = "Summary:\n" + "\n".join(
                f"  {i+1}. {s}" for i, s in enumerate(summary)
            )
            summary_img = make_text_image(
                summary_display,
                font_size=18,
                max_width=900,
                color=(180, 255, 180)
            )
            summary_text = ImageClip(summary_img, transparent=True)
            summary_text = summary_text.with_position((90, 555))
            summary_text = summary_text.with_start(video_duration * 0.80)
            summary_text = summary_text.with_duration(video_duration * 0.20)
            summary_text = summary_text.with_effects([vfx.FadeIn(0.5)])

            # -------------------------------------------------------
            # AI ILLUSTRATION — bottom-right corner, no overlap
            #
            # Safe zone for image:
            #   Right side x > 860 (bullets end at x~850 max)
            #   Bottom    y > 480 (bullets end ~y 420+lineheight)
            #   Image max 200×200, placed at bottom-right with 30px margin
            #     x = 1280 - img_w - 30
            #     y = 720  - img_h - 30
            #
            # Visible window: 25% → 75% (bullets animate 10-50%; summary at 80%)
            # -------------------------------------------------------
            illustration_clip = None
            if generated_image_path and os.path.exists(generated_image_path):
                ai_img = Image.open(generated_image_path).convert("RGBA")
                img_w, img_h = ai_img.size   # at most 200 × 200

                x_pos = 1280 - img_w - 30    # right-aligned with 30px margin
                y_pos = 720  - img_h - 30    # bottom-aligned with 30px margin

                illustration_clip = ImageClip(
                    np.array(ai_img), transparent=True
                )
                illustration_clip = illustration_clip.with_position((x_pos, y_pos))
                illustration_clip = illustration_clip.with_start(
                    video_duration * 0.25
                )
                illustration_clip = illustration_clip.with_duration(
                    video_duration * 0.50    # visible from 25% → 75%
                )
                illustration_clip = illustration_clip.with_effects(
                    [vfx.FadeIn(1.0)]
                )

            # =========================================
            # COMBINE VIDEO
            # =========================================

            all_clips = [background, title, divider] + bullet_clips + [summary_text]
            if illustration_clip is not None:
                all_clips.append(illustration_clip)

            final_video = CompositeVideoClip(all_clips)
            final_video = final_video.with_duration(audio.duration)
            final_video = final_video.with_audio(audio)

            # =========================================
            # EXPORT VIDEO
            # =========================================

            output_path = "final_video.mp4"
            final_video.write_videofile(
                output_path,
                fps=10,
                preset="ultrafast",
                threads=4,
                logger=None,
            )

            audio.close()

        # =========================================
        # SAVE TO SESSION STATE
        # =========================================

        st.session_state.generated  = True
        st.session_state.data       = data
        st.session_state.video_path = output_path

        st.rerun()

# =========================================
# DISPLAY RESULTS (persisted via session state)
# =========================================

if st.session_state.generated and st.session_state.data is not None:

    data       = st.session_state.data
    title_text = data.get("title", "Untitled Lesson")
    bullets    = data.get("bullets", [])
    quiz       = data.get("quiz", [])

    # =========================================
    # SHOW LESSON INFO
    # =========================================

    st.subheader("Generated Lesson")
    st.write(f"**{title_text}**")
    for b in bullets:
        st.write(f"- {b}")

    # =========================================
    # SHOW FINAL VIDEO
    # =========================================

    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        st.success("AI Video Generated Successfully!")
        st.video(st.session_state.video_path)
    else:
        st.warning("Video file not found. Please regenerate.")

    # =========================================
    # QUIZ SECTION
    # =========================================

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

            unanswered = [
                i + 1 for i, ans in enumerate(user_answers) if ans is None
            ]
            if unanswered:
                st.warning(
                    f"Please answer all questions. "
                    f"Unanswered: {', '.join(map(str, unanswered))}"
                )
            else:
                score   = 0
                results = []

                for i, q in enumerate(quiz):
                    correct_answer   = q["answer"]
                    user_ans         = user_answers[i]
                    options          = q.get("options", [])

                    is_correct       = False
                    resolved_correct = correct_answer

                    # Exact match
                    if user_ans == correct_answer:
                        is_correct = True
                    # Case/whitespace-insensitive
                    elif user_ans.strip().lower() == correct_answer.strip().lower():
                        is_correct = True
                    # Gemini returned a letter "A"/"B"/"C"/"D"
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

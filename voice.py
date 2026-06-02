from gtts import gTTS

text = """
In this project, an AI-powered PDF Explainer Agent is developed to transform PDF documents,
 into engaging educational videos. Built using Python and Streamlit, the application is powered by the Gemini A P I.
Users begin by uploading a PDF document, such as Java Exception Handling notes. The system analyzes the document 
and extracts key concepts, generating structured presentation slides with concise bullet points and summaries. 
The Gemini A P I returns the extracted content in a structured JSON format containing slide points, narration scripts, and quiz questions.
The generated narration is then converted into audio using Google's Text-to-Speech (g T T S). Using Moviepie in Python, 
the slides and audio are synchronized and combined to automatically create an educational video. Additionally, the system generates a quiz consisting of four questions based on the document content, helping users assess their understanding.
This end-to-end automation simplifies the process of converting study material into interactive learning content, 
making it highly beneficial for students and educators.
Check out the complete live demo and access the website through the Streamlit-hosted live link.

"""

tts = gTTS(
        text=text,
        lang='en'
        )

tts.save("voicefinal.mp3")

print("Audio generated")
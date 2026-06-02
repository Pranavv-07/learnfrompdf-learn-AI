from moviepy import * 
background = ColorClip(
    size=(1280,720),
    color=(110,235,63),
    duration=10
)
Title = TextClip(
    text="CPU scheduling",
    font_size=90,
    color='white',
    margin=(20,20)
)
Title= Title.with_position(('center',120))
Title = Title.with_duration(10)

Text1 = TextClip(
    text="> Manages the Process execution",
    font_size=30,
    color='white',
    margin=(20,20)
)
Text1 = Text1.with_position((120,250))
Text1 = Text1.with_start(2)
Text1 = Text1.with_duration(8)

Text2 = TextClip(
    text="> Optimizes CPU usage",
    font_size=30,
    color='white',
    margin=(20,20)
)
Text2 = Text2.with_position((120,350))
Text2= Text2.with_start(4)
Text2 = Text2.with_duration(8)

Text3= TextClip(
    text="> Improves system efficiency",
    font_size=30,
    color='white',
    margin=(20,20)
)
Text3 = Text3.with_position((120,450))
Text3= Text3.with_start(6)
Text3 = Text3.with_duration(8)

final_video= CompositeVideoClip(
    [
        background,
        Title,
        Text1,
        Text2,
        Text3
    ]
)
audio = AudioFileClip("voice.mp3")
final_video = final_video.with_audio(audio)
final_video.write_videofile("scene.mp4", fps=24)


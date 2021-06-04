import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty("voices")

engine.setProperty("rate", 200)
engine.setProperty("voice", voices[0].id)
engine.say("Hello, what can I do for you? Recording starts…")
engine.runAndWait()

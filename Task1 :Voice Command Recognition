import speech_recognition as sr

def recognize_command():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        # Recognize speech using Google Speech Recognition
        command = recognizer.recognize_google(audio).lower()
        print("You said:", command)
        return command
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand your command.")
        return None
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None

def execute_command(command):
    # Add your actionable tasks here
    if "open browser" in command:
        # Example: Open browser task
        print("Opening browser...")
    elif "play music" in command:
        # Example: Play music task
        print("Playing music...")
    else:
        print("Command not recognized.")

if __name__ == "__main__":
    while True:
        command = recognize_command()
        if command:
            execute_command(command)

import pyttsx3

TEXT_PATH = "data/last_validation.txt"

def main():
    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    main()
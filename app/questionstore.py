import json
import os

def save_question_json(mcq: str, topic: str, subtopic: str):
    folder = "questions_json"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{topic}_{subtopic}.json")
    data = {"mcq": mcq}
    
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing = json.load(f)
    else:
        existing = []
    
    existing.append(data)
    with open(filename, "w") as f:
        json.dump(existing, f, indent=2)

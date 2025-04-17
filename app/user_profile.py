import json

def get_user_profile():
    try:
        with open("user_profile.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def update_user_profile(topic: str, subtopic: str, level: int, mcq: dict = None):
    profile = get_user_profile()
    user_key = f"{topic}_{subtopic}"

    if user_key not in profile:
        profile[user_key] = {"level": level, "mcqs": []}
    
    profile[user_key]["level"] = level
    if mcq:  # Only append MCQ if provided
        profile[user_key]["mcqs"].append(mcq)

    with open("user_profile.json", "w") as file:
        json.dump(profile, file, indent=4)

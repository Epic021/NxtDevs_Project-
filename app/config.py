def get_urls_for_topic(topic: str, subtopic: str = None):
    """
    Returns a list of URLs related to the provided topic and subtopic for document retrieval.
    """
    topic_urls = {
        "frontend": {
            "angular": [
                "https://angular.io/docs",
                "https://www.tutorialspoint.com/angular/index.htm"
            ],
            "react": [
                "https://reactjs.org/docs/getting-started.html",
                "https://reactjs.org/tutorial/tutorial.html"
            ]
        },
        "backend": {
            "django": [
                "https://www.djangoproject.com/start/",
                "https://realpython.com/tutorials/django/"
            ],
            "go": [
                "https://golang.org/doc/tutorial/create-module",
                "https://www.golang-book.com/"
            ],
            "ruby": [
                "https://rubyonrails.org/learn/",
                "https://www.ruby-lang.org/en/documentation/"
            ]
        },
        "mobile": {
            "flutter": [
                "https://flutter.dev/docs",
                "https://www.udemy.com/course/flutter-dart-the-complete-guide/"
            ],
            "kotlin": [
                "https://kotlinlang.org/docs/home.html",
                "https://developer.android.com/kotlin"
            ],
            "swift": [
                "https://developer.apple.com/swift/",
                "https://www.raywenderlich.com/ios/learn"
            ],
            "react_native": [
                "https://reactnative.dev/docs/getting-started",
                "https://reactnative.dev/docs/tutorial"
            ]
        },
        "programming": {
            "python": [
                "https://www.learnpython.org/",
                "https://realpython.com/"
            ],
            "java": [
                "https://www.w3schools.com/java/",
                "https://docs.oracle.com/javase/tutorial/"
            ],
            "javascript": [
                "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
                "https://www.w3schools.com/js/"
            ]
        },
        "interview": {
            "leetcode": [
                "https://leetcode.com/problemset/all/",
                "https://www.geeksforgeeks.org/leetcode-questions/"
            ],
            "puzzles": [
                "https://www.brainzilla.com/logic/logic-puzzles/",
                "https://www.puzzles-to-print.com/"
            ],
            "riddles": [
                "https://www.riddles.com/",
                "https://www.brainpickings.org/2017/11/02/best-riddles/"
            ]
        }
    }

    if not topic in topic_urls:
        return []
    
    if subtopic:
        return topic_urls[topic].get(subtopic, [])
    
    # If no subtopic specified, return all URLs for the topic
    return [url for subtopic_urls in topic_urls[topic].values() for url in subtopic_urls]


def estimate_user_level(responses: list) -> int:
    """
    Estimates user level based on assessment responses.
    Returns a level between 1-3 (beginner, intermediate, advanced)
    """
    if not responses:
        return 1
        
    correct_answers = sum(1 for r in responses if r['selected'] == r['correct'])
    accuracy = correct_answers / len(responses)
    
    if accuracy >= 0.8:
        return 3  # Advanced
    elif accuracy >= 0.5:
        return 2  # Intermediate
    else:
        return 1  # Beginner


def save_assessment_mcqs(topic: str, subtopic: str, mcqs: list):
    """
    Saves assessment MCQs to a JSON file
    """
    import json
    import os
    
    folder = "assessments"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{topic}_{subtopic}_assessment.json")
    
    with open(filename, "w") as f:
        json.dump(mcqs, f, indent=4)

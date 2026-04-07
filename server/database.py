users = {}
scores = []

def add_user(username, password):
    if username in users:
        return {"status": "fail"}
    users[username] = password
    return {"status": "success"}

def check_user(username, password):
    if users.get(username) == password:
        return {"status": "success"}
    return {"status": "fail"}

def add_score(username, score):
    scores.append({"user": username, "score": score})

def get_leaderboard():
    return sorted(scores, key=lambda x: x["score"], reverse=True)
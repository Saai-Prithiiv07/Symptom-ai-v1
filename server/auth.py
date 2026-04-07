from server.database import add_user, check_user

def register_user(username, password):
    return add_user(username, password)

def login_user(username, password):
    return check_user(username, password)
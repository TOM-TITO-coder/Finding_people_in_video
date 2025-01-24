import os
import json

DATABASE_FILE = "app/database.json"

def save_image(file, upload_dir):
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    return file_path

def load_database():
    if not os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "w") as db:
            json.dump([], db)
        
    with open(DATABASE_FILE, "r") as db:
        return json.load(db)
        
def save_to_database(entry):
    database = load_database()
    database.append(entry)
    
    with open(DATABASE_FILE, "w") as db:
        json.dump(database, db, indent=4)
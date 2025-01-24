from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import os
import cv2
from pathlib import Path

from app.processing import extract_frames, extract_feature, detect_faces, match_faces, frames_to_timestamps, save_highlighted_video
from app.utils import save_image, load_database, save_to_database
app = FastAPI()

UPLOAD_DIR = "app/uploads/"
OUTPUT_DIR = "app/static/"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/process/")
async def process_video(video_file: UploadFile, target_image: UploadFile):
    try:
        # Save uploaded files
        video_path = os.path.join(UPLOAD_DIR, video_file.filename)
        target_path = os.path.join(UPLOAD_DIR, target_image.filename)
        
        with open(video_path, 'wb') as video_out:
            video_out.write(await video_file.read())
        with open(target_path, "wb") as target_out:
            target_out.write(await target_image.read())
            
        # Extract embedding for the target image
        target_img = cv2.imread(target_path)
        target_embedding = extract_feature(target_img)
        
        if not target_embedding:
            return JSONResponse(status_code=400, content={"error": "Could not extract target embedding."})
            
        # Processing
        frames, frame_rate = extract_frames(video_path, frame_interval=10)
        detected_faces = detect_faces(frames)
        
        if not detected_faces:
            return JSONResponse(status_code=404, content={"error": "No faces detected in the video"})
        
        # Load known faces from the database
        knowns_faces = load_database()
        
        
        # Match faces in the video with the target image's embedding
        matched_frames = match_faces(detected_faces, target_embedding, knowns_faces)
        timestamps = frames_to_timestamps(matched_frames, frame_rate)
        
        out_put_video_path = os.path.join(OUTPUT_DIR, f"output_{Path(video_file.filename).stem}.mp4")
        save_highlighted_video(video_path, matched_frames, out_put_video_path)
        
        return {
            "output_video_url": f"/static/output_{Path(video_file.filename).stem}.mp4",
            "timestamps": timestamps
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"erros": str(e)})

@app.get("/static/{filename}")
async def serve_static_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    
    return JSONResponse(status_code=404, content={"error": "File not Found"})


@app.post("/upload_image_to_db")
async def upload_image_to_db(image_file: UploadFile, name: str, age: int):
    try:
        image_path = save_image(image_file, UPLOAD_DIR)
        image = cv2.imread(image_path)
        embedding = extract_feature(image)
        
        if not embedding:
            return JSONResponse(
                status_code=400, 
                content={"error": "Failed to generate embedding."}
            )
        
        entry = {
            "name": name,
            "age": age,
            "embedding": embedding
        }
        save_to_database(entry)
        
        return {
            "message": f"Embedding for {name} save successfully!"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import os
import cv2
from pathlib import Path

from app.processing import extract_frames, extract_feature, detect_faces, match_faces, frames_to_timestamps, save_highlighted_video

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
            
        # Processing
        frames, frame_rate = extract_frames(video_path, frame_interval=10)
        detected_faces = detect_faces(frames)
        
        target_image = cv2.imread(target_path)
        target_embedding = extract_feature(target_image)
        
        if not target_embedding:
            return JSONResponse(status_code=400, content={"error": "Could not extract target embedding."})
        
        matched_frames = match_faces(target_embedding, detected_faces)
        timestamps = frames_to_timestamps(matched_frames, frame_rate)
        
        out_put_video_path = os.path.join(OUTPUT_DIR, f"output_{Path(video_file.filename).stem}.mp4")
        save_highlighted_video(video_path, detected_faces, out_put_video_path)
        
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

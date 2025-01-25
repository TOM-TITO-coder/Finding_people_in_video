from fastapi import FastAPI, File, HTTPException, UploadFile, Form, Request, logger
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
import os
import cv2
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging.config
import logging

# from app
from app.config import LOGGING_CONFIG, FACE_DETECTION_THRESHOLD
from app.processing import extract_frames, extract_feature, detect_faces, match_faces, frames_to_timestamps, save_highlighted_video
from app.utils import save_image, load_database, save_to_database
from app.models import FaceEntry, ProcessingRequest, ProcessingResponse

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)

app = FastAPI(
    title="Face Detection Web App",
    description="Advanced face detection and matching application",
    version="1.0.0"
)

UPLOAD_DIR = "app/uploads/"
OUTPUT_DIR = "app/static/"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static file directory
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Set up templates 
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process/", response_model=ProcessingResponse)
async def process_video(
    video_file: UploadFile = File(...), 
    target_image: UploadFile = File(...)
):
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
        matched_frames = match_faces(
            detected_faces, 
            target_embedding, 
            knowns_faces
        )
        
        timestamps = frames_to_timestamps(matched_frames, frame_rate)
        
        # Save highlighted video
        output_filename = f"output_{os.path.splitext(video_file.filename)[0]}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        save_highlighted_video(video_path, matched_frames, output_path)
        
        return {
            "output_video_url": f"/static/{output_filename}",
            "timestamps": timestamps,
            "matched_faces": matched_frames
        }
        
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_to_database")
async def upload_to_database(
    image_file: UploadFile = File(...), 
    name: str = Form(...), 
    age: int = Form(...)
):
    try:
        #save and process image
        image_path = save_image(image_file, UPLOAD_DIR)
        image = cv2.imread(image_path)
        embedding = extract_feature(image)
        
        if not embedding:
            raise HTTPException(status_code=400, detail="Failed to generate embedding")
            # return JSONResponse(
            #     status_code=400, 
            #     content={"error": "Failed to generate embedding."}
            # )
        
        # entry = {
        #     "name": name,
        #     "age": age,
        #     "embedding": embedding
        # }
        entry = FaceEntry(
            name=name,
            age=age,
            embedding=embedding.tolist()
        )
        save_to_database(entry.dict())
        
        logger.info(f"Added {name} to database")
        return {"message": f"Face for {name} saved successfully"}
    
    except Exception as e:
        logger.error(f"Database upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class FaceEntry(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    age: int = Field(..., gt=0, lt=120)
    embedding: List[float]
    
class ProcessingRequest(BaseModel):
    video_filename: str
    target_image_filename: str

class ProcessingResponse(BaseModel):
    output_video_url: str
    timestamps: Dict[str, List[float]]
    matched_faces: Dict[str, List[Dict[str, Any]]]
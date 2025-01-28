import cv2
from deepface import DeepFace
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

# Extract the frames from the video
def extract_frames(video_path, frame_interval=1):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append((frame_count, frame))
        frame_count += 1
    
    cap.release()
    return frames, frame_rate

# Face Detection with MTCNN library
def detect_faces(frames):
    detector = MTCNN()
    detected_faces = []
    
    for frame_idx, frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb_frame)
        
        for detection in detections:
            box = detection['box']
            x, y, w, h = box
            
            cropped_face = rgb_frame[y:y+h, x:x+w]
            
            detected_faces.append({
                'frame_idx': frame_idx,
                'box': box,
                'confidence': detection['confidence'],
                'image': cropped_face
            })
            
    return detected_faces

# Extract feature using DeepFace to extract embedding in target image and video (frames)
def extract_feature(image):
    try:
        embeddings = DeepFace.represent(image, model_name='Facenet', enforce_detection=False)
        if isinstance(embeddings, list) and len(embeddings) > 0:
            return embeddings[0]['embedding']
        else:
            print("No embeddings generated.")
            return None
    except Exception as e:
        print(f"Error extracting feature: {e}")
        return None

# Face Matching
def match_faces(detected_faces, target_embedding, knowns_faces,  threshold=0.5):
    matched_frames = []
    
    for face in detected_faces:
        face_embedding = extract_feature(face["image"])
        if face_embedding is None:
            continue
        
        # Check if the face matches the target embedding
        target_similarity = cosine_similarity([face_embedding], [target_embedding])[0][0]
        
        if target_similarity > threshold:
            match_info = {
                "frame_idx": face["frame_idx"],
                "box": face["box"]
            }
            
            # Check if the same face matches any known face in the database
            for known_face in knowns_faces:
                db_similarity = cosine_similarity([face_embedding], [known_face["embedding"]])[0][0]
                if db_similarity > threshold:
                    match_info["name"] = known_face["name"]
                    match_info["age"] = known_face["age"]
                    break
            
            matched_frames.append(match_info)
            
    return matched_frames

# Calculate the timestamps
def frames_to_timestamps(matched_frames, frame_rate):
    if isinstance(matched_frames, list):
        frame_indices = [frame['frame_idx'] for frame in matched_frames]
        return [frame_idx / frame_rate for frame_idx in frame_indices]
    else:
        raise ValueError("Expected matched_frames to be a list of frame indices.")

def save_highlighted_video(video_path, detected_faces, output_path):
    # Open the input video file
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize the video writer for the output video
    # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_width, frame_height))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'H264'), fps, (frame_width, frame_height))
    frame_count = 0
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get faces detected in the current frame
        current_frame_faces = [face for face in detected_faces if face.get("frame_idx") == frame_count]
        
        for face in current_frame_faces:
            # Extract the bounding box for the face
            x, y, w, h = face["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Check if the face has name and age to display
            name = face.get("name")
            age = face.get("age")
            if name and age and isinstance(name, str) and (isinstance(age, int) or isinstance(age, str)):
                try:
                    # Clean age if needed
                    age = int(age) if isinstance(age, str) and age.isdigit() else age
                    
                    # Text formatting
                    text = f"{name} ({age})"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), (0, 0, 0), -1)
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                except ValueError as e:
                    print(f"Error formatting text for face: {e}")

        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    




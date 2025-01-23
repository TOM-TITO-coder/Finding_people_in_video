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
def match_faces(target_embedding, detected_faces, threshold=0.5):
    matched_frames = []
    
    for face in detected_faces:
        face_embedding = extract_feature(face["image"])
        similarity = cosine_similarity([target_embedding], [face_embedding])[0][0]
        if similarity > threshold:
            matched_frames.append(face["frame_idx"])
            
    return matched_frames

# Calculate the timestamps
def frames_to_timestamps(matched_frame, frame_rate):
    return [frame_idx / frame_rate for frame_idx in matched_frame]

def save_highlighted_video(video_path, detected_faces, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        for face in detected_faces:
            if face["frame_idx"] == frame_count:
                x, y, w, h = face["box"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()

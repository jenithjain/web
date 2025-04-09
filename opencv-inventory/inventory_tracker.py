import cv2
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
import time
from datetime import datetime
import argparse
import PIL.Image
import io
import mediapipe as mp

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env file")
    exit(1)

# Configure Gemini with safety settings
genai.configure(api_key=api_key)
generation_config = {
    "temperature": 0.1,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 100,
}

try:
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config
    )
except Exception as e:
    print(f"Error initializing Gemini model: {str(e)}")
    exit(1)

class MotionRegion:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.center = (x + w//2, y + h//2)
        self.area = w * h
        self.last_seen = time.time()
        
    def get_direction(self, other):
        dx = other.center[0] - self.center[0]
        dy = other.center[1] - self.center[1]
        
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"

def detect_human_and_motion(frame1, frame2, min_area=2000, threshold=25):
    """Detect both human pose and motion between frames"""
    if frame1 is None or frame2 is None:
        return False, [], None, None
    
    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    
    # Motion detection
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    significant_contours = []
    total_motion_area = 0
    largest_region = None
    max_area = 0
    
    # Get hand landmarks if available
    hand_position = None
    if pose_results.pose_landmarks:
        # Get right hand position (you can also check left hand)
        right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        hand_position = (int(right_wrist.x * frame2.shape[1]), 
                        int(right_wrist.y * frame2.shape[0]))
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            significant_contours.append(contour)
            total_motion_area += area
            
            if area > max_area:
                max_area = area
                x, y, w, h = cv2.boundingRect(contour)
                largest_region = MotionRegion(x, y, w, h)
    
    significant_motion = len(significant_contours) > 0 and total_motion_area > min_area * 2
            
    return significant_motion, significant_contours, largest_region, pose_results

def frame_to_pil(frame):
    """Convert OpenCV frame to PIL Image"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(frame_rgb)

def analyze_image_with_gemini(frame, motion_info=None, hand_near_object=False):
    """Analyze the image using Gemini AI with motion and human context"""
    try:
        pil_image = frame_to_pil(frame)
        
        prompt = """You are an inventory tracking system. Analyze this image and tell me:
1. What item is being interacted with (focus on bottles, items on shelves)
2. The action being performed (picking up, putting down, moving)
3. Where the item is being moved from or to
Keep it very brief (1-2 sentences). Focus on inventory-relevant details."""
        
        context = []
        if motion_info:
            context.append(f"Movement detected {motion_info}")
        if hand_near_object:
            context.append("Hand detected near object")
            
        if context:
            prompt += "\nContext: " + ", ".join(context)
        
        response = model.generate_content([prompt, pil_image])
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {str(e)}")
        return "Error analyzing movement"

def add_analysis_overlay(frame, analysis_text, frame_height):
    """Add a semi-transparent overlay with the analysis text"""
    overlay = frame.copy()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    padding = 15
    
    # Calculate text width to ensure it fits
    lines = analysis_text.split('\n')
    max_width = 0
    for line in lines:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        max_width = max(max_width, text_size[0])
    
    # If text is too wide, break it into multiple lines
    if max_width > frame.shape[1] - 30:
        new_lines = []
        for line in lines:
            words = line.split()
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                test_width = cv2.getTextSize(test_line, font, font_scale, thickness)[0][0]
                if test_width < frame.shape[1] - 40:
                    current_line = test_line
                else:
                    new_lines.append(current_line)
                    current_line = word
            if current_line:
                new_lines.append(current_line)
        lines = new_lines
    
    line_height = int(cv2.getTextSize('A', font, font_scale, thickness)[0][1] * 1.5)
    text_height = line_height * len(lines) + 2 * padding
    
    # Make sure overlay is high enough for all text
    overlay_y = frame_height - text_height - 80
    cv2.rectangle(overlay, (0, overlay_y), (frame.shape[1], overlay_y + text_height + padding),
                 (0, 0, 0), -1)
    
    for i, line in enumerate(lines):
        y = overlay_y + padding + (i + 1) * line_height
        cv2.putText(overlay, line, (padding, y), font, font_scale, (255, 255, 255), thickness)
    
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame

def init_camera():
    """Initialize webcam with multiple attempts"""
    print("Initializing webcam...")
    
    # Try different camera indices
    for camera_index in [0, 1]:
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                continue

            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
                
            # Camera warmup
            print(f"Warming up camera {camera_index}...")
            time.sleep(2)
            
            # Try to read test frames
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"Successfully connected to camera {camera_index}")
                    return cap, 30  # Return camera and fps
                time.sleep(0.1)
            
            cap.release()
            
        except cv2.error as e:
            print(f"Error trying camera {camera_index}: {str(e)}")
            if cap is not None:
                cap.release()
    
    return None, None

def process_feed(source="webcam", video_path=None):
    """Process video feed from either webcam or file"""
    if source == "webcam":
        print("Starting webcam feed...")
        cap, fps = init_camera()
        if cap is None:
            print("Error: Could not initialize webcam")
            return
        total_frames = float('inf')  # Infinite frames for webcam
    else:
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nFeed properties:")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    if source != "webcam":
        print(f"Total frames: {total_frames}")
    print("\nControls:")
    print("SPACE - Pause/Resume")
    print("Q - Quit")
    print("+ - Speed up")
    print("- - Slow down")
    print("\nStarting analysis...\n")

    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    
    if not ret1 or not ret2:
        print("Error: Could not read initial frames")
        cap.release()
        return

    last_analysis_time = time.time()
    analysis_cooldown = 2
    motion_active = False
    stable_frames = 0
    required_stable_frames = 8
    last_region = None
    motion_start_frame = None
    paused = False
    frame_count = 0
    playback_speed = 1.0 if source == "webcam" else 0.5
    last_analysis = "Waiting for movement..."

    while True:
        if not paused:
            frame_delay = int(1000 / (fps * playback_speed))
            
            motion_detected, contours, current_region, pose_results = detect_human_and_motion(frame1, frame2)
            
            current_time = time.time()
            frame_count += 1
            
            if motion_detected:
                if not motion_active:
                    motion_active = True
                    motion_start_frame = frame2.copy()
                    stable_frames = 0
            else:
                if motion_active:
                    stable_frames += 1
                    if stable_frames >= required_stable_frames:
                        if (current_time - last_analysis_time) > analysis_cooldown:
                            motion_info = None
                            if current_region and last_region:
                                direction = last_region.get_direction(current_region)
                                motion_info = f"in {direction} direction"
                            
                            # Check if hand is near the motion region
                            hand_near_object = False
                            if pose_results and pose_results.pose_landmarks:
                                right_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                                hand_x = int(right_wrist.x * frame2.shape[1])
                                hand_y = int(right_wrist.y * frame2.shape[0])
                                
                                if current_region:
                                    # Check if hand is near the motion region
                                    if (hand_x >= current_region.x - 50 and 
                                        hand_x <= current_region.x + current_region.w + 50 and
                                        hand_y >= current_region.y - 50 and 
                                        hand_y <= current_region.y + current_region.h + 50):
                                        hand_near_object = True
                            
                            if motion_start_frame is not None:
                                analysis = analyze_image_with_gemini(frame2, motion_info, hand_near_object)
                                last_analysis = f"Analysis: {analysis}"
                                print(f"[Frame {frame_count}] {analysis}")
                                last_analysis_time = current_time
                        
                        motion_active = False
                        motion_start_frame = None

            if motion_detected:
                last_region = current_region

            frame_display = frame2.copy()
            
            # Draw pose landmarks if detected
            if pose_results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame_display,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
            
            if motion_detected and last_region and current_region:
                direction = last_region.get_direction(current_region)
                cv2.putText(frame_display, 
                           f"Motion: {direction}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 0), 2)
            
            status = "Motion Active" if motion_active else "No Motion"
            if not motion_active and stable_frames > 0 and stable_frames < required_stable_frames:
                status = "Motion Settling..."
                
            # Adjust the position of the status text for better visibility in fullscreen
            cv2.putText(frame_display, 
                       status,
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,  # Increased size and position
                       (0, 255, 0) if motion_active else (0, 0, 255), 
                       2)
            
            # Show frame counter with better positioning
            if source != "webcam":
                counter_text = f"Frame: {frame_count}/{total_frames} | Speed: {playback_speed:.1f}x"
            else:
                counter_text = f"Speed: {playback_speed:.1f}x"
                
            cv2.putText(frame_display,
                       counter_text,
                       (20, frame_height - 30),  # Adjusted position
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.9,  # Increased size
                       (255, 255, 255),
                       2)
            
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)

            frame_display = add_analysis_overlay(frame_display, last_analysis, frame_height)

            cv2.imshow("Inventory Tracking", frame_display)

            frame1 = frame2.copy()
            ret, frame2 = cap.read()
            if not ret:
                if source == "webcam":
                    print("\nCamera disconnected")
                else:
                    print("\nEnd of video reached")
                break

        key = cv2.waitKey(frame_delay if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("Feed paused" if paused else "Feed resumed")
        elif key == ord('+') or key == ord('='):
            playback_speed = min(playback_speed + 0.1, 2.0)
            print(f"Playback speed: {playback_speed:.1f}x")
        elif key == ord('-') or key == ord('_'):
            playback_speed = max(playback_speed - 0.1, 0.1)
            print(f"Playback speed: {playback_speed:.1f}x")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

def main():
    parser = argparse.ArgumentParser(description='Inventory Movement Tracker')
    parser.add_argument('--source', type=str, choices=['webcam', 'video'], default='webcam',
                      help='Source of video feed (webcam or video file)')
    parser.add_argument('--video', type=str, help='Path to video file (required if source is video)', nargs='*')
    args = parser.parse_args()
    
    if args.source == 'video':
        if not args.video:
            print("Error: Video path is required when source is 'video'")
            return
        video_path = ' '.join(args.video)
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
        process_feed(source="video", video_path=video_path)
    else:
        process_feed(source="webcam")

if __name__ == "__main__":
    main()
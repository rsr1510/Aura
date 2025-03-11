import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_posture(frame):
    """ Detects body posture and returns an analysis of the user's engagement level. """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Extract key posture points (shoulders, spine)
        keypoints = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]

        # Example: Simple posture analysis
        shoulder_y_diff = abs(keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] - 
                              keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1])
        
        if shoulder_y_diff > 0.1:  # Threshold for slouching detection
            return "slouched"
        return "upright"
    
    return "unknown"

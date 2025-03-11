import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def detect_hands(frame):
    """ Detects hand keypoints and returns them as a list. """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        return [[(lm.x, lm.y) for lm in hand_landmarks.landmark] for hand_landmarks in results.multi_hand_landmarks]
    return []

def is_finger_spelling(hand_keypoints):
    """
    Determines if the detected hand keypoints correspond to a finger-spelling gesture.
    This can be done by checking for minimal wrist movement and distinct finger placements.
    """
    # Example heuristic: Check if most fingers are extended and separated
    if len(hand_keypoints) == 0:
        return False

    # Extract key points for fingers (index, middle, ring, pinky, thumb)
    fingers = hand_keypoints[0][5:9]  # Indices might need tuning based on Mediapipe
    thumb = hand_keypoints[0][4]

    # Check if fingers are spread apart and thumb is positioned accordingly
    spread_threshold = 0.05  # Adjust based on hand keypoints scale
    finger_spread = sum(abs(fingers[i][0] - fingers[i-1][0]) > spread_threshold for i in range(1, len(fingers)))

    return finger_spread >= 3  # If at least 3 fingers are distinctly apart, assume finger spelling

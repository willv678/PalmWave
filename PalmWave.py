import cv2
import mediapipe as mp
import pyautogui

# Proximity distance between fingers fro a click
CLICK_DISTANCE_THRESHOLD = 25 # Adjust this value if camera isn't detecting / is too sensitive

def perform_click():
    # Super mega nerd emoji geeksforgeeks told me to try the euclidien formula so i did
    pyautogui.click()

def move_mouse(x, y):
    # Move the mouse based on the x and y coordinates
    screen_width, screen_height = pyautogui.size()
    mouse_x = int(x * screen_width)
    mouse_y = int(y * screen_height)
    pyautogui.moveTo(mouse_x, mouse_y)

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)


    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    #I'm not sure about this frame counter thing, cuz it should reduce latency but also it would mean the mouse is a lot less accurate as it pings less often
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        frame_counter += 1

        if frame_counter % 2 != 0:
            continue

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for processing by MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the landmarks for each finger
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Get the pixel coordinates of the finger tips
                thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])

                # Calculate the distance between the index and thumb coordinates
                distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5

                # Print a message when the index and thumb are close
                if distance < CLICK_DISTANCE_THRESHOLD:
                    print("Click detected!")
                    perform_click()

                # Move the mouse based on the index finger movement
                move_mouse(index_tip.x, index_tip.y)

                # Draw circles at the finger tip positions
                cv2.circle(frame, (thumb_x, thumb_y), 5, (0, 255, 0), -1)
                cv2.circle(frame, (index_x, index_y), 5, (0, 255, 0), -1)

        # Display the frame with overlays
        cv2.imshow("Hand Tracking", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

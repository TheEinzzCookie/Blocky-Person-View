import cv2
import mediapipe as mp
import numpy as np

# pip install mediapipe opencv-python numpy
# (optional, falls Videoausgabe als virtuelle Kamera erwünscht:)
# pip install pyvirtualcam

USE_VIRTUAL_CAM = False # VC ist aus
try:
    if USE_VIRTUAL_CAM:
        import pyvirtualcam
except ImportError:
    USE_VIRTUAL_CAM = False

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh

def get_head_tilt_angle(lm, w, h):
    # Winkelschätzung: Ohr zu Ohr
    left = np.array([lm[234].x * w, lm[234].y * h])
    right = np.array([lm[454].x * w, lm[454].y * h])
    delta_y = right[1] - left[1]
    delta_x = right[0] - left[0]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    return angle

def get_mouth_opening(lm, w, h):
    A = np.array([lm[13].x * w, lm[13].y * h])   # Oberlippe
    B = np.array([lm[14].x * w, lm[14].y * h])   # Unterlippe
    opening = np.linalg.norm(A-B)
    return opening

def get_pupil_shift(lm, w, h):
    # Grobe Schätzung der Pupillen-Position im linken/rechten Auge
    left_eye_outer = np.array([lm[33].x * w, lm[33].y * h])
    left_eye_inner = np.array([lm[133].x * w, lm[133].y * h])
    right_eye_inner = np.array([lm[362].x * w, lm[362].y * h])
    right_eye_outer = np.array([lm[263].x * w, lm[263].y * h])
    left_eye_center = (left_eye_outer + left_eye_inner) / 2
    right_eye_center = (right_eye_inner + right_eye_outer) / 2

    # verbesserbar
    return left_eye_center, right_eye_center

def draw_avatar(frame, lm, w, h):    
    nose = lm[1]
    x, y = int(nose.x * w), int(nose.y * h)

    # Kopfneigung
    angle = get_head_tilt_angle(lm, w, h)

    # Avatar-Größe
    size = 150
    vis = np.zeros_like(frame)

    # Blockiger Kopf (gedreht):
    R = cv2.getRotationMatrix2D((x, y), angle, 1)
    rect = np.array([
        [x - size // 2, y - size // 2, 1],
        [x + size // 2, y - size // 2, 1],
        [x + size // 2, y + size // 2, 1],
        [x - size // 2, y + size // 2, 1]
    ])
    rect_rot = np.dot(rect, R.T).astype(int)
    cv2.fillConvexPoly(vis, rect_rot, (0, 255, 120))

    # Augen
    eye_y = int(y - size // 5)
    left_eye_x = int(x - size // 4)
    right_eye_x = int(x + size // 4)
    left_eye_center, right_eye_center = get_pupil_shift(lm, w, h)
    # Pupillen (angepasst)
    for ex, ey, ecenter in [(left_eye_x, eye_y, left_eye_center), (right_eye_x, eye_y, right_eye_center)]:
        cv2.circle(vis, (ex, ey), 20, (255,255,255), -1)
        # Berechne horizontalen Pupillen-Shift
        shift = int((ecenter[0] - ex) * 0.7)
        cv2.circle(vis, (ex + shift, ey), 8, (0,0,0), -1)

    # Mund
    mouth_height = get_mouth_opening(lm, w, h)
    if mouth_height > 20:
        cv2.ellipse(vis, (x, y + size//4), (40,25), 0, 0, 360, (0,0,0), -1)
    else:
        cv2.ellipse(vis, (x, y + size//4), (40,10), 0, 0, 180, (0,0,0), 8)

    return vis

def main():
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
        frame_w, frame_h = 640, 480

        if USE_VIRTUAL_CAM:
            import pyvirtualcam
            cam = pyvirtualcam.Camera(width=frame_w, height=frame_h, fps=30)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_disp = np.zeros_like(frame)

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                vis = draw_avatar(frame, lm, w, h)
            else:
                vis = frame_disp

            cv2.imshow("Avatar Demo", vis)
            if USE_VIRTUAL_CAM:
                cam.send(vis)
                cam.sleep_until_next_frame()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


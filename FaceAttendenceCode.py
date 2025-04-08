import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW to fix camera issue

known_faces = {
    "sir": r"photos\sir.jpg",
    "messi": r"photos\messi.jpg",
    "rock": r"photos\rock.jpg",
    "ronaldo": r"photos\ronaldo.jpg"
}

known_face_encodings = []
known_face_names = []

for name, path in known_faces.items():
    image = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

students = known_face_names.copy()

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame. Check your camera.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = []

    if face_locations:
        try:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        except Exception as e:
            print("Encoding error:", e)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        name = ""

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        if name and name in students:
            students.remove(name)
            now = datetime.now()
            lnwriter.writerow([name, now.strftime("%H:%M:%S")])
            print(f"{name} marked present at {now.strftime('%H:%M:%S')}")

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

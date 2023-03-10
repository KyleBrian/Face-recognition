import cv2

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load the trained model for facial recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_model.yml")

# Start the webcam
cap = cv2.VideoCapture(0)

# Continuously capture frames from the webcam
while True:
    # Read the frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate over the faces detected
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Get the region of interest (ROI) for the face
        roi_gray = gray[y:y + h, x:x + w]

        # Use the trained model to predict the label for the ROI
        label, confidence = face_recognizer.predict(roi_gray)

        # Check if the confidence is below a certain threshold
        if confidence < 50:
            # If the confidence is low, assume the face is recognized
            # and start the AI assistant
            speak("Face recognized, starting the AI assistant...")
            # Perform the other functions and tasks of the AI assistant here
        else:
            # If the confidence is high, assume the face is not recognized
            # and display a message
            cv2.putText(frame, "Unknown Face", (x, y), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)

    # Display the frame with the faces detected
    cv2.imshow("Face Recognition", frame)

    # Check if the user pressed the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
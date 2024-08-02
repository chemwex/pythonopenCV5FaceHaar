# pythonopenCV5FaceHaar
We use Haar Cascades to identify facial features like mouth, nose, eyes  and entire face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#These lines load pre-trained Haar cascades for face and eye detection
cap = cv2.VideoCapture(0)                              #initialize the camera
while cap.isOpened():                                  #capture frames from the webcam
    success, image = cap.read()        
    if not success:
        print("Ignoring empty camera frame.")
        continue
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          #convert image to grascale
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))    #detect faces
scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
minSize: Minimum possible object size. Objects smaller than this are ignored.
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)       #for each detected face, eyes etc. draw a rectangle (lines 17-24)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        cv2.imshow('Face and Eye Detection', image)                                 #Display Image:
if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()                                                          #Release camera
cv2.destroyAllWindows()
        

import cv2
import numpy as np
import mediapipe.python.solutions.hands as hands
import mediapipe.python.solutions.drawing_utils as du
import tensorflow as tf
import keras

labels=["Open","Closed","Victory","Thumbs Up","Thumbs Down"]

def main():
    cap = cv2.VideoCapture(0)
    capture_hands=hands.Hands()

    model = keras.models.load_model("training/gesturedetector.keras")

    while True:
        ret,frame = cap.read()
        w=int(cap.get(3))
        h=int(cap.get(4))

        frame = cv2.flip(frame,1)

        rgbframe = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        output_hands = capture_hands.process(rgbframe)
        all_hands = output_hands.multi_hand_landmarks


        if all_hands:
            for hand in all_hands:
                label=0

                du.draw_landmarks(frame,hand,hands.HAND_CONNECTIONS)
                x1,y1,x2,y2 = bounding_box(hand.landmark,w,h)

                data = create_data(hand.landmark,w,h,x2-x1,y2-y1)
                predict_data = np.array([data])

                label = np.argmax(model.predict(predict_data))

                x1-=10;y1-=10;x2+=10;y2+=10
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(frame,(x1,y1-50),(x2,y1),(0,255,0),-1)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(frame,labels[label],(x1,y1-10),font,1,(255,0,0),1,cv2.LINE_AA)

        cv2.imshow("Frame",frame)

        if cv2.waitKey(1) == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


def bounding_box(landmarks,w,h):
    x1=w
    y1=h
    x2=y2=0

    for i in range(21):
        x1=int(min(x1,w*landmarks[i].x))
        x2=int(max(x2,w*landmarks[i].x))
        y1=int(min(y1,h*landmarks[i].y))
        y2=int(max(y2,h*landmarks[i].y))
    
    return (x1,y1,x2,y2)

def create_data(landmarks,w1,h1,w2,h2):
    lms = landmarks
    data=[]
    x0=lms[0].x
    y0=lms[0].y

    for i in range(1,21):
        lms[i].x-=x0
        lms[i].y-=y0

        lms[i].x *= w1/w2
        lms[i].y *= h1/h2

        data.append(lms[i].x)
        data.append(lms[i].y)

    return data   

if __name__=="__main__":
    main()
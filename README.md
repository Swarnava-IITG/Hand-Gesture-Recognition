# Project Description:
 This project uses opencv and mediapipe for hand landmark tracking, converts it into data feedable into a neural network
 and then uses it to train a keras ml model which is used to recognize the hand gestures.

 Model is already trained to recognize 5 gestures: Fist open, Fist closed, Victory sign, Thumbs Up and Thumbs Down

## Installations:
 To use the project you have to install all the libraires mentioned in "libraryrequirements.txt" file

## Training model:
 1. In training folder open "data_collection.py" and increase the value of the variable data_label by 1
 2. Run "data_collection.py" and press s to start saving data (A saving data text also appers on screen), use your hand to show a gesture of your choice for some time (move your hand a bit while training for better results)
 3. Press s again to stop collecting data and press q to exit (you may need to press and hold a bit)
 4. open "ml_model.ipynb" and increase the value of total_recognitions variable by 1 and run all the cells
 5. open "gesture_dec.py" in gesture_detection folder and add the name of you gesture at the last of labels list
 6. now run gesture_dec.py, the model should now be able to recognize your gesture

## Using model:
 Run "gesture_dec.py" in gesture_detection folder , press q to quit
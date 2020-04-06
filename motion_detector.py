import cv2
import pandas
from datetime import datetime


first_frame = None
status_list = [None, None]
times = []
df = pandas.DataFrame(columns=["Start", "End"])

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    check, frame = video.read()
    status = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur the image 'smooth and remove the noise'
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
# (21, 21) width and height of the Gassuian
# 0 standard deviation, Both of them are commonly used

    if first_frame is None:
        first_frame = gray
        continue  # to the beginning of the loop

    delta_frame = cv2.absdiff(first_frame, gray)
    '''
    Classify Values of the delta frame, assign a threshold..
    if the difference between first frame and the current frame is more than 30
    we will classify that as white >> 255,
    less than 30  we assign a black pixel >> 0
    threshhold method returns a tuble the first value neede with otherthreshold
    methods, but the second value for thresh_binary "the actual frame"
    '''
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

# make the frame more smooth, instead of none if you have a kernal array
# pass it there, iterate 2 times through the img to remove holes
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # create contours of the white frame in the threshframe
    contours, hierarchy = cv2.findContours(thresh_frame.copy(),   # noqa ignore=E231
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # filter out these contours, keep counters have an area > 1000
    for i in contours:
        if cv2.contourArea(i) < 10000:
            continue
        status = 1
        # draw the rectangle around that countour
        (x, y, w, h) = cv2.boundingRect(i)  # getting coordinates from i
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow('Gray Frame', gray)
    cv2.imshow("Detla Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow('Color Frame', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

print(status_list)
print(times)

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i+1]}, ignore_index=True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()

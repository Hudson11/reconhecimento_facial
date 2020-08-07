import cv2
import os

def detect(img, classifier):
    min_rectangle = (50, 50)

    rects = classifier.detectMultiScale(img, 1.2, 3, minSize=min_rectangle)

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    return rects, img

def crop(img, startx, endx, starty, endy):
    return img[starty:endy, startx:endx]

def get_imagens(video_path=None, dir_number=0):
    if os.path.isdir('dataset'):
        pass
    else:
        os.mkdir('dataset')

    if os.path.isdir(f'dataset/{dir_number}'):
        pass
    else:
        os.mkdir(f'dataset/{dir_number}')
        pass

    if video_path is not None:
        classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

        cap = cv2.VideoCapture(video_path)
        img_count = 0

        while cap.isOpened():
            if img_count >= 400:
                break
            else:
                res, frame = cap.read()
                (rects, frame) = detect(frame, classifier)
                if len(rects) != 0:
                    f1x1, f1y1, f1x2, f1y2 = rects[0]
                    img = crop(frame, f1x1, f1x2, f1y1, f1y2)

                    path_img = f'dataset/{str(dir_number)}/img{img_count}.jpg'
                    cv2.imwrite(path_img, img)
                    img_count += 1
    cap.release()
        


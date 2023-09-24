# Подключение библиотек
from tkinter import *
from tkinter import messagebox
import cv2, os
import numpy as np
from PIL import Image
import time

# Инициализация переменных
path             = '.\myfaces'
cascadePath      = "haarcascade_frontalface_default.xml"
faceCascade      = cv2.CascadeClassifier(cascadePath)
confidence_limit = 123 
recognizer       = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, confidence_limit)
video_capture    = cv2.VideoCapture(0)
listOfProfiles   = [[],[]] # Список профилей: 0 - порядковый номер, 1 - имя
indexOfProfile   = 0       # Порядковый номер для listOfProfiles

# Восстановление данных о профилях
file = open("listOfProfiles.txt", "r")
indexOfProfile = int(file.readline())
for i in range(indexOfProfile):
    listOfProfiles[0].append(int(file.readline()))
    listOfProfiles[1].append(file.readline().replace("\n",""))
file.close()

# Функция для поиска изображений для обучения
def get_images(path):
    # Ищем все фотографии и записываем их в image_paths
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    
    images = []
    labels = []

    for image_path in image_paths:
        # Переводим изображение в черно-белый формат и приводим его к формату массива
        gray = Image.open(image_path).convert('L')
        image = np.array(gray, 'uint8')

        # Из каждого имени файла извлекаем номер человека, изображенного на фото
        subject_number = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))

        # Определяем области где есть лица
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Если лицо нашлось добавляем его в список images, а соответствующий ему номер в список labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(subject_number)
    return images, labels

# Функция показывающая изображение с веб-камеры
def show(path):
    global video_capture, faceCascade, recognizer, cascadePath, confidence_limit, listOfProfiles
    if video_capture.isOpened() == False:
        video_capture.open(0)
    # Обучение
    images, labels = get_images(path)
    recognizer.train(images, np.array(labels))
    # Цикл, в котором работает веб-камера
    while True:
        # Обновление кадра
        result, video_frame = video_capture.read()
        if result is False:
            print("broke")
            break
        
        # Перевод изображения из cv2 в pil
        color_converted = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(color_converted)
        image = np.array(pil_image, 'uint8')

        # Определение лица на кадре
        faces = faceCascade.detectMultiScale(video_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Если лица найдены, пытаемся распознать их
            # Функция  recognizer.predict в случае успешного расознавания возвращает номер и параметр confidence,
            # этот параметр указывает на уверенность алгоритма, что это именно тот человек, чем он меньше, тем больше уверенность
            number_predicted, conf = recognizer.predict(image[y: y + h, x: x + w])

            # Извлекаем настоящий номер человека на фото и сравниваем с тем, что выдал алгоритм
            for i in range(len(listOfProfiles[0])):
                if listOfProfiles[0][i] == number_predicted:
                    print("{} is correctly recognized with confidence {}".format(listOfProfiles[0][i], conf))
                    cv2.rectangle(
                        video_frame, 
                        (x, y), 
                        (x+w, y+h), 
                        (255-(255/(i+1)), 255-(255/(10*(i+1))), 255-(255/(50*(i+1)))), 
                        2
                    )
                    cv2.putText(
                        video_frame, 
                        listOfProfiles[1][listOfProfiles[0][i]], 
                        (x, y+h),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255-(255/(i+1)), 255-(255/(10*(i+1))), 255-(255/(50*(i+1)))),
                        2
                    )
        cv2.putText(
            video_frame,
            "Press Q to exit",
            (0, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3
        )
        cv2.imshow("Recongnizing Face", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Функция, отрисовывающая прямоугольник вокруг лица
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray_image, 1.1, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(vid, (x,y), (x+w, y+h), (0,255,0), 4)
    return faces

#Функция записывающа лица для обучения
def saveFaces(name,window_to_destroy):
    global listOfProfiles, indexOfProfile
    listOfProfiles[0].append(indexOfProfile)
    listOfProfiles[1].append(name)
    indexOfProfile = indexOfProfile + 1

    if video_capture.isOpened() == False:
        video_capture.open(0)

    timing = time.time()

    while time.time() - timing < 20:
        result, video_frame = video_capture.read()
        if result is False:
            break
        
        if int(time.time()*10) % 5 == 0:
            filename = "myfaces\subject" + str(indexOfProfile-1) + ".face" + str(int((time.time()-timing)*10)) + ".png"
            cv2.imwrite(filename, video_frame)
        print(listOfProfiles[1][indexOfProfile-1])

        faces = detect_bounding_box(video_frame)

        cv2.imshow("Record training data...", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    window_to_destroy.destroy()

# Функция добавления нового профиля
def addProfile():
    profile = Tk()
    profile.title("Add new profile")
    profile.geometry("600x250")

    p_frame = Frame(
        profile,
        padx = 10,
        pady = 10
    )
    p_frame.pack(expand = True)

    p_text = Label(
        p_frame,
        text = "Input profile name:"
    )
    p_text.grid(row = 1, column = 1)

    p_name = Entry(
        p_frame
    )
    p_name.grid(row = 2, column = 1)

    p_about = Label(
        p_frame,
        text = "Input name of new profile, and press continue. Then you'll must show your face for 20 seconds."
    )
    p_about.grid(row = 3, column = 1)

    p_about2 = Label(
        p_frame,
        text = "Try to show all diifferent variants of your face. Your face always must be visible for camera!"
    )
    p_about2.grid(row = 4, column = 1)

    p_button = Button(
        p_frame,
        text = "Continue",
        command = lambda: saveFaces(p_name.get(),profile)
    )
    p_button.grid(row = 5, column = 1)

def DeleteAll():
    global path,indexOfProfile,listOfProfiles
    indexOfProfile = 0
    listOfProfiles = [[],[]]

    for f in os.listdir(path):
        os.remove(os.path.join(path, f))


# Вызов окна
window = Tk()
window.geometry('500x250')

frame = Frame(
    window,
    padx = 10,
    pady = 10
)
frame.pack(expand = True)

btn_profile = Button(
    frame,
    text = "Make new profile",
    command = addProfile
)
btn_profile.grid(row = 1, column = 1)

btn_camera = Button(
    frame,
    text = "Recognition in real time",
    command = lambda: show(path)
)
btn_camera.grid(row = 2, column = 1, pady = 10)

btn_delete = Button(
    frame,
    text = "Delete all profiles",
    command = DeleteAll
)
btn_delete.grid(row = 3, column = 1)

window.mainloop()

# Сохранение данных о профилях
file = open("listOfProfiles.txt", "w")
file.write(str(indexOfProfile)+"\n")
if indexOfProfile != 0:
    for i in range(indexOfProfile):
        file.write(str(listOfProfiles[0][i]) + "\n" + str(listOfProfiles[1][i]) + "\n")
file.close()
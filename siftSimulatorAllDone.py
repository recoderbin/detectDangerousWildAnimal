import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys, os
from playsound import playsound
import threading
from PyQt5.QtGui import QPixmap

class TrafficWild(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SIFT를 활용한 야생동물 인식 시스템')
        self.setGeometry(200, 200, 700, 780)
    
        self.image_label = QLabel(self)
        self.image_label.setPixmap(QPixmap("noPig.PNG").scaledToWidth(600))
        self.image_label.setGeometry(10, 10, 680, 680)

        signButton = QPushButton('이미지 등록', self)
        roadButton = QPushButton('영상 불러오기', self)
        webcamButton = QPushButton('웹캠 실행', self)
        recognitionButton = QPushButton('인식', self)
        quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)

        signButton.setGeometry(10, 680, 120, 40)
        roadButton.setGeometry(140, 680, 120, 40)
        webcamButton.setGeometry(270, 680, 120, 40)
        recognitionButton.setGeometry(400, 680, 120, 40)
        quitButton.setGeometry(530, 680, 120, 40)
        self.label.setGeometry(10, 730, 680, 30)

        signButton.clicked.connect(self.signFunction)
        roadButton.clicked.connect(self.roadFunction)
        webcamButton.clicked.connect(self.webcamFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.classes = [
            ['boar', '멧돼지'], 
            ['person', '사람'], 
            ['deer', '고라니'], 
            ['magpie', '까치']
        ]
        self.classImgs = []
        self.kdList = []
        self.sift = cv.SIFT_create()
        self.cap = None

    def signFunction(self):
        self.label.setText('이미지를 등록 중입니다...')
        self.classImgs.clear()
        self.kdList.clear()

        for folder, label in self.classes:
            imgs, kd = [], []
            path = os.path.join(os.getcwd(), folder)
            if not os.path.exists(path):
                self.label.setText(f"폴더 {folder} 없음")
                return

            for file in os.listdir(path):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    fpath = os.path.join(path, file)
                    img = cv.imread(fpath)
                    if img is None:
                        continue
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    kp, des = self.sift.detectAndCompute(gray, None)
                    imgs.append(img)
                    kd.append((kp, des))
            self.classImgs.append(imgs)
            self.kdList.append(kd)

        self.label.setText('이미지 등록이 완료되었습니다.')

    def roadFunction(self):
        if not self.kdList:
            self.label.setText('먼저 이미지를 등록하세요.')
            return

        fname = QFileDialog.getOpenFileName(self, '영상 선택', './')[0]
        if fname:
            self.cap = cv.VideoCapture(fname)
            if not self.cap.isOpened():
                self.label.setText('동영상을 불러올 수 없습니다.')
                return
            self.label.setText('동영상이 로드되었습니다. 인식 버튼을 눌러주세요.')
        else:
            self.label.setText('영상 선택이 취소되었습니다.')

    def webcamFunction(self):
        if not self.kdList:
            self.label.setText('먼저 이미지를 등록하세요.')
            return

        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            self.label.setText('웹캠을 열 수 없습니다.')
            return
        self.label.setText('웹캠이 시작되었습니다. 인식 버튼을 눌러주세요.')
        
    def recognitionFunction(self):
        if self.cap is None or not self.cap.isOpened():
            self.label.setText('먼저 동영상이나 웹캠을 실행하세요.')
            return

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            result_img, msg = self.detectAnimal(frame)
            self.label.setText(msg)

             # 결과 이미지 크기 조정 (예: 폭 800px 기준으로 축소)
            display_img = cv.resize(result_img, (800, int(result_img.shape[0] * 800 / result_img.shape[1])))

            cv.imshow('인식 결과', display_img)

            key = cv.waitKey(8000)

            self.cap.release()
            cv.destroyAllWindows()

    def detectAnimal(self, frame):
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        kp2, des2 = self.sift.detectAndCompute(gray, None)

        best_matches = []
        for kd_class, img_class in zip(self.kdList, self.classImgs):
            max_good, best_pair, best_img = 0, None, None
            for i, (sign_kp, sign_des) in enumerate(kd_class):
                if sign_des is None or des2 is None:
                    continue
                matches = matcher.knnMatch(sign_des, des2, 2)
                good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.6 * m[1].distance]
                if len(good) > max_good:
                    max_good = len(good)
                    best_pair = (sign_kp, good, sign_des)
                    best_img = img_class[i]
            best_matches.append((max_good, best_pair, best_img))

        best_index = np.argmax([bm[0] for bm in best_matches])
        max_count, best_data, best_img = best_matches[best_index]

        if best_data is None or max_count < 4:
            frame_with_kp = cv.drawKeypoints(frame, kp2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return frame_with_kp, "인식된 대상이 없습니다."

        sign_kp, good_match, sign_des = best_data

        # 매칭 결과 시각화
        match_img = cv.drawMatches(
            best_img, sign_kp,
            frame, kp2,
            good_match[:20], None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        label = self.classes[best_index][1]
        msg = f"{label}가 감지되었습니다. 주의하세요!"

        # 경고음 재생
        if self.classes[best_index][0] in ['boar', 'deer', 'magpie']:
            threading.Thread(target=self.play_warning_mp3).start()
        elif self.classes[best_index][0] == 'person':
            threading.Thread(target=self.play_manual_mp3).start()

        return match_img, msg

    def play_warning_mp3(self):
        audio_file = os.path.join(os.getcwd(), "warning.mp3")
        if os.path.exists(audio_file):
            playsound(audio_file)
        else:
            print("warning.mp3 파일이 없습니다.")

    def play_manual_mp3(self):
        audio_file = os.path.join(os.getcwd(), "manual.mp3")
        if os.path.exists(audio_file):
            playsound(audio_file)
        else:
            print("manual.mp3 파일이 없습니다.")

    def quitFunction(self):
        if self.cap:
            self.cap.release()
        cv.destroyAllWindows()
        self.close()


app = QApplication(sys.argv)
win = TrafficWild()
win.show()
app.exec_()

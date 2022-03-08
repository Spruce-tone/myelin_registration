import sys, time, os
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, \
                            QDesktopWidget, QHBoxLayout, QLabel, \
                            QWidget, QLineEdit, QVBoxLayout, QPushButton,\
                            QMessageBox
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QPixmap
from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
from numpy.random import uniform
from pkg.trace_stack_regi import search_regi_files, segment_crop, regi_cropped_img

class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._layout()

            
    def _layout(self):
        self.vbox = QVBoxLayout()

        self.vbox.addStretch(3)

        self.label = QLabel('약속된 암호를 입력하시오')
        self.password = QLineEdit()
        self.enter_btn = QPushButton('Enter')
        self.enter_btn.clicked.connect(self.random_Gacha)

        self.vbox.addWidget(self.label)
        self.vbox.addWidget(self.password)
        self.vbox.addWidget(self.enter_btn)

        self.vbox.addStretch(3)

        self.hbox = QHBoxLayout()
        self.hbox.addStretch(3)
        self.hbox.addLayout(self.vbox)
        self.hbox.addStretch(3)

        self.setLayout(self.hbox)

    @pyqtSlot()
    def random_Gacha(self):
        if self.password.text()=='':
            self.label.setText('제대로 입력하세요')
        else:
            good_luck = np.floor(uniform(0, 1000))

            if (good_luck >= 7) and (good_luck < 9):
                open_text_image('린저씨 축하해여 ㅎㅎ. 0.2%를 뚫었군여! Registration 시작!')
                file_path_regi = search_regi_files(raw_path='./data', regi_path='./registration')
                segment_crop(file_path_regi)
                regi_cropped_img()
            elif ((good_luck >= 0) and (good_luck < 7)) or ((good_luck >= 9) and (good_luck < 200)):
                open_text_image('비밀번호 그거 아닌데~~~')
                time.sleep(2)
                cv2.destroyAllWindows()
            elif (good_luck >= 200) and (good_luck < 600):
                open_text_image('천둥의 호흡, 제 일의 형. 벽력일섬 6연!')
                for idx, img_name in enumerate(os.listdir('text image')):
                    img = cv2.imread(os.path.join(f'text image/{img_name}'))
                    cv2.namedWindow(str(idx))
                    cv2.resizeWindow(str(idx), 1920, 1080)
                    cv2.moveWindow(str(idx), 700, 700)
                    cv2.imshow(str(idx), np.array(img))
                    cv2.waitKey(25)
                    time.sleep(1/40)
                
                time.sleep(5)
                cv2.destroyAllWindows()
            elif (good_luck >= 600) and (good_luck < 700):
                open_text_image(self.password.text())
                time.sleep(2)
                cv2.destroyAllWindows()
            elif (good_luck >= 700) and (good_luck < 900):
                open_text_image('똥쟁이똥쟁이똥쟁이똥쟁이에베베베베베벱베베베베베')
                time.sleep(2)
                cv2.destroyAllWindows()
            elif (good_luck >= 900) and (good_luck < 1000):
                open_text_image('이성호씨가 좋아하는 랜덤 가챠가챠. 레지스트레이션 확률은 과연 몇%??')
                time.sleep(2)
                cv2.destroyAllWindows()

def open_text_image(text: str):
    position = [(100, 100), (3000, 100), (100, 1400), (3000, 1400)]
    for idx, i in enumerate(text):
        img, W, H = drawtextimage(i)
        cv2.namedWindow(str(idx))
        cv2.resizeWindow(str(idx), W, H)
        cv2.moveWindow(str(idx), position[idx%4][0], position[idx%4][1])
        cv2.imshow(str(idx), np.array(img))
        cv2.waitKey(25)
        time.sleep(0.25)
    time.sleep(2)
    cv2.destroyAllWindows()


def drawtextimage(word: str):
        color = ['black', 'red', 'green', 'blue', 'magenta']
        randcolor = int(np.floor(uniform(0, 5)))
        #배경 이미지의 크기
        W, H = (600, 600) 
        #배경 이미지를 흰색으로 하여 생성
        image = Image.new('RGB', (W, H), (255, 255, 255)) 
        #해당 폰트와 사이즈 설정
        font = ImageFont.truetype('seoulfont.ttf', size=600) 
        #이미지 생성
        draw = ImageDraw.Draw(image) 
        #글자의 크기
        w, h = draw.textsize(word, font=font) 
        #배경 이미지와 글자의 크기를 이용하여 배경 중간에 글자 배치
        draw.text(((W - w) / 2, (H - h) / 2), word, fill=color[randcolor], font=font) 
        #이미지 저장
        # image.save(f"{title}{e}.jpg")
        return image, W, H

            


class pupil(QMainWindow):    
    def __init__(self, height=500, width=500):
        super().__init__()
        self.H = 700    
        self.W = 700
        
        self.initUI()
        self.main_widget = MainWidget()
        self.setCentralWidget(self.main_widget)

    def initUI(self):
        self.setWindowTitle('Pupilometry')
        self._main_windowsize()
        self._windowcenter()
        self.statusBar().showMessage('Initialize')
        self._menubar()
        self.show()

    def _main_windowsize(self):
        # self.setGeometry(300, 300, 400, 400) # (x, y, width, height) of window
        # top left=(0, 0)
        # as go from left to right, x increases
        # as go from top to bottom, y increases
        self.resize(self.W, self.H)

    def _windowcenter(self):
        window_geometry = self.frameGeometry()
        monitor_center = QDesktopWidget().availableGeometry().center()
        window_geometry.moveCenter(monitor_center)
        self.move(window_geometry.topLeft())

    def _menubar(self):
        self.mainMenu = self.menuBar()
        self.mainMenu.setNativeMenuBar(False)

        # File menu
        self.fileMenu = self.mainMenu.addMenu('&File')

        # Exit menue
        exitAction = QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        self.fileMenu.addAction(exitAction)



if __name__=='__main__':
    app = QApplication(sys.argv)
    ex = pupil()
    sys.exit(app.exec_())

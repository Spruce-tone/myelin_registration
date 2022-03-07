import sys, time
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, \
                            QDesktopWidget, QHBoxLayout, QLabel, \
                            QWidget, QLineEdit, QVBoxLayout, QPushButton,\
                            QMessageBox
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QPixmap
import cv2

import numpy as np
from numpy.random import uniform
from trace_stack_regi import search_regi_files, segment_crop, regi_cropped_img

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
            for i in range(3):
                cv2.namedWindow(str(i))
                cv2.resizeWindow(str(i), i*100, i*100)
                img = cv2.imread("aaa.jpg")
                cv2.moveWindow(str(i), 300*i, 100)
                cv2.imshow(str(i), img)

            if (good_luck >= 7) and (good_luck < 9):
                self.label.setText('린저씨 축하해여 ㅎㅎ. 0.2%를 뚫었군여!')
                file_path_regi = search_regi_files(raw_path='./data', regi_path='./registration')
                segment_crop(file_path_regi)
                regi_cropped_img()
            elif ((good_luck >= 0) and (good_luck < 7)) or ((good_luck >= 9) and (good_luck < 300)):
                self.label.setText('비밀번호 그거 아닌데~')
                time.sleep(10)


            
            time.sleep(1)

    def img_dialog(self):
        self.winTable = WinTable()
        self.winTable.show()
            


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

#! /usr/bin/env python

# -*- coding:utf-8 -*-


import sys

from PyQt5.QtWidgets import QApplication, QWidget

from PyQt5.QtCore import Qt
import sys

from PyQt5.QtCore import *

from PyQt5.QtGui import *

from PyQt5.QtWidgets import *


class NoBorderWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.window_UI()

        # self.qss()

    def window_UI(self):
        # self.resize(3840, 2160)
        # self.move(-3840, 3+1080 - 2160)

        self.resize(1920, 1080)
        self.move(0, 0)


        # self.move(1-3840,1+ 1080-2160)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.use_palette()

    def use_palette(self):
        # self.setWindowTitle("设置背景图片")
        window_pale = QPalette()
        window_pale.setBrush(self.backgroundRole(), QBrush(QPixmap(r"D:\sysDef\Documents\GitHub\pytest\cv\全屏\test2.png")))
        self.setPalette(window_pale)

    # def qss(self):
    #     self.qssfile = "./qss/noborder.qss"
    #
    #     self.style = CommonStyleSheet.loadqss(self.qssfile)
    #
    #     self.setStyleSheet(self.style)


class CommonStyleSheet:

    def __init__(self):
        pass

    @staticmethod
    def loadqss(style):
        with open(style, "r", encoding="utf-8") as f:
            return f.read()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    win = NoBorderWindow()
    #
    # lab1 = QLabel()
    #
    # lab1.setPixmap(QPixmap("./images/python.png"))
    # win.addWidget(lab1)

    win.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)

    win.show()

    # win.activateWindow()
    # win.setWindowState(win.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
    # win.showNormal()


    sys.exit(app.exec_())
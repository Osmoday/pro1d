from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import apriori


class Window(QtWidgets.QOpenGLWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        btn = QPushButton('Button', self)
        btn.resize(btn.sizeHint())
        self.Painter = QtGui.QPainter(self)

    def paintEvent(self, e):
        pass


class Application(QtWidgets.QApplication):
    def __init__(self, *args):
        super().__init__(*args)


app = QtWidgets.QApplication(sys.argv)
# flag = QtCore.Qt.Widget
# w = QtWidgets.QOpenGLWidget()
w = Window(flags=QtCore.Qt.Window)
w.show()
sys.exit(app.exec_())
# w = QtWidgets.QWidget()
# w.show()


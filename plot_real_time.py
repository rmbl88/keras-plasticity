# import random
# import sys
# import pyqtgraph as pg
# import time
# from PyQt5 import QtWidgets, QtCore

# class MyThread(QtCore.QThread):
#     signal = QtCore.pyqtSignal(object)
#     def __init__(self, parent=None):
#         super(MyThread, self).__init__(parent=parent)
#         self.data = [0]

#     def __del__(self):
#         self.exiting = True
#         self.wait()

#     def run(self):
#         while True:
#             self.data.append(random.random())
#             self.signal.emit(self.data)
#             time.sleep(0.1)


# class PlottingWidget(QtWidgets.QWidget):
#     def __init__(self, parent=None):
#         super(PlottingWidget, self).__init__(parent)
#         self.myThread = MyThread()
#         layout = QtWidgets.QHBoxLayout()
#         self.button = QtWidgets.QPushButton('Start Plotting')
#         layout.addWidget(self.button)
#         self.plot = pg.PlotWidget()
#         layout.addWidget(self.plot)
#         self.setLayout(layout)
#         self.curve = self.plot.getPlotItem().plot()
#         self.button.clicked.connect(self.start)


#     def plotter(self, data):
#         self.curve.setData(data)

#     def start(self):
#         self.myThread.start()
#         self.myThread.signal.connect(self.plotter)


# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self, parent=None):
#         super(MainWindow, self).__init__(parent)
#         self.centralwidget = QtWidgets.QWidget(self)
#         self.setCentralWidget(self.centralwidget)
#         self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
#         self.login_widget_1 = PlottingWidget(self)
#         self.horizontalLayout.addWidget(self.login_widget_1)
#         # self.login_widget_2 = PlottingWidget(self)
#         # self.horizontalLayout.addWidget(self.login_widget_2)
#         # self.setCentralWidget(self.centralwidget)

# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     w = MainWindow()
#     w.show()
#     sys.exit(app.exec_())


# import sys
# from time import sleep
# from PyQt5.QtCore import QObject, QThread, pyqtSignal

# from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import (
#     QApplication,
#     QLabel,
#     QMainWindow,
#     QPushButton,
#     QVBoxLayout,
#     QWidget,
# )

# # Step 1: Create a worker class
# class Worker(QObject):
#     finished = pyqtSignal()
#     progress = pyqtSignal(int)

#     def run(self):
#         """Long-running task."""
#         for i in range(5):
#             sleep(1)
#             self.progress.emit(i + 1)
#         self.finished.emit()

# class Window(QMainWindow):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.clicksCount = 0
#         self.setupUi()

#     def setupUi(self):
#         self.setWindowTitle("Freezing GUI")
#         self.resize(300, 150)
#         self.centralWidget = QWidget()
#         self.setCentralWidget(self.centralWidget)
#         # Create and connect widgets
#         self.clicksLabel = QLabel("Counting: 0 clicks", self)
#         self.clicksLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
#         self.stepLabel = QLabel("Long-Running Step: 0")
#         self.stepLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
#         self.countBtn = QPushButton("Click me!", self)
#         self.countBtn.clicked.connect(self.countClicks)
#         self.longRunningBtn = QPushButton("Long-Running Task!", self)
#         self.longRunningBtn.clicked.connect(self.runLongTask)
#         # Set the layout
#         layout = QVBoxLayout()
#         layout.addWidget(self.clicksLabel)
#         layout.addWidget(self.countBtn)
#         layout.addStretch()
#         layout.addWidget(self.stepLabel)
#         layout.addWidget(self.longRunningBtn)
#         self.centralWidget.setLayout(layout)

#     def countClicks(self):
#         self.clicksCount += 1
#         self.clicksLabel.setText(f"Counting: {self.clicksCount} clicks")

#     def reportProgress(self, n):
#         self.stepLabel.setText(f"Long-Running Step: {n}")

#     def runLongTask(self):
#         # Step 2: Create a QThread object
#         self.thread = QThread()
#         # Step 3: Create a worker object
#         self.worker = Worker()
#         # Step 4: Move worker to the thread
#         self.worker.moveToThread(self.thread)
#         # Step 5: Connect signals and slots
#         self.thread.started.connect(self.worker.run)
#         self.worker.finished.connect(self.thread.quit)
#         self.worker.finished.connect(self.worker.deleteLater)
#         self.thread.finished.connect(self.thread.deleteLater)
#         self.worker.progress.connect(self.reportProgress)
#         # Step 6: Start the thread
#         self.thread.start()

#         # Final resets
#         self.longRunningBtn.setEnabled(False)
#         self.thread.finished.connect(
#             lambda: self.longRunningBtn.setEnabled(True)
#         )
#         self.thread.finished.connect(
#             lambda: self.stepLabel.setText("Long-Running Step: 0")
#         )

# app = QApplication(sys.argv)
# win = Window()
# win.show()
# sys.exit(app.exec())

# import numpy as np
# from PyQt5 import QtWidgets
# from PyQt5.QtCore import pyqtSignal, pyqtSlot
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtGui
# from pyqtgraph import GraphicsLayoutWidget
# from threading import Thread, Event
# import time

# # Routine to acquire and serve data
# # This might be a camera driver, notifying when a new frame is available
# def generate_data(callback, threadkill):
#     while not threadkill.is_set():
#         width = 1600
#         data = np.zeros(width)
#         data += np.cos(np.arange(0, 10*np.pi, 10*np.pi/width) - 9*time.monotonic())
#         data += np.cos(np.arange(0, 4*np.pi, 4*np.pi/width) + 4*time.monotonic())
#         callback(data)
#         time.sleep(0.01)

# class PyQtGraphTest(GraphicsLayoutWidget):

#     # Signal to indicate new data acquisition
#     # Note: signals need to be defined inside a QObject class/subclass
#     data_acquired = pyqtSignal(list)

#     def __init__(self):
    
#         super().__init__()

#         self.setWindowTitle('Test pyqtgraph paint signals')
#         self.resize(640, 400)
#         self.plot = self.addPlot()
#         self.spectrum = self.plot.plot()
#         self.plot.enableAutoRange(pg.ViewBox.XYAxes)

#         # Connect the signal
#         self.data_acquired.connect(self.update_data)

#         # Make and start the background thread to acquire data
#         # Pass it the signal.emit as the callback function
#         self.threadkill = Event()
#         self.thread = Thread(target=generate_data, args=(self.data_acquired.emit, self.threadkill))
#         self.thread.start()

#     # Kill our data acquisition thread when shutting down
#     def closeEvent(self, close_event):
#         self.threadkill.set()

#     # Slot to receive acquired data and update plot
#     @pyqtSlot(np.ndarray)
#     def update_data(self, data):
#         self.spectrum.setData(data)

# if __name__ == '__main__':
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     window = PyQtGraphTest()
#     window.show()
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         sys.exit(app.exec_())

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import multiprocessing
import time
import random
from tkinter import *


#Create a window
window=Tk()


def main():
    #Create a queue to share data between process
    q = multiprocessing.Queue()

    #Create and start the simulation process
    simulate=multiprocessing.Process(None,simulation,args=(q,))
    simulate.start()

    #Create the base plot
    plot()

    #Call a function to update the plot when there is new data
    updateplot(q)

    window.mainloop()
    print('Done')


def plot():    #Function to create the base plot, make sure to make global the lines, axes, canvas and any part that you would want to update later

    global line,ax,canvas
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(1,1,1)
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
    line, = ax.plot([1,2,3],[1,0,10])

def updateplot(q):
    try:       #Try to check if there is data in the queue
        result=q.get_nowait()

        if result !='Q':
             print(result)
                 #here get crazy with the plotting, you have access to all the global variables that you defined in the plot function, and have the data that the simulation sent.
             line.set_ydata([1,result,10])
             ax.draw_artist(line)
             canvas.draw()
             window.after(500,updateplot,q)
        else:
             print('done')
    except:
        print("empty")
        window.after(500,updateplot,q)


def simulation(q):
    iterations = range(100)
    for i in iterations:
        if not i % 10:
            time.sleep(1)
                #here send any data you want to send to the other process, can be any pickable object
            q.put(random.randint(1,10))
    q.put('Q')

if __name__ == '__main__':
    main()
import sys
import time
import datetime

#simple logging class, call the type of message, pass in a string
class Logging(object):
#TODO add filepath handling from sys
    def __init__(self, filepath=""):
        self.filepath = filepath
        self.start = datetime.datetime.utcnow()
        print("[log] started {}" .format(self.start))

    def error(self, msg):
        print("[error] {}".format(datetime.datetime.utcnow()), msg)

    def info(self, msg):
        print("[info] {}".format(datetime.datetime.utcnow()), msg)
    
    def warning(self, msg):
        print("[warning] {}".format(datetime.datetime.utcnow()), msg)

    def results(self, msg):
        print("[results] {}".format(datetime.datetime.utcnow()), msg)
    
    def refresh(self, msg):
        print("[refresh] {}".format(datetime.datetime.utcnow()), msg)


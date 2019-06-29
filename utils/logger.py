import os, sys
import logging

class Logger(object):
    def __init__(self, log_name):
        self.terminal = sys.stdout

        if not os.path.exists(os.path.dirname(log_name)):
            os.makedirs(os.path.dirname(log_name))

        self.log = open(log_name, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    
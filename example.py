from tools import *

reader = Reader.Reader()
if reader.read_from_file('3-2.png'):
    if reader.result:
        print("up to standard")
    else:
        print("NG")
else:
    print("read error")



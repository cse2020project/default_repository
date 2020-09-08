import pathlib
import os

def adjust(a):
    line = a.split()
    line[2]= '%.6f' % ((float(line[2])*1024-384)/640)
    line[4]= '%.6f' % (float(line[4])*1.6)
    line = line[0]+" "+line[1]+" "+line[2]+" "+line[3]+" "+line[4]+"\n"
    return line


TXT_PATHS = list(pathlib.Path('.').glob("*.txt"))
for file in TXT_PATHS:
    NEW_PATHS = pathlib.Path('C:/Users/eunji/Desktop/New/'+os.path.basename(file))
    f = open(file, 'r')
    f_new = open(NEW_PATHS, 'w')
    while True:
        line = f.readline()
        if not line: break
        line = adjust(line)
        f_new.write(line)
    f.close()
    f_new.close()

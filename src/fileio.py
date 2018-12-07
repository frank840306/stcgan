import pickle
import csv
import json

def readFile(fin):
    with open(fin, encoding='utf-8') as file:
        content = file.read()
    return content.splitlines()    

def readJson(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def writeJson(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)
    return

def readCSV(file):
    # return 2d array of string
    data = []
    with open(file, 'r') as f:
        for row in csv.reader(f):
            data.append(row)
    return data

def writeCSV(data, file):
    # data is 2d array
    with open(file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for line in data:
            writer.writerow(line)
    return

def readPickle(fin, encoding=None):
    if encoding != None:
        with open(fin, 'rb') as file:
            content = pickle.load(file, encoding=encoding)
    else:
        with open(fin, 'rb') as file:
            content = pickle.load(file)
    return content
    
def writePickle(data, fout):
	with open(fout, 'wb') as file:
		pickle.dump(data, file)
	return
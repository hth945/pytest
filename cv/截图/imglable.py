import pickle
import os


class ImageLable(object):
    def __init__(self, ):
        self.paths = []
        self.PixelNumbers = []
        self.lables = []

    def multiplexing(self,srcPath,targetPath):
        if srcPath in self.paths:
            index = self.paths.index(srcPath)
            self.append(targetPath,self.PixelNumbers[index].copy(),self.lables[index].copy())

    def getByPath(self,path):
        if path in self.paths:
            index = self.paths.index(path)
            return self.paths[index], self.PixelNumbers[index], self.lables[index]
        return None,None,None

    def delete(self,path):
        if path in self.paths:
            index = self.paths.index(path)
            self.paths.pop(index)
            self.PixelNumbers.pop(index)
            self.lables.pop(index)

    def append(self,path,PixelNumber, lable):
        if path not in self.paths:
            self.paths.append(path)
            self.PixelNumbers.append(PixelNumber)
            self.lables.append(lable)
        else:
            index = self.paths.index(path)
            print(index)
            self.paths[index] = path
            self.PixelNumbers[index] = PixelNumber
            self.lables[index] = lable

    def save(self,filePath):
        data = self.dataPack()

        out_file_dir = os.path.split(filePath)[0]
        if not os.path.isdir(out_file_dir):
            os.makedirs(out_file_dir)
        with open(filePath, 'wb') as f:
            print('writeOK')
            f.write(data)
        f.close()

    def load(self,filePath):
        if os.path.exists(filePath):
            f = open(filePath, 'rb')
            data = f.read()
            f.close()
            self.dataUnpack(data)
        else:
            print('not find file!!')

    def dataPack(self):
        data = [self.paths, self.PixelNumbers, self.lables]
        return pickle.dumps(data)

    def dataUnpack(self,b):
        self.paths, self.PixelNumbers, self.lables = pickle.loads(b)


if __name__ == '__main__':
    # il = ImageLable()

    # il.append('test', [2048,2408],[1, 2, 3])
    # il.append('test2', [2048,2408],[7, 8])
    # il.append('test3', [2048,2408],[4, 5, 6])
    # il.save('testTem.pkl')
    # print(il.lables)
    # print(il.paths)
    # il.append('test2', [2048, 2408], [0, 8])
    # print(il.lables)
    # print(il.paths)
    # il.save('testTem.pkl')
    # il.append('test2', [2048, 2408], [3, 8])
    # print(il.lables)
    # print(il.paths)
    # il.load('testTem.pkl')
    # print(il.lables)
    # print(il.paths)

    il = ImageLable()
    i2 = ImageLable()
    il.load('tem/testTem.pkl')
    il.append('test', [2048, 2408], [1, 2, 3])
    il.multiplexing('test','test2')
    il.append('test', [2048, 2408], [1, 2, 4])
    print(il.lables)
    print(il.paths)
    il.save('tem/ttestTem.pkl')
    i2.load('tem/ttestTem.pkl')
    print(i2.lables)
    print(i2.paths)





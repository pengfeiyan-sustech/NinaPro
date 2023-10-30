import os
import csv
import scipy.io
import numpy as np


def main():
    # 数据集基路径
    rootPath = r"E:\LocalRepository\NinaPro\rawData"
    subFolderList = [
        os.path.join(rootPath, subFolder) for subFolder in os.listdir(rootPath)
    ]
    # 处理所有子文件夹的函数
    processSubFoldersList(subFolderList=subFolderList)


def processSubFoldersList(subFolderList):
    for subFolder in subFolderList:
        print("开始处理子文件夹：", subFolder)
        # 处理单个子文件夹的函数
        processSub(subFolder=subFolder)


def processSub(subFolder):
    allFiles = [os.path.join(subFolder, matFile) for matFile in os.listdir(subFolder)]
    for matFile in allFiles:
        if "_E1_" in matFile:
            print("-->开始处理文件：", matFile)
            processFile(matFile=matFile)


def processFile(matFile):
    savePath = os.path.dirname(matFile)
    print('基路径是', os.path.dirname(matFile))
    dataDict = scipy.io.loadmat(matFile)
    emg = dataDict["emg"]
    labels = dataDict["restimulus"].reshape(
        -1,
    )
    unique_elements, counts = np.unique(labels, return_counts=True)
    assert len(unique_elements) == 18, "错误, 标签种类不是18"
    # 给出分隔点列表
    startDict, endDict = getSplitList(labels=labels)
    for keyStart, keyEnd in zip(startDict, endDict):
        print(f'开始处理第{keyStart}类数据')
        # 处理每一类标签信号
        startList, endList = startDict[keyStart], endDict[keyEnd]
        for index, (start, end) in enumerate(zip(startList, endList)):
            datatmp = emg[start:end, :]
            csv_file = os.path.join(savePath, f'motion{keyStart}_repeat{index+1}.csv')
            print('保存数据', csv_file)
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(datatmp)
            
        


def getSplitList(labels):
    startDict = {str(i): [] for i in range(1, 18)}
    endDict = {str(i): [] for i in range(1, 18)}
    for i in range(1, len(labels) - 1):
        if labels[i] != 0 and labels[i - 1] == 0 and labels[i + 1] != 0:
            startDict[str(labels[i])].append(i)
        elif labels[i] != 0 and labels[i - 1] != 0 and labels[i + 1] == 0:
            endDict[str(labels[i])].append(i)

    return startDict, endDict


if __name__ == "__main__":
    main()

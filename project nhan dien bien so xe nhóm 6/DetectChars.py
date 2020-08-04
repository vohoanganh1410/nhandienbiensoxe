
import cv2
import numpy as np
import math
import random

import Main
import Preprocess
import PossibleChar


kNearest = cv2.ml.KNearest_create()

        # Các hằng số này để kiểm tra 1 kí tự
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

        # hằng số để so sánh 2 kí tự
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0


MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100

###################################################################################################
def loadKNNDataAndTrainKNN():
    allContoursWithData = []
    validContoursWithData = []

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)            # đọc file training classifications
    except:
        print("error\n")
        return False
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)           # đọc file training images
    except:
        print("error,\n")
        return False
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # chuyển mảng thành 1 chiều để có thể đào tạo

    kNearest.setDefaultK(1)                                                             # thiết lập k = 1

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    return True
# end function

###################################################################################################
def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:          # danh sách các tấm rỗng
        return listOfPossiblePlates
    # end if


    for possiblePlate in listOfPossiblePlates:          # duyệt mỗi tấm

        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate)     # tiền xử lý xám hóa và phân ngưỡng mỗi tấm

        if Main.showSteps == True:
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c", possiblePlate.imgThresh)
        # end if

                # tăng kích thước của ảnh lên để nhận diện dễ hơn
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

                # phân ngưỡng 1 lần nữa để loai bỏ khu vực màu xám
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                # tìm tất cả các kí tự có thể có trong tấm,
                # chức năng này trước tiên tìm thấy tất cả các đường viền, sau đó chỉ bao gồm các đường viền có thể là ký tự
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        if Main.showSteps == True:
            height, width, numChannels = possiblePlate.imgPlate.shape
            imgContours = np.zeros((height*2, width*2, 3), np.uint8)
            del contours[:]                                         # ban dau rỗng

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE) # vẽ tất cả các đường viền có trong tấm

            cv2.imshow("6", imgContours)
        # end if

        # đưa ra một danh sách tất cả các ký tự có thể, tìm các nhóm các ký tự khớp trong bảng
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if (len(listOfListsOfMatchingCharsInPlate) == 0):			# nếu không có kí tự nào đc tìm thấy trong tấm
            possiblePlate.strChars = ""
            continue
        # end if

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                              # trong mỗi ds kí tự phù hợp
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)        # sắp xếp kí tự từ trái qua phải
        # end for

        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

                # lặp qua tất cả các vectơ của ký tự khớp, lấy chỉ số của ký tự có nhiều ký tự nhất
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

                # suppose that the longest list of matching chars within the plate is the actual list of chars
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]
        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

        if Main.showSteps == True: # show steps ###################################################
            print("kí tự được tìm thấy " + str(intPlateCounter) + " = " + possiblePlate.strChars + "click để tiếp tục")
            intPlateCounter = intPlateCounter + 1
            cv2.waitKey(0)


    if Main.showSteps == True:
        print("\nHoàn thành, click tiếp tục \n")
        cv2.waitKey(0)
    # end if

    return listOfPossiblePlates
# end function

###################################################################################################
def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []                        # khởi tạo giá trị trả về
    contours = []
    imgThreshCopy = imgThresh.copy()

            # tìm tất cá các viền(contour) trong tấm
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                        # lặp với mỗi viền
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):              # nếu đường viền có thể là 1 char
            listOfPossibleChars.append(possibleChar)       # Thêm vào
        # end if
    # end if

    return listOfPossibleChars
# end function

###################################################################################################
def checkIfPossibleChar(possibleChar):
            # kiểm tra đường viền có thể là 1 char không
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if
# end function

###################################################################################################
def findListOfListsOfMatchingChars(listOfPossibleChars):
    listOfListsOfMatchingChars = []                  # khởi tạo giá trị trả về

    for possibleChar in listOfPossibleChars:                        # lặp với mỗi char trong ds
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        # tìm tất cả các kí tự trong ds phù hợp vs kí tự hiện tại

        listOfMatchingChars.append(possibleChar)                # thêm vào danh sách

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     # nếu danh sách không đủ dài để tạo thành 1 tấm
            continue
        # end if

        listOfListsOfMatchingChars.append(listOfMatchingChars)      # them vào danh sách những kí tự phù hợp

        listOfPossibleCharsWithCurrentMatchesRemoved = []

        # xóa kí tụ trùng lặp,
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # cho mỗi danh sách các ký tự trùng khớp được tìm thấy bởi cuộc gọi đệ quy
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # thêm vào danh sách ban đầu của chúng tôi danh sách các ký tự phù hợp
        # end for

        break       # exit for

    # end for

    return listOfListsOfMatchingChars
# end function

###################################################################################################
def findListOfMatchingChars(possibleChar, listOfChars):
            # đưa ra một char có thể và một danh sách lớn các ký tự có thể
    listOfMatchingChars = []                # khởi tạo giá trị trả về

    for possibleMatchingChar in listOfChars:                # mỗi char trong danh sách
        if possibleMatchingChar == possibleChar:    # nếu kí tự trùng khớp
            continue
        # end if
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

                # kiểm tra nếu kí tự phù hợp
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        # nếu các ký tự trùng khớp, thêm char hiện tại vào danh sách các ký tự khớp
        # end if
    # end for

    return listOfMatchingChars                  # return result
# end function

###################################################################################################
# tính d của 2 kí tự
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function

###################################################################################################
# tính góc của 2 kí tự
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX)) # tính gt tuyệt đối
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                     # kiểm tra khác 0
        fltAngleInRad = math.atan(fltOpp / fltAdj)      # nếu khác không -> tính góc
    else:
        fltAngleInRad = 1.5708
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # tính theo độ

    return fltAngleInDeg
# end function
###################################################################################################
# this is where we apply the actual char recognition
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sắp xếp kí tự từ trái sang phải

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     # tạo ảnh ngưỡng

    for currentChar in listOfMatchingChars:                                         # duyệt mỗi char trong tấm
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)           # vẽ 1 hộp mày xanh xung quanh kí tự

                # cắt kí tư ra khỏi ảnh ngưỡng
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))           # Thay đổi kích thước hình ảnh, điều này là cần thiết để nhận dạng char

        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        # làm phẳng ảnh thành numpy 1 chiều

        npaROIResized = np.float32(npaROIResized)               # chuyển sang dạng float 32 bit

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)              # gọi findNearest

        strCurrentChar = str(chr(int(npaResults[0][0])))            # lấy ra kết quả

        strChars = strChars + strCurrentChar                        # thêm kí tự kết quả vào strChars khởi tạo ban đầu

    # end for

    if Main.showSteps == True:
        cv2.imshow("10", imgThreshColor)
    # end if

    return strChars
# end function









import cv2
import numpy as np
import DetectChars
import DetectPlates
import PossiblePlate


SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = True



def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

    if blnKNNTrainingSuccessful == False:
        print("\nerror\n")
        return
    # end if

    imgOriginalScene  = cv2.imread("bs3.jpg")               # mở ảnh

    if imgOriginalScene is None:
        print("\nerror\n\n")
        # os.system("pause")
        return
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # dò tìm biển số

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # dò tìm kí tự trong biển số

    cv2.imshow("imgOriginalScene", imgOriginalScene)            # hiển thị ảnh

    if len(listOfPossiblePlates) == 0:                          # không tìm được biển
        print("\nkhong tim duoc bien so xe\n")
    else:
                # sắm xếp thứ tự tấm từ ít đến nhiều kí tự nhất
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)


        licPlate = listOfPossiblePlates[0]

        cv2.imshow("imgPlate", licPlate.imgPlate)           # hiển thị ảnh ngưỡng
        cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # nếu kí tự không tìm được trong biển số
            print("\nkhong tim duoc ki tu\n\n")
            return
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # vẽ viền đỏ xung quanh biển số

        print("\nbien so xe doc tu anh = " + licPlate.strChars + "\n" )
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # viết văn bản lên hình ảnh

        cv2.imshow("imgOriginalScene", imgOriginalScene)

        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)

    # end if else

    cv2.waitKey(0)

    return
# end main

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # lấy ra 4 đỉnh

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # vẽ 4 viền đỏ
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # khởi tạo tâm nơi văn bản được viết
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # khởi tạo điểm phía dưới bên phải nơi văn bản được viết
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # font chữ
    fltFontScale = float(plateHeight) / 30.0                    # tỷ lệ font với chiều cao
    intFontThickness = int(round(fltFontScale * 1.5))           # độ dày cơ bản

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # gọi getTextSize

    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # tâm là 1 số nguyên
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # nếu biển số nằm trên 3/4 ảnh
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # Viết kí tự bên dưới biển số
    else:                                                                                       # nếu biển số nằm trên 1/4 ảnh
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # Viết kí tự bên trên biển số
    # end if

    textSizeWidth, textSizeHeight = textSize                # kích thước của text

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # góc bên trái của bùng văn bản
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))

            # viết text vào
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function

###################################################################################################
if __name__ == "__main__":
    main()


















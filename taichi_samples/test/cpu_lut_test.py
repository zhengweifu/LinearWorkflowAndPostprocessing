import cv2
import os

CELL_SIZE = 16.0

def LookUpTable(srcImg, lutImg):
    h, w = srcImg.shape[0], srcImg.shape[1]
    lutH, lutW = lutImg.shape[0], lutImg.shape[1]
    cellRow = lutH / CELL_SIZE
    cellCol = lutW / CELL_SIZE
    for i, j in range(h, w):
        color = srcImg[i, j] / 255.0
        print(color)

def Main():
    imgDir = os.path.join(os.getcwd(), "images")
    # srcImgPath = os.path.join(imgDir, "sh_smaller.png")
    srcImgPath = os.path.join(imgDir, "test.png")
    srcImg = cv2.imread(srcImgPath, cv2.IMREAD_UNCHANGED)

    lutImgPath = os.path.join(imgDir, "LUT_Sepia.webp")
    lutImg = cv2.imread(lutImgPath, cv2.IMREAD_UNCHANGED)

    LookUpTable(srcImg, lutImg)

    titleName = "Color Grading"
    cv2.imshow(titleName, srcImg)
    cv2.waitKey()

if __name__ == "__main__":
    Main()

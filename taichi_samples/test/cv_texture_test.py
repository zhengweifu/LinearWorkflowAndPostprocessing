import cv2
import os


def Main():
    imgDir = os.path.join(os.getcwd(), "images")
    imgPath = os.path.join(imgDir, "RGBTable16x1.webp")
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    w, h = img.shape[1], img.shape[0]
    for r in range(h):
        for c in range(w):
            print(r, c, img[r, c])
            if c > 20:
                break

if __name__ == "__main__":
    Main()

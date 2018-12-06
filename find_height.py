import numpy as np
import cv2 as cv


def main(imageName):

    image = cv.imread(imageName)
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    cv.imwrite("gray.jpg", gray)

    edged = cv.Canny(gray, 10, 250)
    cv.imwrite("edged.jpg", edged) 

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20)) 
    closed = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel) 
    cv.imwrite("closed.jpg", closed)

    cnts = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1]
    c = cnts[0]

    extTop = tuple(c[c[:, :, 1].argmin()][0])

    print(extTop)

    cv.circle(image, extTop, 10, (255, 0, 0), -1)

    cv.imwrite("output.jpg", image)


main("bush3.jpg")


# CV - Project 1 - Question 2

import cv2
import numpy as np
import sys
import math

class Pixel:
    L = 0
    u = 0
    v = 0
    def __init__(self, L, u, v):
        self.L = L
        self.u = u
        self.v = v

# ----------------------------------------------------------------------------------------------------------
# non-linear sRGB to LUV
# ----------------------------------------------------------------------------------------------------------

# non-linear sRGB to non-linear RGB
def sRGBtoRGB(n):
    n = float(n/255)
    return n

# non-linear RGB to linear RGB -> INV GAMMA CORR
def invGamma(v):
    if (v < 0.03928):
        v = v/12.92
    else:
        v = pow(((v+0.055)/1.055), 2.4)
    return v    

# linear RGB to XYZ -> Linear Transformation
def RGBtoXYZ(r, g, b):
    convMatrix = [[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]]
    return np.matmul(convMatrix, [r, g, b])

# XYZ to LUV 
def XYZtoLUV(X, Y, Z):
    Xw, Yw, Zw = 0.95, 1.0, 1.09
    uw = 4*Xw/(Xw + 15*Yw + 3*Zw)
    vw = 9*Yw/(Xw + 15*Yw + 3*Zw)
    t = Y/Yw
    if (t > 0.008856):
        L = 116*pow(t, (1/3)) - 16
    else:
        L = 903.3*t
    d = X + 15*Y + 3*Z
    # if (d == 0): to avoid division by 0, we consider
    uDash = 0
    vDash = 0
    if (d != 0):
        uDash = 4*X/d
        vDash = 9*Y/d

    u = 13* L* (uDash - uw)
    v = 13* L* (vDash - vw)

    return L, u, v 

# ----------------------------------------------------------------------------------------------------------
# LUV to non-linear sRGB
# ----------------------------------------------------------------------------------------------------------

def LUVtoXYZ(L, u, v):
    Xw, Yw, Zw = 0.95, 1.0, 1.09
    uw = 4*Xw/(Xw + 15*Yw + 3*Zw)
    vw = 9*Yw/(Xw + 15*Yw + 3*Zw)
    # if L == 0, then we consider,
    uDash = 0
    vDash = 0
    if (L != 0):
        uDash = (u + 13*uw*L)/(13*L)
        vDash = (v + 13*vw*L)/(13*L)

    if (L > 7.9996):
        Y = pow(((L+16)/116), 3) * Yw
    else:
        Y = (L/903.3)*Yw

    # if vDash == 0, then we consider,
    X = 0
    Z = 0
    if (vDash != 0):
        X = ( Y * 2.25 * uDash ) / vDash
        Z = ( Y * (3 - 0.75*uDash - 5*vDash)/ vDash )

    return X, Y, Z

def XYZtosRGB(X, Y, Z):
    convMatrix = [[3.240479, -1.53715, -0.498535], [-0.969256, 1.875991, 0.041556], [0.055648, -0.204043, 1.057311]]
    return np.matmul(convMatrix, [X, Y, Z]) 


def clip(n):
    if (n < 0):
        n = 0
    elif (n > 1):
        n = 1
    return n

def gamma(d):
    if (d < 0.00304):
        return 12.92*d
    else:
        return (1.055*pow(d, (1/2.4)) - 0.055)
    

# ----------------------------------------------------------------------------------------------------------
# MAIN function
# ----------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    # accepting parameters
    if(len(sys.argv) != 7) :
        print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
        print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
        print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
        sys.exit()

     # assigning parameters
    w1 = float(sys.argv[1])
    h1 = float(sys.argv[2])
    w2 = float(sys.argv[3])
    h2 = float(sys.argv[4])
    name_input = sys.argv[5]
    name_output = sys.argv[6]

    # checking constraints
    if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
        print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
        sys.exit()

    # checking if input image exists
    inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
    if(inputImage is None) :
        print(sys.argv[0], ": Failed to read image from: ", name_input)
        sys.exit()

    # if yes, display image
    cv2.imshow("input image: " + name_input, inputImage)


    # fake image for now
    # ipImage = np.array([
    # [ [0,0,0], [0,0,0], [0, 0, 0], [0, 0, 0]] ,
    # [ [255, 0, 0] ,[255, 0, 0], [255, 0, 0] ,[255, 0, 0] ],
    # [ [100, 100, 100] ,[100, 100, 100] ,[100, 100, 100] ,[100, 100, 100] ] ,
    # [ [0, 100, 100] , [0, 100, 100] , [0, 100, 100] , [0, 100, 100] ] ])

    
    # The transformation should be based on the
    # historgram of the pixels in the W1,W2,H1,H2 range.
    # The following code goes over these pixels

    # ----------------------------------------------------------------------------------------------------------
    # Non-linear sRGB to LUV
    # ----------------------------------------------------------------------------------------------------------
       
    # get size of image
    rows = len(inputImage)
    cols = len(inputImage[0])

    tempImage  = [[0 for x in range(cols)] for y in range(rows)] 

    # print("The size of the matrix is: [", rows, cols, "]" )
    for i in range(rows) :
        for j in range(cols) :
            
            b, g, r = inputImage[i, j]
            
            # non-linear sRGB to non-linear RGB
            r, g, b = sRGBtoRGB(r), sRGBtoRGB(g), sRGBtoRGB(b)
            

            # non-linear RGB to linear RGB -> INV GAMMA CORR
            r, g, b = invGamma(r), invGamma(g), invGamma(b)

            # linear RGB to XYZ -> Linear Transformation
            X, Y, Z = RGBtoXYZ(r, g, b)

            # XYZ to LUV 
            L, u, v = XYZtoLUV(X, Y, Z)        
          
            tempImage[i][j] = Pixel(L, u, v)
           
            # print(tempImage[i][j].L, tempImage[i][j].u, tempImage[i][j].v)
            # print("one pixel done")

    # ----------------------------------------------------------------------------------------------------------
    # Histogram equalization in 0 to 100 with discretization step for real valued L 
    # ----------------------------------------------------------------------------------------------------------

    rows, cols, bands = inputImage.shape # bands == 3
    W1 = round(w1*(cols-1))
    H1 = round(h1*(rows-1))
    W2 = round(w2*(cols-1))
    H2 = round(h2*(rows-1))

   #calculate the freq and then Histogram table from the H1W1 H2W2 window 
    freq = np.zeros(101)
    data = np.zeros(101)

    for i in range(H1, H2) :
        for j in range(W1, W2) :
            L, u, v = tempImage[i][j].L, tempImage[i][j].u, tempImage[i][j].v
            L = int(round(L))
            if(L <= 0):
                freq[0]= freq[0] + 1
            elif(L >= 100):
                freq[100]= freq[100] + 1
            else:
                freq[L]= freq[L] + 1

    for i in range(1, len(freq)):
        freq[i] = freq[i] + freq[i-1]

    for i in range(0, len(freq)):
        if(i == 0):
            data[i] = math.floor((freq[i]/2)*(101/freq[100]))
        else:
            data[i] = math.floor(((freq[i]+freq[i-1])/2)*(101/freq[100]))    

    #apply L values using above VALUES array to this whole image.
    for i in range(0, rows) :
        for j in range(0, cols) :
            L = tempImage[i][j].L
            L = int(round(L))
            if(L < 0):
                L = data[0]
            elif(L > 100):
                L = data[100]
            else:
                L = data[L]
            tempImage[i][j].L = L

    # ----------------------------------------------------------------------------------------------------------
    # LUV to Non-linear sRGB (BACK TO NORMAL)
    # ----------------------------------------------------------------------------------------------------------

    outputImage = np.zeros([rows, cols, bands], dtype = np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            L, u, v = tempImage[i][j].L, tempImage[i][j].u, tempImage[i][j].v

            # LUV to XYZ
            X, Y, Z = LUVtoXYZ(L, u, v)

            # XYZ to linear sRGB -> linear transformation
            r, g, b = XYZtosRGB(X, Y, Z)

            # clip r, g, b values that are not in the range [0, 1]
            r, g, b = clip(r), clip(g), clip(b)

            # linear sRGB to non-linear sRGB
            r, g, b = 255*gamma(r), 255*gamma(g), 255*gamma(b)

            outputImage[i, j] = [b, g, r]

    cv2.imshow("Output", outputImage)
    cv2.imwrite(name_output, outputImage);            


# end of example of going over window




    # wait for key to exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()

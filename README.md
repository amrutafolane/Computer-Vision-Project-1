# Computer-Vision-Project-1
This project deals with the linear stretching in the LUV domain and histogram equalization of the images from a particular window of the image.

It contains two parts:
1) A program that gets as input a color image, performs linear scaling in the Luv domain in the specified window, and writes
the scaled image as output.
2) Histogram Equalization in Luv applied to the luminance values, as computed in the specified window.

For more detailed information, refer to Project1.pdf

Instructions for running the code:
Navigate to the folder containing the code files and from the cmd/ terminal, type:

For Question 1:
python Cv_Proj1_Q1.py 0.2 0.2 0.8 0.8 flower.jpg flowerOut_Q1.jpg
python Cv_Proj1_Q1.py 0.2 0.2 0.8 0.8 horses.jpg horsesOut_Q1.jpg

For Question 2:
python Cv_Proj1_Q2.py 0.2 0.2 0.8 0.8 flower.jpg flowerOut_Q2.jpg
python Cv_Proj1_Q2.py 0.2 0.2 0.8 0.8 horses.jpg horsesOut_Q2.jpg

? How did I handle the divide by zero situations ?
A.	In such situations, I pre-assigned them to 0, and then reassigned them by putting the condition that if the denominator is not 0, then what values would the affected variables take.

? Situations where the image looked bad ?
A.	I usually tried the window 0.2 0.2 0.8 0.8, which was mostly the center of the image. However, when I made the entire image as a window, i.e. 0 0  1 1, the color of the output image became quite dark, some people might not like this photo effect. But, according to me, it made the image quite sharp with distinct borders.

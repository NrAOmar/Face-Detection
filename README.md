# CRV_Face_Detection

## TODO
- ✅ add flags that allow us to disable all of these features/improvements like the Professor
    - ✅ flag for tilting
    - ✅ flag for which model to use HAAR/DNN
    - ✅ flag for model fusion
    - ✅ flag for face recognition (biometric)
    - ✅ flag for multiple cameras

- ✅ add tilting
    - ✅ rotate 15 degrees => 24 images per frame
    - ✅ get the conrer points of the box
    - ✅ map this center point using the inverse rotation matrix
    - ✅ draw the new box
    - ✅ Combine all boxes
    - ✅ Fix the problem with the pixelated image
    - ✅ Use multiple threads for operations to reduce delay 
    - ✅ Use image without cropping. 

- ✅ use DNN for 1 or 2 angles only instead of many angles like HAAR
    - this is done to reduce redundant computations
    - although this will only detect rotated faces for **very** close faces

- ✅ fuse both algorithms for faraway faces
- ✅ add biometrics
- ✅ add plotting function that plots in a grid
- ✅ use multiple cameras together
- ✅ adjust text size to depend on frame size


## Python environment setup
- only works for old python versions like python 3.10.0 or python 3.9.6
- python3 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
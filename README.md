# CRV_Face_Detection

## TODO
- ✅ add plotting function that plots in a grid
- ✅ use 2 cameras together
- ✅ add flags that allow us to disable all of these features/improvements like the prof
    - ✅ flag for tilting
    - ✅ flag for which model to use HAAR/DNN/BOTH and the fusion
    - ✅ flag for biometric
    - flag for *"low pass filter"*
- add tilting
    - ✅ rotate 15 degree => 24 images per frame
    - ✅ get the conrer points of the box
    - ✅ map this center point using the inverse rotation matrix
    - ✅ draw the new box
    - ✅ Combine all boxes
    - ❌ Fix the problem with the pixelated image
    - ❌ Use multiple threads for operations to reduce delay 
    - ✅ Use image without cropping. 
- fuse both algorithms for faraway faces
    Options:
    - use DNN for 1 angle only: only work for **very** close faces
    - use DNN for all angles: more threads required (slower)
- add biometrics
    - change to only add the name to current boxes instead of adding new boxes
    - use 1 worker for each name
- adjust text size to depend on frame size

- fix writing to .mp4 file
- add low pass filter for faces
    - will introduce a delay when detecting faces


## Python environment
- python3 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
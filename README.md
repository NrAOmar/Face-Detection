# CRV_Face_Detection

## TODO
- add tilting
    - ✅ rotate 15 degree => 24 images per frame
    - ✅ get the conrer points of the box
    - ✅ map this center point using the inverse rotation matrix
    - ✅ draw the new box
    -  ❌ combine all boxes
- add low pass filter for faces
    - will introduce a delay when detecting faces
- call ahmed and ask about biometrics
- fuse both algorithms for faraway faces

## Python environment
- python3 -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt
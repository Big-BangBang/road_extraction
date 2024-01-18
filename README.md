# Requirements

- Cuda 11.2
- Python 3.8
- Pytorch 1.11
- cv2

# Usage

### Data
Place '*train*', '*valid*' and '*test*' data folders in the '*dataset*' folder.

Data is from [DeepGlobe Road Extraction Challenge](https://competitions.codalab.org/competitions/18467#participate-get_starting_kit). You should sign in first to get the data.

### Train
- Run `python train.py` to train the default D-LinkNet34.

### Predict
- Run `python test.py` to predict on the default D-LinkNet34.


# PyTorch Dlib Face Rec
Port of Davis King's dlib face recognition resnet model to PyTorch

Please obtain the model itself from https://github.com/davisking/dlib-models/blob/master/dlib_face_recognition_resnet_model_v1.dat.bz2

There is some trickery required to get the *tag*, *add_prev* and *skip* layers to work - the file layer_tags.txt contains additional metadata to identify layers of those types.

How to use:

Import PyTorch and set device
``` 
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Import FaceRecResNet class and create instance of the model
```
from face_rec_resnet import FaceRecResNet
model = None
with torch.no_grad():
    model = FaceRecResNet("dlib_face_recognition_resnet_model_v1.dat", "layer_tags.txt").to(device)
```

Import FaceMatcher class and create instance
```
from face_matcher import FaceMatcher
matcher = FaceMatcher(model, device)
```

Check faces for match

```
# note that the images must of size 150x150
matcher.is_match("faces/face0.png", "faces/face8.png")
matcher.is_match("faces/face0.png", "faces/face4.png")
```

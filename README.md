# LGPN-net
This is an official implementation of Layout-guided Indoor Panorama Inpainting with Plane-aware Normalization.

## Enviroment

- python 3.6
- Pytorch 1.7
- scipy 1.2
- scikit-image 0.14.2
- tensorboard
- tensorboardX
- tqdm 
- pyyaml 
- shapely


## Codes

We built our code based on [EdgeConnect]. Part of the code were derived from [PanoDR], [SEAN], [Partial convolution], [HorizonNet], [Structured3D].

If you have any questions about the code logic, you can refer to the source code for more detailed information.

| File | logic |
| ------ | ------ |
| src/dataset.py | Dataloader: RGB image(3); mask(1); layout(1); layout instance(100) |
| src/config.py | Configuration logic. |
| src/SIInpainting.py | Model operation: train; eval; test; sample... |
| src/model.py | Network operation: inference; backpropogation...|
| src/network.py | Build the neural network architecture. |
| src/loss.py | Loss function defination.|
| src/util.py | Some image processing/visualization  tools(refer to [EdgeConnect]). |
| src/horizon_net.py | HorizonNet pretrained-model setting. Not used in this setting.(refer to [HorizonNet])|
| src/SEAN/normalization.py | SEAN normalization(refer to [SEAN]&[PanoDR],)|
| src/SEAN/partialconv2d.py | Partial convolution(refer to [Partial convolution])|
| src/SEAN/spade_arch.py | SEAN normalization(refer to [SEAN]&[PanoDR])|
| src/misc/* | Drawing layout from S3D txt file. (refer to [Structured3D])|
| config.yml.example | Configuration template file. |
| main.py | Operation interface. |
| test.py | Testing |
| train.py | Training |

And of course Dillinger itself is open source with a [public repository][dill]
 on GitHub.

## Training/testing

The training process following the setting of `config.yml.example`.
Or you can execute the training command first and then stop instantly, then modify `config.yaml` in the checkpoint folder(automatically created after executing the training command), finally execute the command to continue training.

```sh
python train.py --checkpoint <checkpoint_dir>
```
The testing can be performed directly by executing the following commands or define the testing dataset path in `config.yaml`.
```sh
python test.py --checkpoint <checkpoint_dir> --input <input dir or file> --mask <mask dir or file> --output <output dir> --dubug <optional>
```
The path of the training data set defined in `config.yml.example` uses the [Structured3D] official flist. Note that format(e.g. scene_id/2D_rendering/room_id/panorama) must match the settings of the dataloader to accurately locate the dataset.

The testing set we used for evaluation can be found in the following link:
https://drive.google.com/file/d/1qgt0wPOPTtKmoFHJ0-W9Oz-wtQqFM5iX/view?usp=share_link

## Pretrained model
Download link:
https://drive.google.com/file/d/1J9ZgPxZCbuWrRDaIngvM2T65An-cmVdb/view?usp=share_link

Please unzip to your project folder.
In fact, only the pre-trained weight files is what you need, and other dependent files will be automatically downloaded when the program is executed.


[//]: # ()

   [EdgeConnect]: <https://github.com/knazeri/edge-connect>
   [PanoDR]: <https://github.com/VCL3D/PanoDR>
   [SEAN]: <https://github.com/ZPdesu/SEAN>
   [Partial convolution]: <https://github.com/NVIDIA/partialconv>
   [Structured3D]: <https://github.com/bertjiazheng/Structured3D>
   [HorizonNet]: <https://github.com/sunset1995/HorizonNet>

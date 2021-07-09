# 3dssg
3D scene graph generation using GCN

reproduction of CVPR2020 "Learning 3D Semantic Scene Graphs from 3D Indoor Reconstructions"

### Setup

The code works under pytorch 1.6.0 with only one card supported. Execute the following command to install PyTorch:  

```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

Install the necessary packages listed out in requirements.txt:  

```shell
pip install -r requirements.txt
```

After all packages are properly installed, please run the following commands to compile the CUDA modules for the PointNet++ backbone:  

```shell
cd lib/pointnet2
python setup.py install
```

### Usage

Train the default GCN model with the following command:

```shell
python scripts/train.py
```

It also makes sense to change the hyperparameters using command line arguments like `--batch_size`, `--epoch` etc.  
Use argument `--use_pretrained` to load pretrained model. Use argument `--vis` for visualization and the results will be saved under `vis` folder.

### Visualization
Scene-id: 7747a50c-9431-24e8-877d-e60c3a341cc2

![7747a50c-9431-24e8-877d-e60c3a341cc2](https://user-images.githubusercontent.com/50099204/125017572-00f23b80-e0a6-11eb-9a75-186dae056a02.png)

Scene-id: 43b8cadf-6678-2e38-9920-064144c99406

![43b8cadf-6678-2e38-9920-064144c99406](https://user-images.githubusercontent.com/50099204/125017851-837afb00-e0a6-11eb-9bdd-7d9230ef9ad1.png)

Scene-id: ba6fdaaa-a4c1-2dca-8163-a52b18bf6b64

![ba6fdaaa-a4c1-2dca-8163-a52b18bf6b64](https://user-images.githubusercontent.com/50099204/125017936-ab6a5e80-e0a6-11eb-9722-d6dbe10e19a4.png)

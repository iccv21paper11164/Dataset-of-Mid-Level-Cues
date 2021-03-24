# A Scalable Multi-Task Dataset of Mid-Level Cues

Anonymized code in submission for ICCV 2021

Paper ID:11164

## Dataset
We provide a sample dataset icluding `14 midlevel cues` (332 images for each cue) from a Replica mesh. You can download the dataset from [here](https://drive.google.com/file/d/1HgRwEHB-c-5QJ7eVrMZBEZTbnHSPMFUR/view?usp=sharing).

## Dataloader
The code includes a Dataloader that apply necessary transforms for reading in each modality to analytic values.

## Instructions
Download the dataset provided in [here](https://drive.google.com/file/d/1HgRwEHB-c-5QJ7eVrMZBEZTbnHSPMFUR/view?usp=sharing), and extract the zip file in `dataset/` folder.
The folder structure should be as following:
```bash
dataset 
└─── apartment_2
│   └─── point_info
│   └─── rgb
│   └─── normal
|   └─── depth_zbuffer
│   └─── depth_euclidean
|   └─── reshading
|   └─── principal_curvature
|   └─── keypoints3d
|   └─── keypoints2d
|   └─── edge_occlusion
|   └─── edge_texture
|   └─── segment_unsup2d
|   └─── segment_unsup25d
|   └─── semantic
|   └─── mask_valid
```
    
To use the dataloader for loading different modalites run:

```
python load_midlevel_cues.py --midlevel_cues 'rgb' 'normal' \
                             --batch_size 8 --image_size 512 --data_path dataset/
```
You can specify different modalities to load in each batch using the tag `--midlevel_cues`.

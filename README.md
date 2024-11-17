## ReadMe
---
The goal of the project was to either create or finetune a Vision model in order to do scene detection utilizing Meta's [Project Aria](https://www.projectaria.com/) Smart AR glasses along with the open  datasets from [project aria's open datasets](https://www.projectaria.com/datasets)

The first thing you'll probably need to do is clone this repository which can be done with this command

```
git clone git@github.com:jay-rai/project-aria-deep-learning.git
```
### Dataset Preprocessing
---
Our first goal was to get the aria data set in to a format for LLaVa to be able to read. In order to do that we need to download the dataset from LLaVa. In this repository is the json file for the *Aria Digital Twin* data set.  In order to get the dataset prepared we can build it with.

**Important steps** to do before running:

In order to build the dataset you will need to first make sure you installed all the requirements, please utilize some sort of virtual environment. If you use anaconda you can type the following to make a new environment

```
conda create -n "llava" python=3.10
```

```
conda activate llava
```

Make sure you go through and have most of the dependecies

After of which we still need some dependencies you most likely don't have, `ffmpeg`, to do this you'll have to type to the following

- **Windows**
```
choco install ffmpeg
```
- **Mac**
```
brew install ffmpeg
```
- **Linux**
```
sudo apt install ffmpeg
```

*Note:* Please do the above in **system administrator**, i.e right click your PowerShell window or whatever and run as admin.

#### Getting the Data

Congrats, you finally have enough to put together the dataset. Once you've done everything above you can finally run the following to build the dataset, keep in mind the parameters you pass.

```
cd projectdirectory/
```

Quick download : downloads max 10 videos from the dataset to conserve your drive space. Automatically creates the dataset directory for you

```
python get_data.py --json_path 'ADT_download_urls.json'
```

Custom download : downloads however many you specify to wherever you specify, automatically creates and fills out dataset directory for you

```
python get_data.py --json_path 'ADT_download_urls.json' --dataset_dir 'aria_dataset' --max_download 10
```
- json_path = wherever you stored *ADT_download_urls.json*
- datset_dir = wherever you want the final dataset to be (has default parameters)
- max_download = amount of videos and ground truths you want do download and extract

Your output from this should be a folder structure that appears as the following
```
aria_dataset/ 
├── video1/ 
│ ├── frames/ 
│ │ ├── frame_0001.jpg 
│ │ ├── frame_0002.jpg 
│ │ └── ... 
│ ├── annotations/ 
│ │ ├── 2d_bounding_box.csv 
│ │ ├── 3d_bounding_box.csv 
│ │ ├── aria_trajectory.csv 
│ │ ├── eyegaze.csv 
│ │ ├── instances.json 
│ │ ├── metadata.json 
│ │ └── scene_objects.csv 
├── video2/ 
│ └── ... 
└── ...
```

Make sure that all the frame extraction and all the ground truths have been extracted properly

#### Preparing the Data
As the current implementation goes, we know that `instances.json` give us information on objects are *static* and *dynamic* so why not not feed in the dynamic objects into LLaVa in order to make it better at classifying actions, or at least thats the hope. 

Once you have downloaded an confirmed that the data is in the format as above, then you should run the next file `prepare_data.py` or run 

```
python prepare_data.py
```

This should make a folder called `prepared_dataset` which should contain the json format of the image location, the dynamic objects, and the prompt for LLaVa

**Extra Steps most likely to be added  / all the above subject to change at any moment**
### Finetuning

Welp we haven't got this working yet oops...

ReadMe last updated (11/16/2024)
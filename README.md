### ReadMe
---
The goal of the project was to either create or finetune a Vision model in order to do scene detection utilizing Meta's [Project Aria](https://www.projectaria.com/)Smart AR glasses

The datasets we used are from [project aria's open datasets](https://www.projectaria.com/datasets/). Mainly the hot3d dataset was used

#### How to run the model
---
The following instruction are broken down into, dataset preproccesing and model training and evaluation

##### Dataset Preprocessing
---
To download the data, the json provided file from the hot3d data set gives all the required link depending on what you want to download, for example you can run 
```
curl -o hot3d_data/video.mp4 "https://scontent.xx.fbcdn.net/m1/v/t6/An--UmZuYb6rSv-RDMsmk--Z2-w2LLvDsOweEGyzvFj4H53inoERaJ7YHTLPtdkm-ygPtKWWz-yI98M_vTZJtrzL7pXn46OqZ31-MSVORiT7-nuHjPjzV9CB5Wv-N5Kc58YGj5Cwx-fvV9P8Q5tUyTzcRhgr0ZU.mp4"
```

Once you have the videos you can run the getFrames
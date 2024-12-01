## ReadMe 
---

The goal of the project was to either create or finetune a Vision model in order to do scene detection utilizing Meta's [Project Aria](https://www.projectaria.com/)Smart AR glasses along with the open  datasets from [project aria's open datasets](https://www.projectaria.com/datasets)

The first thing you'll probably need to do is clone this repository which can be done with this command

```
git clone git@github.com:jay-rai/project-aria-deep-learning.git
```
### Running main notebook
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

**Install `FFMPEG`** 

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

After you have these basic ffmpeg requirement, all the dependency install commands are included in the main notebook.

# podalize
Podalize: Podcast Transcription and Analysis


## How to install
1- Install [Anaconda](https://www.anaconda.com/)

2- Clone/download this repo to your local machine. 

Get a pyannote.adudio access token by following the instructions 
[here](https://github.com/mave5/podalize/blob/main/configs.py)


3- Lanch anaconda prompt and navigate to the repo on your local machine

4- Create a conda environment from environment.yml

```
$ conda env create -f environment.yml
```

4- Activate the conda environment

```
$ conda activate podalize
```

5- Run streamlit app

```
$ streamlit run podalize_app.py
```



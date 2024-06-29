## Intro

Hi I'm Adam Walsh. This is a project I developed along with IKEA. Code was built on top of existing work done by a previous student. 

## Package 
There are a handful of methods and classes required to run any of the models. They are in the recommenders folder. It is required to install this package.


## Preprocessing of the data 

Data Preparation
You'll find all the important scripts for handling the data in the scripts/IKEA/data/ folder. Notice: No data is present in this github.

Step 1: Preprocess, Split, and Upload
First, we take the data chunks in JSON format from GCP, clean them up, combine them into one big file, split this into train, validation, and test sets, and then upload everything back to GCP. This is all handled by the preprocess_split_upload.sh script. Rewards were added via a gcp notebook later on and redownloaded

Creating the Data Loader preprocessed data. 
Use the create_all_buffers.sh script to load in the data appropriately. This would need modification depending on the dataset.


## Training

My workflow consisted of making changes/updates to my code. Using pre-written scripts to push new changes to artifact registry. The data during preprocessing was left in a bucket on gcp and is referenced in the training code. I then run a vertex ai job via the gcp CLI. 

## Result Analysis 

Results were collected in wandb and analysis is in thesis document.

## Reach Out

This is a bare-bones explanation of the project, feel free to reach out here: walsha480@gmail.com for further explanation.
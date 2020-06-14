# EmotionalConversionStarGAN
This repository contains code to replicate results from the ICASSP 2020 paper "StarGAN for Emotional Speech Conversion: Validated by Data Augmentation of End-to-End Emotion Recognition".

- stargan: code for training the Emotional StarGAN and performing emotional generation (originally here - https://github.com/max-elliott/StarGAN-Emotional-VC).

- aug_evaluation: code for performing the data augmentation experiments (coming soon)

- samples: some samples selectively (coming soon - checking with IEMOCAP if we can publicly share according to GDPR)

The IEMOCAP database requires the signing of an EULA; please communicate with the handlers: https://sail.usc.edu/iemocap/

# Preparing
**- Requirements:**
* python>3.7.0
* pytorch
* numpy
* argparse
* librosa
* scikit-learn
* tensorflow < 2.0
* pyworld
* matplotlib
* yaml

**- Clone repository:**
```
git clone https://github.com/glam-imperial/EmotionalConversionStarGAN.git
cd EmotionalConversionStarGAN
```
**- Download IEMOCAP dataset from https://sail.usc.edu/iemocap/**

# IEMOCAP Preprocessing
Running the script **run_preprocessing.py** will prepare the IEMOCAP as needed for training the model. It assumes that IEMOCAP is already downloaded and is stored in an arbitrary directory <DIR> with this file structure
```
<DIR>
  |- Session1  
  |     |- Annotations  
  |     |- Ses01F_impro01  
  |     |- Ses01F_impro02  
  |     |- ...  
  |- ...
  |- Session5
        |- Annotations
        |- Ses05F_impro01
        |- Ses05F_impro02
        |- ...
```
where Annotations is a directory holding the label .txt files for all Session<x> (Ses01F_impro01.txt etc.), and each other directory (Ses01F_impro01, Ses01F_impro02 etc.) holds the .wav files for each scene in the session.
  
 To preprocess run
 ```
 python run_preprocessing.py --iemocap_dir <DIR> 
 ```
 which will move all audio files to ./procesed_data/audio as well as extract all WORLD features and labels needed for training. It will only extract these for samples of the correct emotions (angry, sad, happy) and under the certain hardocded length threshold (to speed up training time). it will also create dictionaries for F0 statistics which are used to alter the F0 of a sample when converting.
After running you should have a file structure:
```
./processed_data
 |- annotations
 |- audio
 |- f0
 |- labels
 |- world
 ```
 # Training EmotionStarGAN
 Main training script is **train_main.py**. However to automatically train a three emotion model (angry, sad, happy) as it was trained for "StarGAN for Emotional Speech Conversion: Validated by Data Augmentation of End-to-End Emotion Recognition", simply call:
 ```
 ./full_training_script.sh
 ```
 This script runs three steps:
 1. Runs classifier_train.py - Pretrains an auxiliary emotional classifier. Saves best checkpoint to ./checkpoints/cls_checkpoint.ckpt
 2. Runs main training for 200k iterations in --recon_only mode, meaning model learns to simply reconstruct the input audio.
 3. Trains model for a further 100k steps, introducing the pre-trained classifier.
 
 A full training run will take ~24 hours on a decent GPU. The auxiliary emotional classifier can also be trained independently using **classifier_train.py**.
 
 # Sample Conversion
 Once a model is trained you can convert IEMOCAP audio samples using **convert.py**. Running
 ```
 python convert.py --checkpoint <path/to/model_checkpoint.ckpt> -o ./processed_data/converted
 ```
 will load a model checkpoint and convert 10 random samples from the test set into each emotion and save the converted samples in /processed_data/converted (currently bugged: run conversion as stated below).
 Specifying an input directory will convert all the audio files in that directory:
 ```
 python convert.py --checkpoint <path/to/model_checkpoint.ckpt> -i <path/to/wavs> -o ./processed_data/converted
 ```
 They currently must be existing files in the IEMOCAP dataset. Code will be updated to convert arbitrary samples later.

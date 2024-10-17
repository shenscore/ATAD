<!-- <img width="200px" src="https://github.com/shenscore/HTAD/blob/master/doc/logo.png" /> -->

# Introduction
HTAD is an Active Learning Tool for Matrix-based Detection of Chromatin Domain

# Installing HTAD
```
$ git clone https://github.com/shenscore/HTAD
$ cd HTAD
$ conda create -n HTAD python=3.11
$ conda activate HTAD
$ pip install -r requirements.txt
```
# Run HTAD
### (i) Feature extraction
Calculate the potential TADs and corresponding TAD features.
```
python /path/to/HTAD/src/calcFeatures.py [cooler] [resolutions] [outPrefix]
# example
python /path/to/HTAD/src/calcFeatures.py test.mcool 10000,20000,40000 Test
```
then the pickle file of potential TADs will be generated (outPrefix.10k.pkl, outPrefix.20k.pkl, outPrefix.40k.pkl)

**parameters**: 
+ **cooler**: cooler file of Hi-C data

+ **resolutions**: resolutions used to calculate potential TADs and corresponding TAD features, separated by ','.

+ **outPrefix**: prefix of output files

**Output:**
+ outPrefix.di_check_value
+ outPrefix.10k.pkl, outPrefix.20k.pkl, outPrefix.40k.pkl


### (ii) Manual annotation
Run the HTAD labeler server and train TAD identification model.

to start the web server, run:
```
cd dataLabel
python manage.py runserver
```
Then visit the corresponding port: e.g. 127.0.0.1:8000

**Input:**

+  the file path of your cooler file and feature file of potential TADs

+  resolution of the feature file

+  label(prefix) for the output TAD identification model file

Then the server will select samples (50 for each round) for manual labeling.

The webpage will show corresponding heatmap with TAD marked by **yellow triangle**.

<img width="1000px" src="https://github.com/shenscore/HTAD/blob/master/docs/HTADdemo.gif" />

You can quickly judge whether current TAD is real. We also provide the prediction of current model in the interface.

After each round, the server will generate a XX_roundX.h5 model file.

We suggest that the model should be trained in around 10 rounds with the highest resolution to get the best performance.

Note that we could just use the model trained by the highest resolution to predict the real TAD from other resolutions.

**Output:**
+ \*round\*.h5
+ *finalmodel.h5

### (iii) TAD identification
given the well trained TAD model file model.h5, run:
```
python predictTAD.py [model] [feature files of potential TADs] [resolutions] [n] [outPrefix]
# example
python predictTAD.py model.h5 potentialTAD.10k.pkl,potentialTAD.20k.pkl,potentialTAD.40k.pkl 10000,20000,40000 0 Test
```
**parameters:**
+ **model**: TAD model file generated from previous step.
+ **feature files of potential TADs**: feature files of potential TADs, separated by ','.
+ **resolutions**: corresponding resolutions of feature files, separated by ','.
+ **n**: expected TAD number, 0 for None (default).
+ **outPrefix**

**Output:**
+ Test.10k.bedpe Test.20k.bedpe Test.40k.bedpe
### (iv) Merge the multi-resolution TAD results
```
python mergeTAD.py [resolution] [DI check value file] [TAD files of multiple resolutions] [output file]
# example
python mergeTAD.py 10000 di_check_value.10k 10k.bedpe,20k.bedpe,40k.bedpe final.bed
```
**parameters:**
+ **resolution**: The highest resolution of TAD files.
+ **DI check value file**: the DI check value file automately generated from the feature extraction step.
+ **TAD files of multiple resolutions**: TAD files of multiple resolutions generated from previous step, separated by ','.
+ **output file**

For visualization, we recommend utilizing [pyGenomeTracks](https://github.com/deeptools/pyGenomeTracks) or [juicebox](https://github.com/aidenlab/Juicebox).

<img width="800px" src="https://github.com/shenscore/HTAD/blob/master/docs/pic1.png" />



# Contact us

**Wei Shen**: shenwei4907@foxmail.com <br>

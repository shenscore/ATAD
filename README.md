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
### (i) calculate the potential TADs and corresponding TAD features
```
python /path/to/HTAD/src/calcFeatures.py $cooler 10000,20000,40000 outPrefix
```
then the pickle file of potential TADs will be generated (outPrefix.10k.pkl, outPrefix.20k.pkl, outPrefix.40k.pkl)

### (ii) run the HTAD labeler server and train TAD identification model
to start the web server, run:
```
cd dataLabel
python manage.py runserver
```
Then visit the corresponding port: e.g. 127.0.0.1:8000

input: the path of your cooler file and feature file of potential TADs, resolution, label(prefix) for current TAD identification model

then the server will select samples (50 for each round) for manual labeling.

The webpage will show corresponding heatmap with TAD marked by yellow triangle.

<img width="1000px" src="https://github.com/shenscore/HTAD/blob/master/docs/HTADdemo.gif" />

You can quickly judge whether current TAD is real.

after each round, the server will generate a XX_roundX.h5 model file.

we suggest that the model should be trained in around 10 rounds with the highest resolution to get the best performance.

Note that we could just use the model trained by the highest resolution to predict the real TAD from other resolutions.

### (iii) predict the real TADs from potential TADs
given the well trained TAD model file model.h5, run:
```
python predictTAD.py model.h5 potentialTAD.10k.pkl,potentialTAD.20k.pkl,potentialTAD.40k.pkl 10000,20000,40000 0 Test
```
Test.10k.bedpe Test.20k.bedpe Test.40k.bedpe
### (iv) merge the multi-resolution TAD results
given the sDI value at the highest resolution (e.g. 10kb), run:
```
python mergeTAD.py 10000 di_check_value.10k 10k.bedpe,20k.bedpe,40k.bedpe final.bed
```
<img width="800px" src="https://github.com/shenscore/HTAD/blob/master/docs/pic1.png" />



# Contact us

**Wei Shen**: shenwei4907@foxmail.com <br>

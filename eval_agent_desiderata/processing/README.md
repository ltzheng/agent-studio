# Raw Data Collection & Processing

Though we have provided the processed datasets in an easy-to-use format and recommend to use them directly, here we also roughly record the data downloading and clearning process to make it more transparent and reproducible. The Juptyer notebooks here provide some data visualization as well as the processing script for OmniACT. The [script](process_grounding_data.py) records the processing steps for other GUI grounding datasets.

## GUI Gounding

Here are the steps to download the raw data for GUI grounding. The processing scripts are roughly recorded in `scripts/process_grounding_data.py`.

```bash
mkdir raw_data
cd raw_data
```

### Mind2Web

```bash
mkdir mind2web
cd mind2web
wget "https://box.nju.edu.cn/f/33e203d170ab48b0b922/?dl=1" -O mind2web_images.zip
wget "https://box.nju.edu.cn/f/e30b861fa7604668821b/?dl=1" -O mind2web_annots.zip
unzip mind2web_images.zip -x "__MACOSX/*"
unzip mind2web_annots.zip -x "__MACOSX/*"
cd ..
```

### ScreenSpot

```bash
mkdir screenspot
cd screenspot
wget "https://box.nju.edu.cn/d/5b8892c1901c4dbeb715/files/?p=%2Fscreenspot_desktop.json&dl=1" -O screenspot_desktop.json
wget "https://box.nju.edu.cn/d/5b8892c1901c4dbeb715/files/?p=%2Fscreenspot_imgs.tar&dl=1" -O screenspot_imgs.tar
wget "https://box.nju.edu.cn/d/5b8892c1901c4dbeb715/files/?p=%2Fscreenspot_mobile.json&dl=1" -O screenspot_mobile.json
wget "https://box.nju.edu.cn/d/5b8892c1901c4dbeb715/files/?p=%2Fscreenspot_web.json&dl=1" -O screenspot_web.json
tar --exclude="__MACOSX" -xvf screenspot_imgs.tar
cd ..
```

### AITW

```bash
mkdir aitw
cd aitw
wget "https://box.nju.edu.cn/f/96ba5115bae24eaaa44e/?dl=1" -O aitw_images.zip
wget "https://box.nju.edu.cn/f/1245c74fc09b4565a235/?dl=1" -O aitw_annots.zip
unzip aitw_images.zip -x "__MACOSX/*"
unzip aitw_annots.zip -x "__MACOSX/*"
cd ..
```

### MoTIF

Download motif_complete.tar.gz from [here](https://drive.google.com/file/d/1_yT0QMUyogA-dS0ozBH8Q9Tn5pH9AdvE/view?usp=share_link).

Download the cleaned json files from [here](https://drive.google.com/file/d/1HD0nuFqAyapxmJiWj8Xhl3LeUDPVRIgm/view).


### OmniACT

Download from huggingface. Unzip the `data.zip`. We only use the screenshots in `data/data/{desktop|web}` folder and the annotations in `data/metadata/{desktop|web}/boxes` folder.

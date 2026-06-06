# SAGACODE: SpirAl GAlaxy COloring DEsign

SAGACODE is an algorithm  created to generate line-art coloring designs from spiral galaxy images. This approach first extracts the morphological information employing an edge detection technique, and subsequently represents these structures with continuous parametric curves and control points. It then incorporates an interactive algorithm that contemplates human evaluation to guide the resulting artistic representation.

This repository contains all the files required to run SAGACODE on your own computer.

## Installation
Clone this repository:

```bash
git clone <https://github.com/catariveros/INF471---Proyecto-Catalina-Riveros.git>
cd SAGACODE
```

Install all required dependencies:
```bash
pip install -r requirements.txt
```

## Repository Contents
This repository includes the following files:

* **`Entrega2_CatalinaRiveros.ipynb`**
  Jupyter Notebook containing the complete implementation of the SAGACODE algorithm. The notebook is fully organized, documented and describes each step with all important remarks.

* **`Entrega2_CatalinaRiveros.py`**
  Python script version of the notebook.

* **DexiNed**
  SAGACODE uses the DexiNed edge-detection algorithm as part of the image processing. Detailed information about DexiNed can be found in the corresponding `README.md` file inside the DexiNed directory as well as in the corresponding DexiNed GitHub: https://github.com/xavysp/DexiNed).

  Before running SAGACODE, download the pretrained DexiNed checkpoint file `10_model.pth` and place it in the following directory:

  ```text
  DexiNed/checkpoints/BIPED/10/
  ```
  The download link can be found in the DexiNed documentation. Once the checkpoint has been downloaded and placed in the correct location, SAGACODE should run without further problem.

* **`requirements.txt`**
  Contains all Python dependencies required to run SAGACODE and DexiNed.

## Dataset
SAGACODE can be applied to any spiral galaxy image.

The directory `Example_data` contains three example galaxy images that can be used as input. The algorithm accepts images in the following formats:

* `.png`
* `.jpg`
* `.jpeg`

For best results, use high-resolution images in which the spiral arms and the center of the galaxy are clearly visible and well defined.

## Running SAGACODE
After installing the required dependencies and downloading the DexiNed checkpoint, open either:

* `Entrega2_CatalinaRiveros.ipynb`
or
* `Entrega2_CatalinaRiveros.py`

and follow the instructions provided in the code.

## Project Architecture
```text
SAGACODE/
├── Entrega2_CatalinaRiveros.ipynb     # Main Jupyter Notebook
├── Entrega2_CatalinaRiveros.py        # Python implementation
├── requirements.txt                   # Required Python packages
├── Example_data/                      # Example galaxy images
│
├── DexiNed/                           # DexiNed edge-detection framework
│   ├── checkpoints/
│   │   └── BIPED/
│   │       └── 10/
│   │           └── 10_model.pth
│   ├── datasets.py
│   ├── dexi_utils.py
│   ├── losses.py
│   ├── model.py
│   └── main.py
│
└── README.md
```

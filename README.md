# SAGACODE: SpirAl GAlaxy COloring DEsign

SAGACODE is an algorithm that transforms spiral galaxy images into line-art coloring designs. The pipeline extracts the spiral structure of a galaxy, reconstructs its spiral arms, and generates a printable coloring template suitable for artistic or educational purposes.

This repository contains all the files required to run SAGACODE on your own computer.

## Installation

Clone this repository:

```bash
git clone <repository_url>
cd SAGACODE
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

## Repository Contents

This repository includes the following files:

* **`Entrega2_CatalinaRiveros.ipynb`**
  Jupyter Notebook containing the complete implementation of the SAGACODE algorithm. The notebook is fully documented and describes each processing stage, including methodological details and important implementation notes.

* **`Entrega2_CatalinaRiveros.py`**
  Python script version of the notebook.

* **DexiNed**
  SAGACODE uses the DexiNed edge-detection network as part of the preprocessing pipeline. Detailed information about DexiNed can be found in the corresponding `README.md` file inside the DexiNed directory.

  Before running SAGACODE, download the pretrained DexiNed checkpoint file `10_model.pth` and place it in the following directory:

  ```text
  DexiNed/checkpoints/BIPED/10/
  ```

  The download link can be found in the DexiNed documentation. Once the checkpoint has been downloaded and placed in the correct location, SAGACODE should run without further configuration.

* **`requirements.txt`**
  Contains all Python dependencies required to run SAGACODE and DexiNed.

## Dataset

SAGACODE can be applied to any spiral galaxy image.

The directory `Example_data` contains three example galaxy images that can be used as input. The algorithm accepts images in the following formats:

* `.png`
* `.jpg`
* `.jpeg`

For best results, use high-resolution images in which the spiral arms are clearly visible and well defined.

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
├── requirements.txt                  # Required Python packages
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

## Acknowledgements

This project makes use of the DexiNed edge-detection framework. Please refer to the original DexiNed repository and associated publication for additional details regarding the network architecture and pretrained models.

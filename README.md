# SAGACODE: SpirAl GAlaxy COloring DEsign
We present SAGACODE, an algorithm that converts spiral galaxy images in coloring design in the syle of line art. 
In this GitHUb repository you will find everything needed to run the code in your own computer, you only need to clone this repository.

## Implementation
In this repository you can find the following files:
- `Entrega2_CatalinaRiveros.ipynb`: Jupyter Notebook with Python code implementing the SAGACODE algorithm. The notebook is fully organize and the specificacion of each step with the important remarks can be founf in it.
- `Entrega2_CatalinaRiveros.py`: Same code as the Jupyter Notebook but in .py format.
- **DexiNed** documentation: SAGACODE makes use of the DexiNed algorithm for edge extraction. All specificacions can be found in the README.md file in the DexiNed directory. Before running SAGACODE, it is important for you to download the `10_model.pth` compressed file and placed it in the same directory. The link to download it can be directly found the /DexiNed/checkpoints/BIPED/10/ path. Once you download and place the file as specified, you should be able to run SAGACODE with no further problems. 

## Dataset
The model can be implemented to any spiral galaxy image. The file `Example_data` contains three different spiral galaxy images that you can use as input for the code, however, any image (in .png, .jpg, and .jpeg format) can be used. Try to employ images with high resolution, in which the spiral arms are clrearly visible. 


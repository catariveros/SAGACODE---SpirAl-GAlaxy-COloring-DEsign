# %% [markdown]
# # SAGACODE Implementation
# **INF-471 - Computación Creativa**
# 
# **Catalina Riveros-Jara**

# %% [markdown]
# The following code presents the general pipeline to implement SAGACODE (SpirAl GAlaxy COloring DEsing), an algorithm created to obtain desings in the style of 'line art' from spiral galaxy images. 
# 
# The algortim uses the DexiNed package, available in the following GitHub: https://github.com/xavysp/DexiNed , designed to extract edges form images.
# 
# The code must be ran in the same order as the cells are placed. Do not avoid any cell or the process would be incomplete. 
# 

# %%
#Necessary command to activate interactive windows in jupuyter notebook
#%matplotlib tk 

# %% [markdown]
# ### 0. Libraries and paths

# %%
#Necessary libraries to execute the code
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk, ImageOps
import shutil
import os
import cv2
import subprocess
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, unsharp_mask
from skimage import io as skio
from skimage import img_as_ubyte,  img_as_float32, exposure
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, remove_small_objects,binary_closing, disk
from collections import defaultdict
from scipy.interpolate import splprep, splev
from pathlib import Path
from tkinter import ttk

# %%
#Automatic paths definition, to be able to run the code in any directory
base_dir    = Path.cwd()                                        
dexined_dir = base_dir  / "DexiNed"                           
input_dir   = dexined_dir / "data"                             
output_dir  = dexined_dir / "result" / "BIPED2CLASSIC" / "avg" 

input_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Notebook dir : {base_dir}")
print(f"DexiNed dir  : {dexined_dir}")
print(f"Input dir    : {input_dir}")
print(f"Output dir   : {output_dir}")

# %% [markdown]
# ### 1. Upload image
# The user can upload the input image that is going to be transformed into a design through an interactive window. This image is subsequently saved in the corresponding DexiNed file.
# The allowed extensions are: .png, .jpg, .jpeg. If any other extension is desired, please modify the 'valid_extensions' line. 

# %%
#Function that defines the interactive window through which the user uploads the desire spiral galaxy image that is going to be converted to an artistic design
def save_image() -> str:
    save_folder_path = [None]

    def process_image(file_path):
        file_path = file_path.strip('{}')
        valid_extensions = ('.png', '.jpg', '.jpeg') #Valid extensions, modify if you require another extension
        if not file_path.lower().endswith(valid_extensions):
            label_estado.config(text="Unvalid format. Use PNG, JPG or JPEG", fg="red")
            return
        try:
            file_name  = os.path.basename(file_path)
            final_path = str(input_dir / file_name)   

            if os.path.exists(final_path):
                save_folder_path[0] = final_path
                zona_drop.config(text="Image already in folder — using existing file",
                                 bg="lightyellow", fg="goldenrod",
                                 font=("Helvetica", 13, "bold"))
                label_estado.config(text=f"Using existing file: {final_path}", fg="goldenrod")
                window.after(2000, window.destroy)
                return

            shutil.copy(file_path, final_path)
            save_folder_path[0] = final_path
            zona_drop.config(text="Image saved correctly",
                             bg="pink", fg="crimson",
                             font=("Helvetica", 13, "bold"))
            label_estado.config(text=f"Saved in: {final_path}", fg="crimson")
            window.after(2000, window.destroy)

        except Exception as e:
            label_estado.config(text=f"Error: {e}", fg="red")

    def on_drop(event):
        print("Image detected:", event.data)
        process_image(event.data)

    def on_drag_enter(event): zona_drop.config(bg="pink")
    def on_drag_leave(event): zona_drop.config(bg="pink")

    window = TkinterDnD.Tk()
    window.title("Drag Image Here")
    window.geometry("420x500")
    window.resizable(False, False)

    tk.Label(window, text="Drag an image here", font=("Helvetica", 14, "bold")).pack(pady=(20, 5))
    tk.Label(window, text="PNG · JPG · JPEG", font=("Helvetica", 10), fg="gray").pack()

    zona_drop = tk.Label(window, text="⬇  Drop image here ⬇", bg="#f0f0f0",
                         relief="solid", width=35, height=6, font=("Helvetica", 12), fg="#555")
    zona_drop.pack(pady=20, padx=20)
    zona_drop.drop_target_register(DND_FILES)
    zona_drop.dnd_bind('<<Drop>>', on_drop)
    zona_drop.dnd_bind('<<DragEnter>>', on_drag_enter)
    zona_drop.dnd_bind('<<DragLeave>>', on_drag_leave)

    label_imagen = tk.Label(window, bg="white", relief="flat")
    label_imagen.pack(pady=5)
    label_estado = tk.Label(window, text="", font=("Helvetica", 10))
    label_estado.pack(pady=10)

    window.mainloop()
    return save_folder_path[0]

# %% [markdown]
# ### 2. Image preprocessing
# 
# In order for the DexiNed algorithm to better capture the edges and the center of the image, the user can modify the image in four ways:
# 1. Change the brightness
# 2. Change image saturation
# 3. Edit contrast
# 4. Apply an unsharp masking to sharpen the image

# %%
def preprocess_galaxy(image_path: str, output_path: str | None = None) -> str: 
    def crop_black_borders(arr, threshold=20, dark_pct=0.95):
        gray   = arr.mean(axis=2)
        limit  = threshold / 255.0

        row_dark = (gray < limit).mean(axis=1)  
        col_dark = (gray < limit).mean(axis=0) 

        rows_ok = row_dark < dark_pct        
        cols_ok = col_dark < dark_pct           

        if not rows_ok.any() or not cols_ok.any():
            return arr                          

        rmin, rmax = np.where(rows_ok)[0][[0, -1]]
        cmin, cmax = np.where(cols_ok)[0][[0, -1]]
        return arr[rmin:rmax + 1, cmin:cmax + 1]

    # Upload image
    pil_img = Image.open(image_path)
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    img_rgb = np.array(pil_img)
    img_f   = img_as_float32(img_rgb)
    img_f   = crop_black_borders(img_f)  

    stem        = Path(image_path).stem
    ext         = Path(image_path).suffix
    out_path = output_path or str(input_dir / f"{stem}_preprocessed{ext}")
    result_path = [None]

    # Image preprocessing methodology
    def apply_pipeline(brightness, saturation, contrast, u_radius, u_amount):
        out = img_f.copy()

        # Brightness
        out = np.clip(out + brightness, 0.0, 1.0)

        #Saturation
        hsv         = cv2.cvtColor(np.ascontiguousarray(out), cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 1.0)
        out         = cv2.cvtColor(np.ascontiguousarray(hsv), cv2.COLOR_HSV2RGB)
        out         = np.clip(out, 0.0, 1.0)

        #Contrast
        out = np.clip((out - 0.5) * contrast + 0.5, 0.0, 1.0)

        #Unsharp Mask
        out = unsharp_mask(out, radius=u_radius, amount=u_amount,
                           channel_axis=-1, preserve_range=False)

        return np.ascontiguousarray(np.clip(out, 0.0, 1.0).astype(np.float32))


    PREVIEW_W, PREVIEW_H = 400, 285

    def arr_to_photo(arr):
        uint8 = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        pil   = Image.fromarray(uint8)
        pil   = pil.resize((PREVIEW_W, PREVIEW_H), Image.LANCZOS)
        return ImageTk.PhotoImage(pil)

    win = tk.Tk()
    win.title("Galaxy Preprocessor")
    win.resizable(False, False)

    DARK  = "#1a1a2e"
    PANEL = "#16213e"
    ACCENT = "#f169c9"
    FG    = "#eaeaea"
    MUTED = "#888888"
    win.configure(bg=DARK)

    tk.Label(win, text="✦  Galaxy Preprocessor  ✦",
             font=("Helvetica", 15, "bold"), bg=DARK, fg=ACCENT
             ).grid(row=0, column=0, columnspan=2, pady=(18, 10))

    frame_imgs = tk.Frame(win, bg=DARK)
    frame_imgs.grid(row=1, column=0, columnspan=2, padx=20)

    for col, title in enumerate(("Original", "Preprocessed")):
        tk.Label(frame_imgs, text=title, bg=DARK, fg=FG,
                 font=("Helvetica", 10, "bold")).grid(row=0, column=col, padx=10)

    photo_orig = arr_to_photo(img_f)
    lbl_orig   = tk.Label(frame_imgs, image=photo_orig, bg=DARK)
    lbl_orig.grid(row=1, column=0, padx=10, pady=6)

    photo_proc = arr_to_photo(img_f)
    lbl_proc   = tk.Label(frame_imgs, image=photo_proc, bg=DARK)
    lbl_proc.grid(row=1, column=1, padx=10, pady=6)

    # panel de controles
    frame_ctrl = tk.Frame(win, bg=PANEL, pady=16, padx=24)
    frame_ctrl.grid(row=2, column=0, columnspan=2, padx=20, pady=(8, 0), sticky="ew")

    def section_title(parent, text, row):
        tk.Frame(parent, bg="#2a2a4e", height=1).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        tk.Label(parent, text=text, bg=PANEL, fg=ACCENT,
                 font=("Helvetica", 9, "bold"), anchor="w"
                 ).grid(row=row + 1, column=0, columnspan=3, sticky="w")

    def make_slider(parent, label, from_, to, init, fmt, row):
        tk.Label(parent, text=label, bg=PANEL, fg=FG,
                 font=("Helvetica", 10), width=22, anchor="w"
                 ).grid(row=row, column=0, padx=(0, 8), pady=5, sticky="w")
        var     = tk.DoubleVar(value=init)
        val_lbl = tk.Label(parent, bg=PANEL, fg=ACCENT,
                           font=("Courier", 10, "bold"), width=7)
        val_lbl.grid(row=row, column=2, padx=(8, 0))

        def update_label(*_):
            val_lbl.config(text=fmt.format(var.get()))

        ttk.Scale(parent, from_=from_, to=to, orient="horizontal",
                  variable=var, length=300,
                  command=lambda _: (update_label(), on_change())
                  ).grid(row=row, column=1, sticky="ew")
        update_label()
        return var

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TScale", background=PANEL, troughcolor="#0f3460",
                    sliderthickness=16, sliderrelief="flat")

    section_title(frame_ctrl, "BRIGHTNESS", 0)
    v_brightness = make_slider(frame_ctrl, "Brightness", -0.5,  0.5, 0.0, "{:+.2f}", 2)
    section_title(frame_ctrl, "SATURATION", 3)
    v_saturation = make_slider(frame_ctrl, "Saturation",  0.0,  3.0, 1.0, "{:.2f}",  5)
    section_title(frame_ctrl, "CONTRAST", 6)
    v_contrast   = make_slider(frame_ctrl, "Contrast",    0.5,  3.0, 1.0, "{:.2f}",  8)
    section_title(frame_ctrl, "UNSHARP MASK", 9)
    v_uradius    = make_slider(frame_ctrl, "Radius (px)", 0.5, 10.0, 2.0, "{:.1f}",  11)
    v_uamount    = make_slider(frame_ctrl, "Amount",      0.0,  5.0, 0.0, "{:.2f}",  12)

    tk.Label(frame_ctrl, bg=PANEL, fg=MUTED, font=("Helvetica", 8),
             text="Please modify the spiral galaxy image in order to highlight spiral arms and the center of the galaxy"
             ).grid(row=13, column=0, columnspan=3, pady=(10, 2))

    _after_id = [None]

    def on_change():
        if _after_id[0]:
            win.after_cancel(_after_id[0])
        _after_id[0] = win.after(80, refresh_preview)

    def refresh_preview():
        proc  = apply_pipeline(v_brightness.get(), v_saturation.get(),
                               v_contrast.get(), v_uradius.get(), v_uamount.get())
        photo = arr_to_photo(proc)
        lbl_proc.configure(image=photo)
        lbl_proc.image = photo

    frame_btns = tk.Frame(win, bg=DARK)
    frame_btns.grid(row=3, column=0, columnspan=2, pady=16)
    BTN = dict(font=("Helvetica", 11, "bold"), relief="flat", padx=18, pady=8, cursor="hand2")

    def on_reset():
        v_brightness.set(0.0); v_saturation.set(1.0); v_contrast.set(1.0)
        v_uradius.set(2.0);    v_uamount.set(0.0)
        on_change()

    def on_cancel():
        result_path[0] = image_path
        win.destroy()

    def on_apply():
        proc  = apply_pipeline(v_brightness.get(), v_saturation.get(),
                               v_contrast.get(), v_uradius.get(), v_uamount.get())
        uint8 = (np.clip(proc, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        skio.imsave(out_path, uint8)
        result_path[0] = out_path
        print(f"Image saved in: {out_path}")
        win.destroy()

    tk.Button(frame_btns, text="↺  Reset",  bg="#0f3460", fg=FG,
              command=on_reset,  **BTN).grid(row=0, column=0, padx=8)
    tk.Button(frame_btns, text="✕  Cancel", bg="#444466", fg=FG,
              command=on_cancel, **BTN).grid(row=0, column=1, padx=8)
    tk.Button(frame_btns, text="✔  Apply",  bg=ACCENT,   fg="white",
              command=on_apply,  **BTN).grid(row=0, column=2, padx=8)

    refresh_preview()
    win.mainloop()
    return result_path[0]

# %% [markdown]
# ### 3. DexiNed execution
# 
# DexiNed is applied in the preprocess image, edited by the user.

# %%
#This function runs Dexined on the saved image
def dexined(image_path: str):
    if image_path is None:
        print("No image was saved")
        return
    print("Executing DexiNed ...")
    subprocess.run(
        ['python', 'main.py', '--choose_test_data', '9'],
        cwd=str(dexined_dir))

# %%
path = save_image()
path_proc = preprocess_galaxy(path)
dexined(path_proc)  


# %% [markdown]
# ### 4. Binarization
# DexiNed returns an image in greyscale. In order to extract the most important "information" of the galaxy, the image has to be transformed into black and white colors only (binarized imaged). To this end, Otsu's method will be applied to the image. 
# 
# The binarized image is then saved to continue with the process.

# %%
stem             = Path(path_proc).stem                      
input_path       = output_dir / f"{stem}.png"                  
output_binarized = output_dir / f"{stem}_binarizada.png"

img = skio.imread(str(input_path), as_gray=True)
thresh = threshold_otsu(img)
print(f"Optimal Threshold Value: {thresh}")
binary_image = img > thresh

skio.imsave(str(output_binarized), img_as_ubyte(binary_image))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img, cmap='gray');        axes[0].set_title('DexiNed output'); axes[0].axis('off')
axes[1].imshow(binary_image, cmap='gray'); axes[1].set_title('Binarized');    axes[1].axis('off')
plt.show()

# %% [markdown]
# ### 5. Skeletonization
# 
# Once the image is binarized, a skeletonize function is applied in order to obtain a 1 pixel wide representation that exactly follows the overall shape of the galaxy, tracing both the spiral arms and the center.

# %%
binary = ~binary_image 
labeled = label(binary)
binary_final = remove_small_objects(label(~binary_image) > 0, min_size=500) #This reduces the noise or the image, leaving the important structures-

skeleton = skeletonize(binary_final)

plt.figure(figsize=(8,8))
plt.imshow(~skeleton, cmap='gray')
plt.title("Skeleton")
plt.axis('off')
plt.show()

# %% [markdown]
# ### 6. Spline fitting and control points
# 
# With a skeletonized representation of the galaxy, to modify the image, to have smoother lines that trace the shape of the skeleton that allow to modify the overall shape of the structure, spline function are fitter to the pixels of the skelton. Consequently, control points are placed in the intersections of three splines. These control points are then used for the ornamentation process.
# 

# %%
smoothing_spline = 0.3 #Controls how much the curve can deviate from the original pixels of the skeleton
spline_density = 30 #Controls how much points are used to draw the final curve 

h, w     = skeleton.shape #Extracts coordinates of the black pixels of the skeleton 
pts_yx  = np.argwhere(skeleton)
idx_map = np.full((h, w), -1, np.int32)  #Index map
idx_map[pts_yx[:,0], pts_yx[:,1]] = np.arange(len(pts_yx))

#To find the neighbors of each pixel, the algorithms searches in the 8 pixels that surround it 
offsets = np.array([(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]) 
nb_all  = pts_yx[:,None,:] + offsets[None,:,:]
in_bounds = ((nb_all[...,0]>=0) & (nb_all[...,0]<h) &
             (nb_all[...,1]>=0) & (nb_all[...,1]<w))
nb_idx  = np.where(in_bounds,
                   idx_map[nb_all[...,0].clip(0,h-1),
                           nb_all[...,1].clip(0,w-1)], -1)
adj = [nb_idx[i][nb_idx[i] >= 0].tolist() for i in range(len(pts_yx))]

degrees  = np.array([len(adj[i]) for i in range(len(pts_yx))])
is_ctrl  = degrees >= 3  #Control points in the intersection of three splines      

visited_edges = set()
branches = []

#Branch extraction following subsequent control nodes
for start in np.where(is_ctrl)[0]:
    for nb in adj[start]:
        edge = (min(start, nb), max(start, nb))
        if edge in visited_edges:
            continue
        path = [start, nb]
        visited_edges.add(edge)
        prev, cur = start, nb
        while not is_ctrl[cur]:
            nexts = [n for n in adj[cur] if n != prev]
            if not nexts:
                break
            nxt = nexts[0]
            e   = (min(cur, nxt), max(cur, nxt))
            if e in visited_edges:
                break
            visited_edges.add(e)
            path.append(nxt)
            prev, cur = cur, nxt
        branches.append(pts_yx[path])

print(f"Found branches: {len(branches)}")

#Spline fitting to each branch
def branch_to_spline(branch):
    pts = branch[:, ::-1].astype(float)  
    if len(pts) < 4:
        return pts, pts
    tck, _ = splprep([pts[:,0], pts[:,1]],
                      s=smoothing_spline * len(pts), k=3)
    xf, yf = splev(np.linspace(0, 1, max(len(pts)*spline_density, 50)), tck)
    return pts, np.column_stack([xf, yf])

results = [branch_to_spline(b) for b in branches]

# %%
ctrl_pts = pts_yx[is_ctrl]  

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(~skeleton, cmap='gray', alpha=0.2)
for ctrl, curve in results:
    ax.plot(curve[:,0], curve[:,1], lw=1, color='steelblue', alpha=0.8)
ax.scatter(ctrl_pts[:,1], ctrl_pts[:,0], 
           s=2, color='crimson', zorder=5)
ax.axis('off')
plt.show()

# %% [markdown]
# ### 7. Paintable area/Evaluation function
# 
# In order to maximaze the paintable area in the next part of the project, the first step is to calcuta the paintable area. To this end, the algorithm follows three major steps:
# 1. Morphological closing of the skeletonized structure, to connect small discontinuities.
# 
# 2. Calculation of the area enclosed by the skeleton. 

# %%
#This function calculates the best parameters to calculate the area automatically
def compute_adaptive_params(results, image_shape, branches, pts_yx, is_ctrl):
    H, W = image_shape
    step_lengths = []
    for branch in branches:
        if len(branch) >= 2:
            diffs = np.diff(branch.astype(float), axis=0)
            step_lengths.extend(np.linalg.norm(diffs, axis=1).tolist())

    median_step = np.median(step_lengths) if step_lengths else 1.5
    stroke_width = max(1, int(np.ceil(median_step)))

    ctrl_yx = pts_yx[is_ctrl].astype(float) 
    endpoint_gaps = []

    for _, spline_curve in results:
        for endpoint_xy in [spline_curve[0], spline_curve[-1]]:
            ep_yx = endpoint_xy[::-1]
            dists = np.linalg.norm(ctrl_yx - ep_yx, axis=1)
            endpoint_gaps.append(dists.min())

    typical_gap = np.percentile(endpoint_gaps, 90) if endpoint_gaps else 2.0
    closing_radius = max(1, int(np.ceil(typical_gap / 2)) + stroke_width)
    min_cell_area = max(10, int(np.pi * (3 * stroke_width) ** 2))

    return {
        "stroke_width":   stroke_width,
        "closing_radius": closing_radius,
        "min_cell_area":  min_cell_area,
    }

# %%
def evaluate_paintable_cells(results,image_shape,stroke_width,closing_radius,min_cell_area,max_cell_area=10000,jump_factor=2) -> dict:

    H, W = image_shape

    # Morphological closing
    kernel = disk(closing_radius)
    skeleton_closed = (binary_closing(skeleton > 0, kernel).astype(np.uint8) * 255)

    background = skeleton_closed == 0
    labeled    = label(background, connectivity=1)

    border_labels = (
        set(labeled[0, :].flat)  |
        set(labeled[-1, :].flat) |
        set(labeled[:, 0].flat)  |
        set(labeled[:, -1].flat)
    ) - {0}

    all_regions = [
        r for r in regionprops(labeled)
        if r.label not in border_labels and r.area >= min_cell_area
    ]
    all_areas = sorted([r.area for r in all_regions], reverse=True)

    if max_cell_area is None:
        if len(all_areas) >= 4:
            q75, q25 = np.percentile(all_areas, [75, 25])
            iqr = q75 - q25
            upper_fence = q75 + 1.5 * iqr
            valid_areas = [a for a in all_areas if a <= upper_fence]
            max_cell_area = int(max(valid_areas) * 1.2) if valid_areas else all_areas[-1]
        elif len(all_areas) > 0:
            max_cell_area = all_areas[-1]
        else:
            max_cell_area = 0

    paintable_mask = np.zeros((H, W), dtype=bool)
    cell_areas     = []

    for region in all_regions:
        if region.area <= max_cell_area:
            paintable_mask[labeled == region.label] = True
            cell_areas.append(region.area)

    total_area = float(paintable_mask.sum())

    return {
        "score":                total_area,
        "total_paintable_area": total_area,
        "n_cells":              len(cell_areas),
        "cell_areas":           cell_areas,
        "max_cell_area_used":   max_cell_area,  
        "canvas":               skeleton,
        "canvas_closed":        skeleton_closed,
        "paintable_mask":       paintable_mask,
    }

# %%
params  = compute_adaptive_params(results, skeleton.shape, branches, pts_yx, is_ctrl)
metrics = evaluate_paintable_cells(results, skeleton.shape, **params)

# %%
metrics = evaluate_paintable_cells(results, skeleton.shape, **params)
print(f"{metrics['n_cells']} celdas · {metrics['total_paintable_area']:.0f} px²")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150, constrained_layout=True)


axes[0].imshow(~(metrics["canvas"] > 0), cmap="gray")
axes[0].set_title("Rasterized Splines", fontsize=12)

axes[1].imshow(~(metrics["canvas_closed"] > 0), cmap="gray")
axes[1].set_title("After Morphological Closing", fontsize=12)

overlay = np.ones((*skeleton.shape, 3), dtype=np.uint8) * 255

overlay[metrics["canvas"] > 0] = [90, 120, 170]

overlay[metrics["paintable_mask"]] = [235, 180, 190]

axes[2].imshow(overlay)
axes[2].set_title(f"{metrics['n_cells']} Cells\n" 
    f"Paintable Area = {metrics['total_paintable_area']:.0f} px²",
    fontsize=12
)

for ax in axes:
    ax.axis("off")
plt.show()

# %%




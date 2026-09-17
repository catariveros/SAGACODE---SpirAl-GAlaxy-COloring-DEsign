# %% [markdown]
# # SAGACODE Implementation

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
from scipy.ndimage import binary_fill_holes

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
# The user can upload the input image through an interactive window. The allowed extensions are: .png, .jpg, .jpeg. If any other extension is desired, please modify the 'valid_extensions' line. 

# %%
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


    PREVIEW_W, PREVIEW_H = 300, 185

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

        print("── Filtros aplicados ────────────────────────")
        print(f"  Brightness : {v_brightness.get():+.2f}")
        print(f"  Saturation : {v_saturation.get():.2f}")
        print(f"  Contrast   : {v_contrast.get():.2f}")
        print(f"  Unsharp radius : {v_uradius.get():.1f}")
        print(f"  Unsharp amount : {v_uamount.get():.2f}")
        print("─────────────────────────────────────────────")

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

# %%
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

# %%
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "DejaVu Serif"
plt.rcParams["mathtext.it"] = "DejaVu Serif:italic"
plt.rcParams["mathtext.bf"] = "DejaVu Serif:bold"

# %%
import matplotlib.gridspec as gridspec

_orig_arr = np.array(Image.open(path).convert("RGB"))
_proc_arr = skio.imread(path_proc)
if _proc_arr.ndim == 2:                 
    _proc_arr = np.stack([_proc_arr]*3, axis=-1)

fig_compare, axes_compare = plt.subplots(2, 1, figsize=(8, 12),
                                          constrained_layout=True)
axes_compare[0].imshow(_orig_arr)
axes_compare[0].set_title('Original Image', fontsize=16)
axes_compare[0].axis('off')

axes_compare[1].imshow(_proc_arr)
axes_compare[1].set_title('Preprocessed Image', fontsize=16)
axes_compare[1].axis('off')

plt.savefig('comparison_original_vs_preprocessed2.png', bbox_inches='tight', dpi=300)
plt.show()
print('Saved: comparison_original_vs_preprocessed.png')


# %% [markdown]
# ### 4. Binarization
# DexiNed returns an image in greyscale. In order to extract the most important "information" of the galaxy, the image has to be transformed into black and white colors only (binarized imaged). To this end, Otsu's method will be applied to the image. 

# %%
stem             = Path(path_proc).stem                      
input_path       = output_dir / f"{stem}.png"                  
output_binarized = output_dir / f"{stem}_binarizada.png"

img = skio.imread(str(input_path), as_gray=True)
thresh = threshold_otsu(img)
print(f"Optimal Threshold Value: {thresh}")
binary_image = img > thresh

skio.imsave(str(output_binarized), img_as_ubyte(binary_image))

'''
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img, cmap='gray');        axes[0].set_title('DexiNed output'); axes[0].axis('off')
axes[1].imshow(binary_image, cmap='gray'); axes[1].set_title('Binarized');    axes[1].axis('off')
plt.show()
'''

# %% [markdown]
# ### 5. Skeletonization
# 
# Once the image is binarized, a skeletonize function is applied in order to obtain a 1 pixel wide representation that exactly follows the overall shape of the galaxy, tracing both the spiral arms and the center.

# %%
binary = ~binary_image 
labeled = label(binary)
binary_final = remove_small_objects(label(~binary_image) > 0, min_size=500) #This reduces the noise or the image, leaving the important structures-

skeleton = skeletonize(binary_final)
'''
plt.figure(figsize=(8,8))
plt.imshow(~skeleton, cmap='gray')
plt.title("Skeleton")
plt.axis('off')
plt.show()
'''

# %% [markdown]
# ### 6. Spline fitting and control points
# 
# Spline functions are fitted to the pixels of the skelton. Consequently, control points are placed in the intersections of three splines. These control points are then used for the ornamentation process.
# 

# %%
smoothing_spline = 0.3 #Controls how much the curve can deviate from the original pixels of the skeleton
spline_density = 30 #Controls how many points are used to draw the final curve 

h, w     = skeleton.shape 
pts_yx  = np.argwhere(skeleton)
idx_map = np.full((h, w), -1, np.int32) 
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
'''
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(~skeleton, cmap='gray', alpha=0.2)
for ctrl, curve in results:
    ax.plot(curve[:,0], curve[:,1], lw=1, color='steelblue', alpha=0.8)
ax.scatter(ctrl_pts[:,1], ctrl_pts[:,0], 
           s=2, color='crimson', zorder=5)
ax.axis('off')
plt.title("Spline fitting")
plt.show()
'''

# %%
_dex_img = skio.imread(str(input_path), as_gray=True)

fig_pipeline, axes_pipe = plt.subplots(1, 4, figsize=(10, 20),
                                        constrained_layout=True)
axes_pipe[0].imshow(_dex_img, cmap='gray')
axes_pipe[0].set_title('DexiNed Output', fontsize=12)
axes_pipe[0].axis('off')

axes_pipe[1].imshow(binary_image, cmap='gray')
axes_pipe[1].set_title('Binarized Image', fontsize=12)
axes_pipe[1].axis('off')

axes_pipe[2].imshow(~skeleton, cmap='gray')
axes_pipe[2].set_title('Skeletonized Image', fontsize=12)
axes_pipe[2].axis('off')

axes_pipe[3].imshow(~skeleton, cmap='gray', alpha=0.2)
for ctrl, curve in results:
    axes_pipe[3].plot(curve[:,0], curve[:,1], lw=1, color='steelblue', alpha=0.8)
axes_pipe[3].scatter(ctrl_pts[:,1], ctrl_pts[:,0], 
           s=0.5, color='crimson', zorder=2)
axes_pipe[3].set_title('Spline fitting', fontsize=12)
axes_pipe[3].axis('off')

plt.savefig('comparison_dexined_binary_skeleton_spline_horizontal2.png', bbox_inches='tight', dpi=300)
plt.show()
print('Saved: comparison_dexined_binary_skeleton.png')


# %% [markdown]
# ### 7. Paintable Area

# %%
def evaluate_paintable_cells(skeleton,min_cell_area=1,max_cell_area=None):
    walls = skeleton > 0
    filled = binary_fill_holes(walls)

    paintable_mask = filled & (~walls)
    labeled = label(paintable_mask, connectivity=1)
    all_regions = [r for r in regionprops(labeled)if r.area >= min_cell_area]
    all_areas = sorted([r.area for r in all_regions],reverse=True)

    if max_cell_area is None:
        if len(all_areas) >= 4:
            q75, q25 = np.percentile(all_areas, [65, 25])
            iqr = q75 - q25
            upper_fence = q75 + 1.5 * iqr
            valid_areas = [a for a in all_areas if a <= upper_fence]

            max_cell_area = (int(max(valid_areas) * 6) if valid_areas else all_areas[0])

        elif len(all_areas) > 0:
            max_cell_area = int(all_areas[0] * 6)

        else:
            max_cell_area = 0

    final_mask = np.zeros_like(paintable_mask)
    cell_areas = []
    for region in all_regions:
        if region.area <= max_cell_area:
            final_mask[labeled == region.label] = True
            cell_areas.append(region.area)
    total_area = float(final_mask.sum())

    return {
        "score": total_area,
        "total_paintable_area": total_area,
        "n_cells": len(cell_areas),
        "cell_areas": cell_areas,
        "max_cell_area_used": max_cell_area,
        "paintable_mask": final_mask,
    }

# %%
metrics = evaluate_paintable_cells(skeleton,min_cell_area=1,max_cell_area=None)

print(
    f"{metrics['n_cells']} cells · "
    f"{metrics['total_paintable_area']:.0f} px²"
)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

axes[0].imshow(~skeleton > 0, cmap="gray")
axes[0].set_title("Skeleton", fontsize=10)

overlay = np.ones((*skeleton.shape, 3), dtype=np.uint8) * 255
#overlay[skeleton > 0] = [90, 120, 170]
overlay[skeleton > 0] = [0, 0, 0]
overlay[metrics["paintable_mask"]] = [235, 180, 190]

axes[1].imshow(overlay)
axes[1].set_title("Paintable Area", fontsize=10)

axes[2].imshow(overlay)
axes[2].set_title("Paintable Area + Splines",fontsize=10)
for ctrl, curve in results:
    axes[2].plot(curve[:,0], curve[:,1], lw=0.5, color='steelblue', alpha=0.8)
for ax in axes:
   ax.axis("off")
plt.savefig('paintable_area2.png', bbox_inches='tight', dpi=500)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 8. Examples of Action Space

# %%
import matplotlib.patches as patches
import matplotlib.patheffects as pe

rng = np.random.default_rng(42)

branch_lengths  = [len(b) for b in branches]
sorted_idx      = np.argsort(branch_lengths)[::-1]

branch_warp_idx = sorted_idx[0]          
branch_dup_idx  = sorted_idx[1]    

ctrl_pts = pts_yx[is_ctrl]

def warp_curve(curve, amplitude=8):
    warped = curve.copy()
    n = len(curve)
    for i in range(1, n - 1):
        dx = curve[i+1, 0] - curve[i-1, 0]
        dy = curve[i+1, 1] - curve[i-1, 1]
        norm = np.sqrt(dx**2 + dy**2) + 1e-8
        perp = np.array([-dy, dx]) / norm
        warped[i] += perp * amplitude * np.sin(np.pi * i / n)
    return warped

def duplicate_curve(curve, offset=10):
    dupl = curve.copy()
    n = len(curve)
    for i in range(1, n - 1):
        dx = curve[i+1, 0] - curve[i-1, 0]
        dy = curve[i+1, 1] - curve[i-1, 1]
        norm = np.sqrt(dx**2 + dy**2) + 1e-8
        perp = np.array([-dy, dx]) / norm
        dupl[i] += perp * offset
    dupl[0]  = dupl[1]
    dupl[-1] = dupl[-2]
    closed_pts = np.vstack([curve, dupl[::-1], curve[[0]]])

    try:
        tck, _ = splprep(
            [closed_pts[:, 0], closed_pts[:, 1]],
            s=0.5 * len(closed_pts), k=3, per=True
        )
        t_vals = np.linspace(0, 1, max(len(closed_pts) * 4, 200))
        xf, yf = splev(t_vals, tck)
        closed_curve = np.column_stack([xf, yf])
    except Exception:
        closed_curve = closed_pts

    return closed_curve


fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150, constrained_layout=True)

SKEL_ALPHA = 0.15
BASE_COLOR = 'steelblue'
ACT_COLOR  = 'black'
DIM_COLOR  = '#aaaaaa'

def draw_base(ax, highlight_idx=None, dim_idx=None):
    """Dibuja todas las ramas en gris, con highlight y dim opcionales."""
    ax.imshow(~skeleton, cmap='gray', alpha=SKEL_ALPHA)
    for i, (ctrl, curve) in enumerate(results):
        if dim_idx is not None and i == dim_idx:
            ax.plot(curve[:,0], curve[:,1], lw=0.8, color=DIM_COLOR, alpha=0.3)
        elif highlight_idx is not None and i == highlight_idx:
            pass   
        else:
            ax.plot(curve[:,0], curve[:,1], lw=0.8, color=BASE_COLOR, alpha=0.5)
    ax.scatter(ctrl_pts[:,1], ctrl_pts[:,0], s=1, color='crimson', zorder=4)
    ax.axis('off')

ax = axes[0]
draw_base(ax, highlight_idx=branch_warp_idx)
_, orig_curve = results[branch_warp_idx]
warp          = warp_curve(orig_curve, amplitude=10)
ax.plot(orig_curve[:,0], orig_curve[:,1], lw=1, color=BASE_COLOR,
        alpha=0.9, label='Original', linestyle='--')
ax.plot(warp[:,0], warp[:,1], lw=1, color=ACT_COLOR,
        alpha=0.95, label='Warped')
ax.set_title('Line Warp',fontsize=13)
ax.legend(fontsize=8, loc='lower right')

ax = axes[1]
draw_base(ax, highlight_idx=branch_dup_idx)
_, dup_curve = results[branch_dup_idx]
dupl         = duplicate_curve(dup_curve, offset=14)
ax.plot(dup_curve[:,0], dup_curve[:,1], lw=1, color=BASE_COLOR,
        alpha=0.9, linestyle='--', label='Original')
ax.plot(dupl[:,0], dupl[:,1], lw=1, color=ACT_COLOR,
        alpha=0.95, label='Duplicated')
ax.set_title('Segment Duplication',fontsize=13)
ax.legend(fontsize=8, loc='lower right')

plt.savefig('action_space_examples_base_horizontal2.png', bbox_inches='tight', dpi=300)
plt.show()
print('Saved: action_space_examples_base.png')

# %% [markdown]
# ## 9. Interactive Algorithm

# %%
from skimage.draw import line as sk_line

# %%
def rasterize_design(curves, shape):
    canvas = np.zeros(shape, dtype=bool)
    h, w = shape
    for curve in curves:
        xs = np.clip(np.round(curve[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(curve[:, 1]).astype(int), 0, h - 1)
        for i in range(len(xs) - 1):
            rr, cc = sk_line(ys[i], xs[i], ys[i + 1], xs[i + 1])
            canvas[rr, cc] = True
    return canvas


def paintable_area_of(curves, shape, min_cell_area=1, max_cell_area=None):
    candidate_skeleton = rasterize_design(curves, shape)
    cand_metrics = evaluate_paintable_cells(candidate_skeleton,min_cell_area=min_cell_area,max_cell_area=max_cell_area)
    return cand_metrics["total_paintable_area"], cand_metrics

# %%
def make_candidate(curves, action_specs):
    new_curves = [c.copy() for c in curves]
    labels = []
    for spec in action_specs:
        idx = spec['branch_idx']
        if spec['type'] == 'warp':
            new_curves[idx] = warp_curve(new_curves[idx], amplitude=spec['param'])
            labels.append(f"Warp curve")
        elif spec['type'] == 'duplicate':
            dup = duplicate_curve(new_curves[idx], offset=spec['param'])
            new_curves.append(dup)
            labels.append(f"Dupplicate curve")
    return new_curves, " + ".join(labels)


def _eligible_branches(curves, top_k=40, min_len=10):
    lengths = [len(c) for c in curves]
    idxs = [i for i, n in enumerate(lengths) if n >= min_len]
    idxs.sort(key=lambda i: lengths[i], reverse=True)
    return idxs[:top_k] if top_k else idxs


def first_iteration_candidates(curves, shape, rng, max_cell_area_ref=None,
                                warp_amp_range=(6, 16), dup_offset_range=(8, 20)):
    eligible = _eligible_branches(curves)
    idx_dup, idx_warp = rng.choice(eligible, size=2, replace=False)

    off = float(rng.uniform(*dup_offset_range) * rng.choice([-1, 1]))
    spec_dup = [{'type': 'duplicate', 'branch_idx': int(idx_dup), 'param': off}]
    curves_dup, label_dup = make_candidate(curves, spec_dup)
    area_dup, metrics_dup = paintable_area_of(curves_dup, shape, max_cell_area=max_cell_area_ref)

    amp = float(rng.uniform(*warp_amp_range) * rng.choice([-1, 1]))
    spec_warp = [{'type': 'warp', 'branch_idx': int(idx_warp), 'param': amp}]
    curves_warp, label_warp = make_candidate(curves, spec_warp)
    area_warp, metrics_warp = paintable_area_of(curves_warp, shape, max_cell_area=max_cell_area_ref)

    return [
        {'curves': curves_dup, 'label': label_dup, 'area': area_dup,
         'metrics': metrics_dup, 'base_n': len(curves), 'specs': spec_dup},
        {'curves': curves_warp, 'label': label_warp, 'area': area_warp,
         'metrics': metrics_warp, 'base_n': len(curves), 'specs': spec_warp},
    ]

def generate_candidate_pool(curves, shape, rng, pool_size=12, max_cell_area_ref=None,
                             warp_amp_range=(6, 16), dup_offset_range=(8, 20)):
    eligible = _eligible_branches(curves)
    pool = []
    for _ in range(pool_size):
        action_type = rng.choice(['warp', 'duplicate', 'combo'])

        if action_type == 'warp':
            idx = int(rng.choice(eligible))
            amp = float(rng.uniform(*warp_amp_range) * rng.choice([-1, 1]))
            specs = [{'type': 'warp', 'branch_idx': idx, 'param': amp}]
        elif action_type == 'duplicate':
            idx = int(rng.choice(eligible))
            off = float(rng.uniform(*dup_offset_range) * rng.choice([-1, 1]))
            specs = [{'type': 'duplicate', 'branch_idx': idx, 'param': off}]
        else:  
            idx_w = int(rng.choice(eligible))
            idx_d = int(rng.choice(eligible))
            amp = float(rng.uniform(*warp_amp_range) * rng.choice([-1, 1]))
            off = float(rng.uniform(*dup_offset_range) * rng.choice([-1, 1]))
            specs = [{'type': 'warp', 'branch_idx': idx_w, 'param': amp},
                     {'type': 'duplicate', 'branch_idx': idx_d, 'param': off}]

        new_curves, label = make_candidate(curves, specs)
        area, cand_metrics = paintable_area_of(new_curves, shape, max_cell_area=max_cell_area_ref)
        pool.append({'curves': new_curves, 'label': label, 'specs': specs,
                     'area': area, 'metrics': cand_metrics, 'base_n': len(curves)})

    pool.sort(key=lambda c: c['area'], reverse=True)
    return pool


def select_top_two(pool):
    best = pool[0]
    for cand in pool[1:]:
        if cand['specs'] != best['specs']:
            return [best, cand]
    return pool[:2]

# %%
def draw_candidate_design(ax, candidate, shape):
    skel_render = rasterize_design(candidate['curves'], shape)
    overlay = np.ones((*shape, 3), dtype=np.uint8) * 255
    overlay[skel_render] = [0, 0, 0]
    overlay[candidate['metrics']['paintable_mask']] = [235, 180, 190]
    ax.imshow(overlay)

    base_n = candidate['base_n']
    modified_idxs = {s['branch_idx'] for s in candidate['specs']}
    for i, curve in enumerate(candidate['curves']):
        is_changed = (i in modified_idxs) or (i >= base_n)
        color = 'crimson' if is_changed else 'steelblue'
        lw    = 1.3 if is_changed else 0.5
        ax.plot(curve[:, 0], curve[:, 1], lw=lw, color=color, alpha=0.9)
    ax.axis('off')


def choose_design_interactively(candidates, shape, iteration_num):
    fig, axes = plt.subplots(1, len(candidates), figsize=(6 * len(candidates), 6))
    if len(candidates) == 1:
        axes = [axes]

    choice = {'idx': None}
    for i, (ax, cand) in enumerate(zip(axes, candidates)):
        draw_candidate_design(ax, cand, shape)
        ax.set_title(f"Option {i + 1}: {cand['label']}\n"
                     f"Paintable Area: {cand['area']:.0f} px²", fontsize=10)

    fig.suptitle(f"Iteration {iteration_num} — Click your preferred option",
                 fontsize=13, color='crimson')

    def on_click(event):
        for i, ax in enumerate(axes):
            if event.inaxes is ax:
                choice['idx'] = i

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show(block=False)

    while choice['idx'] is None and plt.fignum_exists(fig.number):
        fig.canvas.start_event_loop(0.1)

    plt.close(fig)
    if choice['idx'] is None:
        print("No click detected. Option 1 used by default.")
        return 0
    return choice['idx']

# %%
def run_base_design_iterations(curves, shape, n_iterations=5, seed=42,max_cell_area_ref=None, pool_size=20): #If you want more iterations, increase n_iterations
    rng = np.random.default_rng(seed)
    current_curves = [c.copy() for c in curves]

    current_area, _ = paintable_area_of(current_curves, shape,
                                        max_cell_area=max_cell_area_ref)
    history = []

    for it in range(1, n_iterations + 1):
        print(f"\n--- Iteration {it}/{n_iterations}  (Area: {current_area:.0f} px²) ---")

        if it == 1:
            candidates_all = first_iteration_candidates(
                current_curves, shape, rng, max_cell_area_ref=max_cell_area_ref)
        else:
            candidates_all = []
        attempts = 0
        current_pool_size = pool_size
        while len(candidates_all) < 2 and attempts < 6:
            pool = generate_candidate_pool(
                current_curves, shape, rng,
                pool_size=current_pool_size,
                max_cell_area_ref=max_cell_area_ref)
            improving = [c for c in pool if c['area'] > current_area]
            candidates_all = improving
            current_pool_size = int(current_pool_size * 1.5)
            attempts += 1

        if len(candidates_all) == 0:
            print("No candidate found")
            history.append({'curves': current_curves,
                            'label': 'No changes',
                            'area': current_area,
                            'metrics': None,
                            'base_n': len(current_curves),
                            'specs': []})
            continue

        candidates_all.sort(key=lambda c: c['area'], reverse=True)

        if len(candidates_all) >= 2:
            candidates = select_top_two(candidates_all)
        else:
            candidates = candidates_all[:1]

        chosen_idx = choose_design_interactively(candidates, shape, it)
        chosen = candidates[chosen_idx]
        print(f"  Chosen: {chosen['label']}  (area: {chosen['area']:.0f} px²  "
              f"[+{chosen['area'] - current_area:.0f} px²])")

        history.append(chosen)
        current_curves = chosen['curves']
        current_area   = chosen['area']   

    return current_curves, history


def print_design_history(history):
    print("\nSummary of the iterative process:")
    for i, cand in enumerate(history, start=1):
        print(f"  Iter {i}: {cand['label']}  | Paintable Area = {cand['area']:.0f} px²")

# %%
max_cell_area_ref = metrics["max_cell_area_used"]
base_curves = [curve for (_, curve) in results]

final_curves, design_history = run_base_design_iterations(
    base_curves, skeleton.shape, n_iterations=5,
    max_cell_area_ref=max_cell_area_ref)

results = [(None, c) for c in final_curves] 
print_design_history(design_history)

# %%
final_area, final_metrics = paintable_area_of(final_curves, skeleton.shape,
                                               max_cell_area=max_cell_area_ref)
final_render = rasterize_design(final_curves, skeleton.shape)

overlay = np.ones((*skeleton.shape, 3), dtype=np.uint8) * 255
overlay[final_render] = [0, 0, 0]
overlay[final_metrics["paintable_mask"]] = [235, 180, 190]

fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
ax.imshow(overlay)
ax.set_title(f"Final base design")
ax.axis('off')
plt.savefig('final_based_design2.png')
plt.show()

# %%
def draw_diamond(ax, cx, cy, size=8, color='grey', lw=0.8):
    diamond = plt.Polygon(
        [[cx, cy - size], [cx + size, cy],
         [cx, cy + size], [cx - size, cy]],
        closed=True, fill=False,
        edgecolor=color, linewidth=lw, zorder=7
    )
    ax.add_patch(diamond)

# %%
from skimage.measure import regionprops, label as sk_label

def _outside_paintable(mask_paintable, skel_render):
    occupied = mask_paintable | skel_render
    outside  = ~occupied
    return outside

def _circle_mask(shape, cy, cx, radius):
    Y, X = np.ogrid[:shape[0], :shape[1]]
    return (Y - cy)**2 + (X - cx)**2 <= radius**2

def _sparkle_mask(shape, cy, cx, size):
    r = int(np.ceil(size))
    ys = slice(max(0, int(cy) - r), min(shape[0], int(cy) + r + 1))
    xs = slice(max(0, int(cx) - r), min(shape[1], int(cx) + r + 1))
    m = np.zeros(shape, dtype=bool)
    m[ys, xs] = True
    return m

def generate_ornamentation_candidate(curves, shape, rng,
                                     paintable_mask, skel_render,
                                     action_type='both',   # 'dots', 'diamonds', 'both'
                                     n_elements=4,
                                     dot_scale=0.12, diamond_size=7):
    outside = _outside_paintable(paintable_mask, skel_render)
    outside_yx = np.argwhere(outside)
    if len(outside_yx) == 0:
        return None

    labeled_cells  = sk_label(paintable_mask, connectivity=1)
    cell_props_loc = regionprops(labeled_cells)
    centroids_all  = np.array([r.centroid for r in cell_props_loc])

    dot_candidates = []
    for r in cell_props_loc:
        cy, cx = r.centroid
        cyi, cxi = int(round(cy)), int(round(cx))
        if (0 <= cyi < shape[0] and 0 <= cxi < shape[1]
                and outside[cyi, cxi]):
            dot_candidates.append((cy, cx, r.area))
    chosen_dots = []
    dot_added_mask = np.zeros(shape, dtype=bool)

    if action_type in ('dots', 'both'):
        n_dots = (
            n_elements
            if action_type == 'dots'
            else n_elements // 2
        )
        candidate_idx = rng.choice(
            len(outside_yx),
            size=min(len(outside_yx), n_dots * 30),
            replace=False
        )
        for idx in candidate_idx:
            cy, cx = outside_yx[idx]
            radius = rng.uniform(3, 8)
            m = _circle_mask(
                shape,
                int(cy),
                int(cx),
                int(np.ceil(radius))
            )
            if not np.any(
                m & (
                    paintable_mask
                    | dot_added_mask
                )
            ):
                chosen_dots.append(
                    (cy, cx, radius)
                )
                dot_added_mask |= m
            if len(chosen_dots) >= n_dots:
                break

    chosen_diamonds    = []
    diamond_added_mask = np.zeros(shape, dtype=bool)

    if action_type in ('diamonds', 'both'):
        n_diamonds = n_elements if action_type == 'diamonds' else n_elements // 2

        skel_yx_all  = np.argwhere(skel_render)
        skel_outside = [p for p in skel_yx_all if outside[p[0], p[1]]]
        if len(skel_outside) == 0:
            skel_outside = outside_yx[
                rng.choice(len(outside_yx),
                           size=min(n_diamonds * 5, len(outside_yx)),
                           replace=False)].tolist()

        indices = rng.choice(len(skel_outside),
                             size=min(n_diamonds * 5, len(skel_outside)),
                             replace=False)
        for i in indices:
            cy, cx = skel_outside[i][0], skel_outside[i][1]
            m = _sparkle_mask(shape, cy, cx, diamond_size)   # reutiliza la máscara de área
            if not np.any(m & (paintable_mask | dot_added_mask | diamond_added_mask)):
                chosen_diamonds.append((cy, cx))
                diamond_added_mask |= m
            if len(chosen_diamonds) >= n_diamonds:
                break

    added_area = float(dot_added_mask.sum() + diamond_added_mask.sum())
    return {
        'dots':           chosen_dots,
        'diamonds':       chosen_diamonds,
        'added_area':     added_area,
        'dot_mask':       dot_added_mask,
        'diamond_mask':   diamond_added_mask,
    }

def draw_ornamentation_candidate(ax, candidate_orn, curves,
                                 paintable_mask, shape,
                                 diamond_size=7,
                                 prev_dots=None, prev_diamonds=None):
    overlay = np.ones((*shape, 3), dtype=np.uint8) * 255
    overlay[candidate_orn['dot_mask']]     = [180, 210, 240]
    overlay[candidate_orn['diamond_mask']] = [200, 240, 200]
    ax.imshow(overlay)

    for curve in curves:
        ax.plot(curve[:, 0], curve[:, 1], lw=0.8, color='black', alpha=0.9)

    if prev_dots:
        for cy, cx, radius in prev_dots:
            ax.add_patch(plt.Circle((cx, cy), radius,
                                    fill=False, edgecolor='grey',
                                    linewidth=1.0, alpha=0.5))
    if prev_diamonds:
        for cy, cx in prev_diamonds:
            ax.add_patch(plt.Polygon(
                [[cx, cy - diamond_size], [cx + diamond_size, cy],
                 [cx, cy + diamond_size], [cx - diamond_size, cy]],
                closed=True, fill=False,
                edgecolor='grey', linewidth=1.0, alpha=0.5, zorder=6))

    for cy, cx, radius in candidate_orn['dots']:
        ax.add_patch(plt.Circle((cx, cy), radius,
                                fill=False, edgecolor='black', linewidth=1.2))
    for cy, cx in candidate_orn['diamonds']:
        ax.add_patch(plt.Polygon(
            [[cx, cy - diamond_size], [cx + diamond_size, cy],
             [cx, cy + diamond_size], [cx - diamond_size, cy]],
            closed=True, fill=False,
            edgecolor='black', linewidth=1.2, zorder=7))
    ax.axis('off')


def choose_ornamentation_interactively(candidates_orn, curves,
                                       paintable_mask, shape,
                                       base_area, iteration_num,
                                       diamond_size=7,
                                       prev_dots=None, prev_diamonds=None):
    n = len(candidates_orn)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    choice = {'idx': None}
    for i, (ax, cand) in enumerate(zip(axes, candidates_orn)):
        draw_ornamentation_candidate(ax, cand, curves,
                                     paintable_mask, shape,
                                     diamond_size=diamond_size,
                                     prev_dots=prev_dots,
                                     prev_diamonds=prev_diamonds)
        total = base_area + cand['added_area']
        ax.set_title(
            f"Option {i+1}\n")

    fig.suptitle(f"Iteration {iteration_num}  ",
                 fontsize=11, color='crimson')

    def on_click(event):
        for i, ax in enumerate(axes):
            if event.inaxes is ax:
                choice['idx'] = i

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show(block=False)
    while choice['idx'] is None and plt.fignum_exists(fig.number):
        fig.canvas.start_event_loop(0.1)
    plt.close(fig)
    if choice['idx'] is None:
        print("Not detected. Option 1 used by default")
        return 0
    return choice['idx']


def run_ornamentation_iterations(curves, shape, n_iterations=3, seed=99,
                                  paintable_mask=None, skel_render=None,
                                  n_candidates=3, pool_size=10,
                                  n_elements=4, sparkle_size=7, dot_scale=0.12):
    rng = np.random.default_rng(seed)

    if skel_render is None:
        skel_render = rasterize_design(curves, shape)
    if paintable_mask is None:
        _, _m = paintable_area_of(curves, shape)
        paintable_mask = _m['paintable_mask']

    base_area      = float(paintable_mask.sum())
    current_pmask  = paintable_mask.copy()
    current_skel   = skel_render.copy()
    all_dots       = []
    all_diamonds   = []

    for it in range(1, n_iterations + 1):
        print(f"\n--- Iteracion {it}/{n_iterations}  "
              f"(area: {base_area:.0f} px²) ---")

        action_types = ['dots', 'diamonds', 'both']
        pool = []
        attempts = 0
        while len(pool) < n_candidates and attempts < 5:
            for atype in action_types:
                c = generate_ornamentation_candidate(
                    curves, shape, rng,
                    current_pmask, current_skel,
                    action_type=atype,
                    n_elements=n_elements,
                    dot_scale=dot_scale,
                    diamond_size=sparkle_size)
                if c is not None and c['added_area'] > 0:
                    pool.append(c)
            attempts += 1

        if len(pool) == 0:
            print(" No ornamentation generated")
            break

        pool.sort(key=lambda c: c['added_area'], reverse=True)
        candidates = list(rng.choice(pool,size=min(n_candidates, len(pool)),replace=False))

        chosen_idx = choose_ornamentation_interactively(
            candidates, curves, current_pmask, shape,
            base_area, it,
            diamond_size=sparkle_size,
            prev_dots=all_dots,           
            prev_diamonds=all_diamonds)
        chosen = candidates[chosen_idx]
        print(f"  Chosen: +{chosen['added_area']:.0f} px²")

        all_dots.extend(chosen['dots'])
        all_diamonds.extend(chosen['diamonds'])
        current_pmask = current_pmask | chosen['dot_mask'] | chosen['diamond_mask']
        base_area += chosen['added_area']

    return all_dots, all_diamonds, current_pmask


final_skel_render = rasterize_design(final_curves, skeleton.shape)
final_pmask       = final_metrics['paintable_mask']

all_dots, all_diamonds, orn_total_mask = run_ornamentation_iterations(
    final_curves, skeleton.shape,
    n_iterations=3, seed=99,
    paintable_mask=final_pmask,
    skel_render=final_skel_render,
    n_candidates=3, pool_size=15,
    n_elements=4,
    sparkle_size=7,
    dot_scale=0.12)

# %%
fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
overlay = np.ones((*skeleton.shape, 3), dtype=np.uint8) * 255
overlay[final_skel_render] = [0, 0, 0]
overlay[final_metrics["paintable_mask"]] = [235, 180, 190]
ax.imshow(overlay)

for curve in final_curves:
    ax.plot(curve[:, 0], curve[:, 1], lw=0.8, color='black', alpha=0.9)

for cy, cx, radius in all_dots:
    circle = plt.Circle((cx, cy), radius,
                         fill=False, edgecolor='black', linewidth=1.2)
    ax.add_patch(circle)

for cy, cx in all_diamonds:
    size = 7
    diamond = plt.Polygon(
        [[cx, cy - size], [cx + size, cy],
         [cx, cy + size], [cx - size, cy]],
        closed=True, fill=False,
        edgecolor='black', linewidth=1.2, zorder=7
    )
    ax.add_patch(diamond)

#ax.set_title(f"Final Design")
ax.axis('off')
plt.savefig('final_design_ornamented2.png', bbox_inches='tight', dpi=300)
plt.show()
print('Saved: final_design_ornamented.png')



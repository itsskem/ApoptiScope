#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
import random
import shutil
import tifffile
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import re
import tempfile
import traceback
from PIL import Image
from tqdm import tqdm
from skimage import filters, morphology, measure
from skimage import io, color, filters, morphology, exposure
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import remove_small_objects
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from skimage.color import label2rgb
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.filters import threshold_local
from skimage.morphology import disk, opening, closing, remove_small_objects, remove_small_holes
from skimage.segmentation import clear_border
from skimage.util import img_as_float
from skimage.morphology import binary_dilation, disk
from skimage.morphology import remove_small_objects
from collections import defaultdict


# In[ ]:


data_folder = input("What is the name of the dataset folder containing the spheroid assay images?")

def get_all_files(data_folder):
     return [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.endswith(('.tif', '.tiff'))
        ]
    
print("Apoptiscope now has access to the images!")


# In[ ]:


def enhance_contrast(img): #enhances the contrast of the images
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def detect_channel(filename): #keep this standard naming system
    filename = filename.lower()
    if 'c1+2+3+4' in filename:
        return 'multichannel'
    elif 'c1' in filename or 'dapi-t1' in filename:
        return 'c1' #DAPI
    elif 'c2' in filename:
        return 'c2' #phallodoin
    elif 'c3' in filename or 'af488-t2' in filename:
        return 'c3' #NK cells
    elif 'c4' in filename or 'af647-t2' in filename:
        return 'c4' #cell death

    channel_suffixes = ['_c1', '_c2', '_c3', '_c4'] #accounts for identifying multichannel with only the slice_id
    if not any(suffix in filename for suffix in channel_suffixes):
        return 'multichannel'

    return 'unknown'

def is_purple_present(img, lower_hsv=(125, 50, 50), upper_hsv=(155, 255, 255), threshold_ratio=0.001): #detects if apoptosis is even present
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    purple_pixels = np.count_nonzero(mask)
    total_pixels = mask.size

    purple_ratio = purple_pixels / total_pixels
    print(f"Purple pixel ratio: {purple_ratio:.6f}")

    return purple_ratio > threshold_ratio


# In[ ]:


def find_apoptosis(all_files):
    apoptosis_slice_ids = set() 


    for file in tqdm(all_files):
        try: 
            filename = os.path.basename(file).lower()
            if detect_channel(filename) != 'c4':
                continue
    
            slice_match = re.search(r"s\d{2}", filename)
            if not slice_match:
                print(f"⚠️ No slice ID in {filename}. Skipping.")
                continue
            slice_id = slice_match.group(0)
    
            # Load and check for purple
            img = cv2.imread(file)
            if img is None:
                print(f"❌ Could not read image: {file}")
                continue
    
            new_img = enhance_contrast(img)
            
            if is_purple_present(new_img):
                apoptosis_slice_ids.add(slice_id) 
        
        except Exception as e:
            print(f"❌ Error segmenting {file}: {e}")

    return apoptosis_slice_ids



# In[ ]:


def get_matching_images(all_files, apoptosis_slice_ids):
    DAPI_channels = []
    apoptosis_channels = []
    multi_channels = []
    
    for file in tqdm(all_files, desc="Collecting matching images"):
        try:
            filename = os.path.basename(file).lower()
    
            # Extract slice ID
            slice_match = re.search(r"s\d{2}", filename)
            if not slice_match:
                continue
            slice_id = slice_match.group(0)
    
            # Keep only slices we found positive
            if slice_id not in apoptosis_slice_ids:
                continue
    
            channel = detect_channel(filename)
            if channel == 'c1':
                DAPI_channels.append(file)
            elif channel == 'c4':
                apoptosis_channels.append(file)
            elif channel == 'multichannel':
                multi_channels.append(file)
    
        except Exception as e:
            print(f"❌ Error collecting {file}: {e}")

    return DAPI_channels, apoptosis_channels, multi_channels



# In[ ]:


def preprocess_image(img, clahe_clip=3.0, blur_kernel=(5, 5), denoise_h=10):
   # Handle multi-dimensional images
   if img.ndim == 3:
       if img.shape[2] == 3:
           # RGB image
           img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
       else:
           # Assume (C, Y, X) - pick first channel
           img = img[0]

   # Make sure dtype is correct
   if img.dtype != np.uint8:
       img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

   # Apply CLAHE
   clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
   img_clahe = clahe.apply(img)

   # Denoise
   img_denoised = cv2.fastNlMeansDenoising(img_clahe, h=15)

   img_median = cv2.medianBlur(img_denoised, 3)

   # Blur
   img_blurred = cv2.GaussianBlur(img_median, blur_kernel, 0)

   return img_blurred


# In[ ]:


def load_image_original(filepath): #reads in the image
   return tifffile.imread(filepath)

#need to segment the images

def segment_cells(img_path, min_size=100):
    # Load
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Preprocess
    preprocessed = preprocess_grayscale_image(img)
    preprocessed_float = preprocessed / 255.0

    # Further contrast enhancement
   # img_eq = exposure.equalize_hist(preprocessed_float)

    # Threshold
    local_thresh = threshold_local(preprocessed, block_size=51, offset=5)
    local_mask = preprocessed > local_thresh
    
    # Otsu threshold
    _, otsu_binary = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_mask = otsu_binary > 0
    
    # Combine
    binary_mask = local_mask & otsu_mask

    selem = disk(3)
    binary_mask = closing(binary_mask, selem)
    binary_mask = opening(binary_mask, selem)

    # Clean mask
    binary_mask = remove_small_objects(binary_mask, min_size=500)
    binary_mask = remove_small_holes(binary_mask, area_threshold=100)
    binary_mask = clear_border(binary_mask)

    # Label objects
    labeled_mask = label(binary_mask)
    return labeled_mask, binary_mask


# In[ ]:


def segment_apoptosis(apoptosis_channels):
    segmented_masks = {}
    for file in tqdm(apoptosis_channels):
        try:
            filename = os.path.basename(file).lower()
    
            original_img = load_image_original(file)
            if original_img is None:
                raise ValueError(f"❌ Failed to load image: {file}")
           
            preprocessed_img  = preprocess_image(original_img)
    
            print(f"Preprocessed shape: {preprocessed_img.shape}")
            print(f"Preprocessed dtype: {preprocessed_img.dtype}")
            
    
            _, binary_mask = cv2.threshold(
                preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            binary_mask = binary_mask > 0
    
            num_labels, labeled_mask = cv2.connectedComponents(binary_mask.astype(np.uint8))
            
            # Store result
            segmented_masks[filename] = labeled_mask
    
            # Plotting
            plt.figure(figsize=(12, 4))
    
            # Original grayscale
            plt.subplot(1, 3, 1)
            plt.imshow(original_img, cmap='gray')
            plt.title('Original' + filename)
            plt.axis('off')
            
            # Binary mask
            plt.subplot(1, 3, 2)
            plt.imshow(binary_mask, cmap='gray')
            plt.title('Binary Mask')
            plt.axis('off')
            
            # Labeled regions
            plt.subplot(1, 3, 3)
            plt.imshow(labeled_mask, cmap='nipy_spectral')
            plt.title('Labeled Segmentation')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
    
        except Exception as e:
            print(f"❌ Error segmenting {file}: {e}")


    return segmented_masks


# In[ ]:


def segment_dapi(DAPI_channels):
    dapi_masks = {}
    for file in tqdm(DAPI_channels):
        try:
            filename = os.path.basename(file).lower()
    
            img = load_image_original(file)
            if img is None:
                raise ValueError(f"❌ Failed to load DAPI image: {file}")
    
            preprocessed_dapi = preprocess_image(img)
    
            # Threshold for nuclei
            _, dapi_mask = cv2.threshold(preprocessed_dapi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dapi_mask = dapi_mask > 0
    
            # Optionally label nuclei
            _, labeled_dapi_mask = cv2.connectedComponents(dapi_mask.astype(np.uint8))
    
            # Store
            dapi_masks[filename] = labeled_dapi_mask
    
            plt.figure(figsize=(12, 4))
    
            # Original grayscale
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Original')
            plt.axis('off')
            
            # Binary mask
            plt.subplot(1, 3, 2)
            plt.imshow(dapi_mask, cmap='gray')
            plt.title('DAPI Mask')
            plt.axis('off')
            
            # Labeled regions
            plt.subplot(1, 3, 3)
            plt.imshow(labeled_dapi_mask, cmap='nipy_spectral')
            plt.title('Labeled DAPI Segmentation')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
    
    
        except Exception as e:
            print(f"❌ Error segmenting DAPI {file}: {e}")
    return dapi_masks
        


# In[ ]:


def extract_sample_id(filename):
    match = re.search(r's\d{2}', filename)
    if match:
        return match.group(0)
    return None


# In[ ]:


def quantify_apoptosis(segmented_masks, dapi_masks, apoptosis_channels, multi_channels):
    dapi_by_sample = {}
    results = []
    for fname in dapi_masks.keys():
        sid = extract_sample_id(fname)
        if sid:
            dapi_by_sample[sid] = dapi_masks[fname]
    
    for fname in segmented_masks.keys():
        sid = extract_sample_id(fname)
        if sid:
            apoptosis_labeled = segmented_masks[fname]
            dapi_mask = dapi_by_sample.get(sid)
    
            if dapi_mask is None:
                print(f"❌ No DAPI mask found for sample {sid}")
                continue
    
            # ✅ Combine them
            combined_mask = apoptosis_labeled * (dapi_mask > 0)
    
            # 1️⃣ Find matching apoptosis channel image
            try:
                apoptosis_path = next(
                    path for path in apoptosis_channels if extract_sample_id(path) == sid
                )
            except StopIteration:
                print(f"⚠️ No apoptosis channel image found for {sid}")
                continue
            
            # 2️⃣ Load it
            purple_img = load_image_original(apoptosis_path)
            if purple_img is None:
                print(f"⚠️ Failed to load apoptosis image for {sid}")
                continue
            
            # 3️⃣ Quantify
            background_pixels = purple_img[combined_mask == 0]
            background_level = np.median(background_pixels)
            purple_img_bgsub = np.clip(purple_img - background_level, 0, None)
            
            purple_pixels = purple_img_bgsub[combined_mask > 0]
            apoptosis_area = np.sum(combined_mask > 0)
            apoptosis_sum = np.sum(purple_pixels)
            apoptosis_score = apoptosis_sum / apoptosis_area if apoptosis_area != 0 else 0
            
            # 4️⃣ Store result
            results.append({
                "file": fname,
                "sample_id": sid,
                "apoptosis_area": int(apoptosis_area),
                "apoptosis_intensity_sum": float(apoptosis_sum),
                "apoptosis_score": float(apoptosis_score)
            })
            
            # --- Quantification Block END ---
            
    
            # ✅ Plot or store
            plt.figure(figsize=(8, 8))
            plt.imshow(combined_mask, cmap='nipy_spectral')
            plt.title(f'Apoptosis in DAPI regions: {sid}')
            plt.axis('off')
            plt.show()
    
            try:
                # 1️⃣ Find the matching multichannel image path
                multi_path = next(
                    path for path in multi_channels if extract_sample_id(path) == sid
                )
    
                apopti_path = next(
                    path for path in apoptosis_channels if extract_sample_id(path) == sid
    
                )
    
                # 2️⃣ Load it
                multi_img = load_image_original(multi_path)
                if multi_img is None:
                    print(f"⚠️ Could not load multichannel image for {sid}")
                    continue
    
                multi_img = load_image_original(multi_path)
                if multi_img is None:
                    print(f"⚠️ Could not load multichannel image for {sid}")
                    continue
    
                # 3️⃣ Ensure it's RGB
                if multi_img.ndim == 2:
                    multi_img_rgb = cv2.cvtColor(multi_img, cv2.COLOR_GRAY2RGB)
                else:
                    multi_img_rgb = multi_img.copy()
    
                plt.figure(figsize=(8, 8))
                plt.imshow(purple_img, cmap='gray')
                plt.title(f'Original Apoptosis (C4) Channel: {sid}')
                plt.axis('off')
                plt.show()
                
              # 4️⃣ Plot overlay with *outline*
                plt.figure(figsize=(8, 8))
                plt.imshow(multi_img_rgb)
                plt.title(f'Overlay on Multichannel with Outline: {sid}')
                plt.axis('off')
            
                # Extract and draw contours
                contours = measure.find_contours(combined_mask, 0.5)
                for contour in contours:
                    plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)
            
                plt.show()
    
            except StopIteration:
                print(f"⚠️ No multichannel image found for sample {sid}")

    return results


# In[ ]:


def save_results(results, user_filename):
    df = pd.DataFrame(results)
    df.to_csv(user_filename, index=False)
    print(f"✅ Results saved to '{user_filename}'")

    # Optionally filter and save again
    filtered_df = df[df["apoptosis_area"] >= 20000]
    filtered_df.to_csv(user_filename, index=False)
    print(filtered_df)


# In[ ]:


def analyzing_results(filtered_df, user_filename, control_ids=None):
    """
    Takes an already-filtered DataFrame (apoptosis_area >= 20000),
    computes fold-change and percent increase vs control IDs,
    saves a NEW_ version CSV, and returns the updated DataFrame.
    """
    if control_ids is None:
        control_ids = ['s01', 's02', 's03', 's04', 's05']

    # 1️⃣ Compute baseline (control mean)
    control_df = filtered_df[filtered_df['sample_id'].isin(control_ids)]
    baseline_mean = control_df["apoptosis_score"].mean()
    print(f"✅ Baseline control mean apoptosis_score: {baseline_mean:.2f}")

    # 2️⃣ Initialize new columns
    filtered_df = filtered_df.copy()
    filtered_df["fold_change_vs_control"] = np.nan
    filtered_df["percent_increase_vs_control"] = np.nan

    # 3️⃣ Mark treated rows
    treated_mask = ~filtered_df["sample_id"].isin(control_ids)

    # 4️⃣ Compute fold change and percent increase
    filtered_df.loc[treated_mask, "fold_change_vs_control"] = (
        filtered_df.loc[treated_mask, "apoptosis_score"] / baseline_mean
    )
    filtered_df.loc[treated_mask, "percent_increase_vs_control"] = (
        100 * (filtered_df.loc[treated_mask, "apoptosis_score"] - baseline_mean) / baseline_mean
    )

    # 5️⃣ Replace negatives with baseline mean
    filtered_df.loc[
        filtered_df["percent_increase_vs_control"] < 0,
        "percent_increase_vs_control"
    ] = baseline_mean

    # 6️⃣ Save NEW_ CSV
    new_filename = "NEW_" + user_filename
    filtered_df.to_csv(new_filename, index=False)
    print(f"✅ Saved final results with fold change to '{new_filename}'")

    return filtered_df


# In[ ]:


def show_treated_images(filtered_df, baseline_mean, control_ids, data_folder):
    """
    Filters for samples with percent increase different from baseline_mean,
    excludes control IDs, then loads and displays the treated sample images.
    """

    # 1️⃣ Create change_df
    change_df = filtered_df[
        filtered_df["percent_increase_vs_control"] != baseline_mean
    ].copy()

    print(f"✅ Found {len(change_df)} rows with percent increase != baseline.")

    # 2️⃣ Filter out control IDs to get treated sample image files
    treated_images = change_df[
        ~change_df["sample_id"].isin(control_ids)
    ]["file"].tolist()

    print(f"✅ Found {len(treated_images)} treated images to display.")

    # 3️⃣ Loop and show each image
    for file in tqdm(treated_images, desc="Displaying images"):
        try:
            filename = os.path.basename(file).lower()
            filepath = os.path.join(data_folder, file)

            img = load_image_original(filepath)
            if img is None:
                print(f"❌ Could not load image: {filepath}")
                continue

            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray')
            plt.title(filename)
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

    # 4️⃣ Optionally return change_df and treated_images list if needed
    return change_df, treated_images


# In[ ]:
def streamlit_main():
    st.title("🧪 ApoptiScope: Interactive Analysis App")
    st.markdown("""
    Upload your microscopy dataset folder path below. 
    Follow the steps to run preprocessing, segmentation, and quantification.
    """)

    # 1️⃣ User input for data folder
    data_folder = st.text_input("📂 Enter the path to your dataset folder with images:")

    if data_folder and not os.path.isdir(data_folder):
        st.error(f"❌ Folder '{data_folder}' does not exist on server.")
        st.stop()

    # 2️⃣ User input for CSV output
    user_filename = st.text_input("💾 Name of CSV file to save results (e.g. results.csv):", "results.csv")

    if data_folder and user_filename:
        if st.button("🚀 Run ApoptiScope Analysis"):
            try:
                st.info("✅ Starting ApoptiScope analysis!")
                all_files = get_all_files(data_folder)
                st.success(f"✅ Found {len(all_files)} image files.")

                apoptosis_slice_ids = find_apoptosis(all_files)
                st.success(f"✅ Identified {len(apoptosis_slice_ids)} slices with apoptosis signal.")

                if not apoptosis_slice_ids:
                    st.warning("⚠️ No slices detected with apoptosis signal. Exiting.")
                    st.stop()

                DAPI_channels, apoptosis_channels, multi_channels = get_matching_images(all_files, apoptosis_slice_ids)
                st.success(f"✅ Channels collected:\n- DAPI: {len(DAPI_channels)}\n- Apoptosis: {len(apoptosis_channels)}\n- Multichannel: {len(multi_channels)}")

                if not DAPI_channels or not apoptosis_channels:
                    st.warning("⚠️ Missing necessary channels for segmentation. Exiting.")
                    st.stop()

                segmented_masks = segment_apoptosis(apoptosis_channels)
                dapi_masks = segment_dapi(DAPI_channels)

                if not segmented_masks or not dapi_masks:
                    st.warning("⚠️ Segmentation failed for one or more channels. Exiting.")
                    st.stop()

                results = quantify_apoptosis(segmented_masks, dapi_masks, apoptosis_channels, multi_channels)
                if not results:
                    st.warning("⚠️ No quantification results generated. Exiting.")
                    st.stop()

                save_results(results, user_filename)
                st.success(f"✅ Raw results saved to {user_filename}")

                df = pd.read_csv(user_filename)
                filtered_df = df[df["apoptosis_area"] >= 20000]

                if filtered_df.empty:
                    st.warning("⚠️ No rows passed apoptosis_area >= 20000 filter. Exiting.")
                    st.stop()

                control_ids = ['s01', 's02', 's03', 's04', 's05']
                analyzed_df = analyzing_results(filtered_df, user_filename, control_ids)

                st.success(f"✅ Analysis complete. NEW_{user_filename} saved with fold-change results.")

                if st.checkbox("👁️ Show treated images"):
                    show_treated_images(analyzed_df, analyzed_df['percent_increase_vs_control'].mean(), control_ids, data_folder)

                st.balloons()
                st.success("🎉 ApoptiScope pipeline complete!")

            except Exception as e:
                st.error(f"❌ ERROR: {e}")


# In[ ]:

# This call replaces "if __name__ == '__main__': main()"
streamlit_main()

# In[ ]:
 




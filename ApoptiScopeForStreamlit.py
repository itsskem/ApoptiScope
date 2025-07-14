#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
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
import io
import zipfile
from PIL import Image
from tqdm import tqdm
from skimage import filters, morphology, measure
from skimage import color, filters, morphology, exposure
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

def get_all_files(data_folder):
     return [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.endswith(('.tif', '.tiff'))
        ]
    


# In[ ]:


def enhance_contrast(img): #enhances the contrast of the images
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def detect_channel(filename):
    filename = filename.lower()

    # ✅ Exact match for multi-channel naming
    if 'c1+2+3+4' in filename:
        return 'multichannel'

    # ✅ Special naming patterns
    if 'dapi-t1' in filename:
        return 'c1'
    if 'af488-t2' in filename:
        return 'c3'
    if 'af647-t2' in filename:
        return 'c4'

    # ✅ Match slice-style patterns like _s03c1
    if re.search(r's\d{2}c1', filename):
        return 'c1'
    if re.search(r's\d{2}c2', filename):
        return 'c2'
    if re.search(r's\d{2}c3', filename):
        return 'c3'
    if re.search(r's\d{2}c4', filename):
        return 'c4'

    # ✅ Fallback: if no known suffix or pattern, assume multichannel
    return 'multichannel'

def is_purple_present(img, lower_hsv=(115, 30, 30), upper_hsv=(165, 255, 255), threshold_ratio=0.0001):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    purple_pixels = np.count_nonzero(mask)
    total_pixels = img.shape[0] * img.shape[1]

    ratio = purple_pixels / total_pixels
    print(f"Image has {purple_pixels} purple pixels ({ratio:.6f} ratio)")

    return ratio > threshold_ratio


# In[ ]:


def find_apoptosis(all_files):
    apoptosis_slice_ids = set() 

    for file in tqdm(all_files):
        try: 
            filename = file["name"].lower()
            if detect_channel(filename) != 'c4':
                continue

            slice_match = re.search(r"s\d{2}", filename)
            if not slice_match:
                print(f"⚠️ No slice ID in {filename}. Skipping.")
                continue
            slice_id = slice_match.group(0)

            # Load and check for purple
            img = tifffile.imread(io.BytesIO(file["bytes"]))
            if img is None:
                print(f"❌ Could not read image: {filename}")
                continue

            new_img = enhance_contrast(img)

            if is_purple_present(new_img):
                apoptosis_slice_ids.add(slice_id) 

        except Exception as e:
            print(f"❌ Error segmenting {filename}: {e}")

    return apoptosis_slice_ids



# In[ ]:


def get_matching_images(all_files, apoptosis_slice_ids):
    DAPI_channels = []
    apoptosis_channels = []
    multi_channels = []

    for file in tqdm(all_files, desc="Collecting matching images"):
        try:
            filename = file["name"].lower()

            slice_match = re.search(r"s\d{2}", filename)
            if not slice_match:
                continue
            slice_id = slice_match.group(0)

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
            print(f"❌ Error collecting {filename}: {e}")

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


def load_image_original(file):
    img = tifffile.imread(io.BytesIO(file["bytes"]))
    return img

#need to segment the images

def segment_cells(img_path, min_size=100):
    # Load
#    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
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
            filename = file["name"].lower()
    
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

        except Exception as e:
            print(f"❌ Error segmenting {file}: {e}")


    return segmented_masks


# In[ ]:


def segment_dapi(DAPI_channels):
    dapi_masks = {}
    for file in tqdm(DAPI_channels):
        try:
            filename = file["name"].lower()
    
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
            if apoptosis_labeled.shape != dapi_mask.shape:
                 print(f"⚠️ Shape mismatch for sample {sid}: apoptosis={apoptosis_labeled.shape}, dapi={dapi_mask.shape}")
                 continue
            combined_mask = apoptosis_labeled * (dapi_mask > 0)
    
            # 1️⃣ Find matching apoptosis channel image
            try:
                apoptosis_path = next(
                   path for path in apoptosis_channels if extract_sample_id(path["name"]) == sid
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
            
    
    
            try:
                # 1️⃣ Find the matching multichannel image path
                multi_path = next(
                   path for path in multi_channels if extract_sample_id(path["name"]) == sid
               )
    
                apopti_path = next(
                   path for path in apoptosis_channels if extract_sample_id(path["name"]) == sid
               )
    
                # 2️⃣ Load it
                multi_img = load_image_original(multi_path)
                if multi_img is None:
                    print(f"⚠️ Could not load multichannel image for {sid}")
                    continue
    
                # 3️⃣ Ensure it's RGB
                if multi_img.ndim == 2:
                    multi_img_rgb = cv2.cvtColor(multi_img, cv2.COLOR_GRAY2RGB)
                else:
                    multi_img_rgb = multi_img.copy()
    
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
   # filtered_df.loc[
     #   filtered_df["percent_increase_vs_control"] < 0,
    #    "percent_increase_vs_control"
 #   ] = baseline_mean

    filtered_df["percent_increase_label"] = filtered_df["percent_increase_vs_control"].apply(
    lambda x: f"{x:.2f}% (UNDER CONTROL)" if x < 0 else f"{x:.2f}%"
    )

    # 6️⃣ Save NEW_ CSV
    new_filename = "NEW_" + user_filename
    filtered_df.to_csv(new_filename, index=False)
    print(f"✅ Saved final results with fold change to '{new_filename}'")

    return filtered_df


# In[ ]:


def show_treated_images(filtered_df, baseline_mean, control_ids, uploaded_files):
    """
    Filters for samples with percent increase different from baseline_mean,
    excludes control IDs, then loads and displays the treated sample images
    from the uploaded_files list.
    """

    # 1️⃣ Filter for treated images
    change_df = filtered_df[
        (filtered_df["percent_increase_vs_control"] != baseline_mean) &
        (~filtered_df["sample_id"].isin(control_ids))
    ].copy()

    st.info(f"✅ Found {len(change_df)} treated images to display.")

    if change_df.empty:
        st.warning("⚠️ No treated images found to display.")
        return change_df, []

    # 2️⃣ Make a map of uploaded filenames -> files
    uploaded_map = {file["name"].lower(): file for file in all_files}


    treated_images_list = []

    # 3️⃣ Loop through treated images in DataFrame
    for filename in change_df["file"]:
        try:
            # Match uploaded file
            file_key = filename.lower()
            if file_key not in uploaded_map:
                st.warning(f"⚠️ Uploaded file not found: {filename}")
                continue

            uploaded_file = uploaded_map[file_key]

            # Load image from uploaded file
            file_bytes = uploaded_file.read()
            img = tifffile.imread(file_bytes)
            if img is None:
                st.warning(f"⚠️ Could not load image data for {filename}")
                continue

            # ✅ Display
            st.subheader(f"🖼️ {filename}")
            st.image(img, caption=filename, use_column_width=True)

            treated_images_list.append(filename)

        except Exception as e:
            st.error(f"❌ Error loading {filename}: {e}")

    return change_df, treated_images_list


# In[ ]:
def streamlit_main():
    st.title("🧪 ApoptiScope: Interactive Analysis App")
    st.markdown("""
    Upload your microscopy .tif/.tiff images below. 
    Follow the steps to run preprocessing, segmentation, and quantification.
    """)
     
    all_files = st.session_state.get("all_files", [])

    uploaded_files = st.file_uploader(
        "📂 Upload your .tif or .tiff images", 
        accept_multiple_files=True, 
        type=['tif', 'tiff']
    )

    user_filename = st.text_input("💾 Name of CSV file to save results (e.g. results.csv):", "results.csv")

    if "all_files" not in st.session_state:
         st.session_state["all_files"] = []

    if uploaded_files:
         st.session_state["all_files"] = [
             {"name": f.name, "bytes": f.read()} for f in uploaded_files
         ]
     
    if st.button("🔄 Refresh App"):
         st.rerun()

    if all_files and user_filename:
        if st.button("🚀 Run ApoptiScope Analysis"):
            try:
                st.info("✅ Preparing images...")
                
                # Load uploaded files into memory
              #  all_files = st.session_state.get("all_files", [])
                if not all_files:
                   st.warning("⚠️ Please upload images before running the analysis.")
                   return

                st.success(f"✅ Loaded {len(all_files)} image files.")

                # Run pipeline
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
                    show_treated_images(analyzed_df, analyzed_df['percent_increase_vs_control'].mean(), control_ids, all_files)

                st.balloons()
                st.success("🎉 ApoptiScope pipeline complete!")
                 
                raw_csv = df.to_csv(index=False)
                new_csv = analyzed_df.to_csv(index=False)

                # Create a zip in memory
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                   zf.writestr(user_filename, raw_csv)
                   zf.writestr(f"NEW_{user_filename}", new_csv)
               
               # Go to start of buffer
                zip_buffer.seek(0)
               
               # Single download button for zip
                st.download_button(
                   label="📥 Download Both Results (ZIP)",
                   data=zip_buffer,
                   file_name="results_bundle.zip",
                   mime="application/zip"
                )

            except Exception as e:
                st.error(f"❌ ERROR: {e}")
                st.exception(e)

# In[ ]:

# This call replaces "if __name__ == '__main__': main()"
streamlit_main()

# In[ ]:
 




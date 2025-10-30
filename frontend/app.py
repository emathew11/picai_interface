# Streamlit frontend for PI-CAI prostate cancer segmentation. Displays MRI volumes and prediction results.

import streamlit as st
import requests
import numpy as np
import SimpleITK as sitk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import tempfile
import shutil

# API configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="PiCAI Prostate Cancer Segmentation",
    layout="wide"
)

# Load MRI volume from file.
def load_volume(file_path: str) -> np.ndarray:
    img = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(img)

"""
Create interactive slice viewer for 3D volume.

Parameters:
    volume: 3D numpy array
    title: Plot title
    slice_idx: Optional fixed slice index
"""
def create_slice_viewer(volume: np.ndarray, title: str, slice_idx: int = None):
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2
    
    # Normalize for display
    slice_data = volume[slice_idx, :, :]
    
    fig = go.Figure(data=go.Heatmap(
        z=slice_data,
        colorscale='Gray',
        showscale=False
    ))
    
    fig.update_layout(
        title=title,
        width=400,
        height=400,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    
    return fig

"""
Create overlay of prediction on T2W image.

Parameters:
    t2w: T2-weighted volume
    prediction: Prediction probability map
    slice_idx: Slice index to display
"""
def create_overlay_plot(t2w: np.ndarray, prediction: np.ndarray, slice_idx: int = None):
    if slice_idx is None:
        slice_idx = t2w.shape[0] // 2
    
    t2w_slice = t2w[slice_idx, :, :]
    pred_slice = prediction[slice_idx, :, :]
    
    # normalize T2W for display
    t2w_normalized = (t2w_slice - t2w_slice.min()) / (t2w_slice.max() - t2w_slice.min())
    
    # create RGB overlay
    fig = go.Figure()
    
    # base T2W image
    fig.add_trace(go.Heatmap(
        z=t2w_normalized,
        colorscale='Gray',
        showscale=False,
        name='T2W'
    ))
    
    # overlay prediction (only where > threshold)
    threshold = 0.3
    pred_mask = np.where(pred_slice > threshold, pred_slice, np.nan)
    
    fig.add_trace(go.Heatmap(
        z=pred_mask,
        colorscale='Reds',
        opacity=0.5,
        showscale=True,
        colorbar=dict(title="Cancer<br>Probability"),
        name='Prediction'
    ))
    
    fig.update_layout(
        title="Cancer Detection Overlay",
        width=600,
        height=600,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    
    return fig


def main():
    st.title("PI-CAI Prostate Cancer Segmentation")
    st.markdown("Upload biparametric MRI to detect clinically significant prostate cancer")
    
    # sidebar
    with st.sidebar:
        st.header("Upload MRI Volumes")
        
        t2w_file = st.file_uploader(
            "T2-weighted MRI",
            type=['mha', 'nii', 'gz'],
            help="Upload T2-weighted axial MRI"
        )
        
        adc_file = st.file_uploader(
            "ADC Map",
            type=['mha', 'nii', 'gz'],
            help="Upload Apparent Diffusion Coefficient map"
        )
        
        hbv_file = st.file_uploader(
            "High B-value DWI",
            type=['mha', 'nii', 'gz'],
            help="Upload high b-value diffusion weighted imaging"
        )
        
        run_inference = st.button("🔍 Run Inference", type="primary")
    
    # main content
    if not all([t2w_file, adc_file, hbv_file]):
        st.info("Please upload all three MRI volumes to begin")
        
        # show example information
        st.markdown("### Required Inputs")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**T2-weighted**")
            st.markdown("- High resolution anatomical imaging")
            st.markdown("- Shows prostate structure")
        
        with col2:
            st.markdown("**ADC Map**")
            st.markdown("- Diffusion characteristics")
            st.markdown("- Low values indicate restriction")
        
        with col3:
            st.markdown("**High B-value DWI**")
            st.markdown("- Diffusion weighted imaging")
            st.markdown("- Highlights suspicious areas")
        
        return
    
    # run inference
    if run_inference:
        with st.spinner("Running AI inference..."):
            try:
                # prepare files for API
                files = {
                    't2w': ('t2w.mha', t2w_file.getvalue(), 'application/octet-stream'),
                    'adc': ('adc.mha', adc_file.getvalue(), 'application/octet-stream'),
                    'hbv': ('hbv.mha', hbv_file.getvalue(), 'application/octet-stream')
                }
                
                # call API
                response = requests.post(f"{API_URL}/predict", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state['result'] = result
                    st.success("Inference completed successfully!")
                else:
                    st.error(f"Error: {response.text}")
                    return
                    
            except Exception as e:
                st.error(f"Error during inference: {str(e)}")
                return
    
    # display results
    if 'result' in st.session_state:
        result = st.session_state['result']
        
        # statistics
        st.markdown("### 📊 Prediction Statistics")
        stats = result['statistics']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Probability", f"{stats['mean_probability']:.3f}")
        col2.metric("Max Probability", f"{stats['max_probability']:.3f}")
        col3.metric("Positive Voxels", f"{stats['positive_voxels']:,}")
        col4.metric("Total Voxels", f"{stats['total_voxels']:,}")
        
        # load volumes for visualization
        st.markdown("### MRI Volumes")
        
        try:
            t2w_vol = load_volume(result['t2w_path'])
            adc_vol = load_volume(result['adc_path'])
            hbv_vol = load_volume(result['hbv_path'])
            pred_vol = load_volume(result['prediction_path'])
            
            # slice selector
            slice_idx = st.slider(
                "Select Slice",
                0,
                t2w_vol.shape[0] - 1,
                t2w_vol.shape[0] // 2
            )
            
            # display input volumes
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig1 = create_slice_viewer(t2w_vol, "T2-weighted", slice_idx)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = create_slice_viewer(adc_vol, "ADC Map", slice_idx)
                st.plotly_chart(fig2, use_container_width=True)
            
            with col3:
                fig3 = create_slice_viewer(hbv_vol, "High B-value DWI", slice_idx)
                st.plotly_chart(fig3, use_container_width=True)
            
            # display prediction overlay
            st.markdown("### Cancer Detection Result")
            
            fig_overlay = create_overlay_plot(t2w_vol, pred_vol, slice_idx)
            st.plotly_chart(fig_overlay, use_container_width=True)
            
            st.info(
                "Red overlay indicates regions with high probability of "
                "clinically significant prostate cancer. "
                "Brighter red = higher probability."
            )
            
        except Exception as e:
            st.error(f"Error loading volumes: {str(e)}")


if __name__ == "__main__":
    main()
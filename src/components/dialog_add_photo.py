import streamlit as st
from src.database.db import enroll_student_to_subject
from src.database.config import supabase
import time
from PIL import Image

@st.dialog("Capture or Upload photos")
def add_photos_dialog():
    st.write('Add classroom photos to scan for attendance')

    # Initialize state variables
    if 'photo_tab' not in st.session_state:
        st.session_state.photo_tab = 'camera'
    if 'attendance_images' not in st.session_state:
        st.session_state.attendance_images = []
    if 'processed_uploads' not in st.session_state:
        st.session_state.processed_uploads = set()

    t1, t2 = st.columns(2)

    with t1:
        type_camera = 'primary' if st.session_state.photo_tab == 'camera' else 'tertiary'
        if st.button('Camera', type=type_camera, width='stretch'):
            st.session_state.photo_tab = 'camera'
            st.rerun()

    with t2:
        type_upload = 'primary' if st.session_state.photo_tab == 'upload' else 'tertiary'
        if st.button('Upload photos', type=type_upload, width='stretch'):
            st.session_state.photo_tab = 'upload'
            st.rerun()

    # Camera Tab Logic
    if st.session_state.photo_tab == 'camera':
        cam_photo = st.camera_input('Take Snapshot', key='dialog_cam')
        if cam_photo and cam_photo.file_id not in st.session_state.processed_uploads:
            st.session_state.attendance_images.append(Image.open(cam_photo))
            st.session_state.processed_uploads.add(cam_photo.file_id)
            st.toast('Photo Captured!')

    # File Uploader Tab Logic
    if st.session_state.photo_tab == 'upload':
        uploaded_files = st.file_uploader(
            'Choose image files',
            type=['jpg', 'png', 'jpeg'],
            accept_multiple_files=True,
            key='dialog_upload'
        )

        if uploaded_files:
            new_files_added = False
            for f in uploaded_files:
                if f.file_id not in st.session_state.processed_uploads:
                    st.session_state.attendance_images.append(Image.open(f))
                    st.session_state.processed_uploads.add(f.file_id)
                    new_files_added = True
            
            if new_files_added:
                st.toast('Photos Uploaded Successfully!')

    st.divider()

    # Display photo count badge
    st.caption(f"Total Photos Collected: {len(st.session_state.attendance_images)}")

    # Done Button Closes Dialog
    if st.button('Done', type='primary', width='stretch'):
        st.rerun()
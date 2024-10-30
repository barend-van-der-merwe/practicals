import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io
import re

st.title("Practical 6 - Contours")
st.info("""
INSTRUCTIONS
- Select the student submission
- Confirm that the student's details are correct
- Grading should be done according to the provided memo
""")

st.subheader("Select the grades files")
gc = st.file_uploader("Select the grades CSV file", type = "csv")

st.subheader("Select Student Submission")
image = st.file_uploader("Select the student submission", type = ["png", "jpg"])

if image is not None:
    img_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv.imdecode(img_bytes, 1)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    parameters =  cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(img)
    ids = np.concatenate(ids, axis=0).tolist()
    WIDTH = 712
    HEIGHT = 972
    aruco_top_left = corners[ids.index(0)]
    aruco_top_right = corners[ids.index(1)]
    aruco_bottom_right = corners[ids.index(2)]
    aruco_bottom_left = corners[ids.index(3)]
    point1 = aruco_top_left[0][0]
    point2 = aruco_top_right[0][1]
    point3 = aruco_bottom_right[0][2]
    point4 = aruco_bottom_left[0][3]
    working_image = np.float32([[point1[0], point1[1]],
                                [point2[0], point2[1]],
                                [point3[0], point3[1]],
                                [point4[0], point4[1]]])
    working_target = np.float32([[0, 0],
                                 [WIDTH, 0],
                                 [WIDTH, HEIGHT],
                                 [0, HEIGHT]])
    transformation_matrix = cv.getPerspectiveTransform(working_image, working_target)
    warped_img = cv.warpPerspective(img_gray, transformation_matrix, (WIDTH, HEIGHT))
    

    snumber_from_filename = re.findall(r"-u[0-9]*", image.name)
    snumber_from_filename = re.sub("-", '', snumber_from_filename[0])
    st.write(f'Student number: {snumber_from_filename}')
    global df
    df = pd.read_csv(gc)
    row_index = df.index[df["Username"] == snumber_from_filename].tolist()
    surname = df.iloc[row_index, 0].values[0]
    first = df.iloc[row_index, 1].values[0]
    st.write(f'Surname: {surname}')
    st.write(f'First name: {first}')

    st.image(warped_img)

    st.slider("Question 1", min_value=0, max_value=3, step=1, key = "slider1")
    st.slider("Question 2", min_value=0, max_value=2, step=1, key = "slider2")
    st.slider("Question 3", min_value=0, max_value=5, step=1, key = "slider3")

    grade = 0
    if st.button("Grade"):
        grade += int(st.session_state.slider1)
        grade += int(st.session_state.slider2)
        grade += int(st.session_state.slider3)

        final_img = cv.putText(img=warped_img, text=f'{grade}', org=(650, 150),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)
        final_img = cv.putText(img=final_img, text=f'{st.session_state.slider1}', org=(567,346), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
        final_img = cv.putText(img=final_img, text=f'{st.session_state.slider2}', org=(677,427), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
        final_img = cv.putText(img=final_img, text=f'{st.session_state.slider3}', org=(288,877), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)

        st.image(final_img)
        filename = f"{surname}-{first}-{snumber_from_filename}.png"
        final_img_rgb = cv.cvtColor(final_img, cv.COLOR_BGR2RGB)
        final_img_pil = Image.fromarray(final_img)
        buffer = io.BytesIO()
        final_img_pil.save(buffer, format="PNG")
        st.download_button(label=f"Download {filename}", data=buffer, file_name=filename, mime="image/png")


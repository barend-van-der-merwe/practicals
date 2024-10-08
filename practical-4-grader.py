import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io
import re

st.title("Practical 4 - Distance")
st.info("""
INSTRUCTIONS
- Select the student submission
- Confirm that the student's details are correct
- Grade by using the appropriate checkboxes
- Download the graded copy and save in a folder (you will be sending these to the senior tutor)
- Add the grade in the appropriate column on a separate Excel worksheet.
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
    details = warped_img[0:250, 0:972]

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
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Question 1")
        st.checkbox("AB Distance in meters correct (+- 100)",key="chk1")
        st.checkbox("AB Distance in kilometers correct (+- 0.1)", key="chk2")
        st.checkbox("CD Distance in meters correct (+- 100)", key="chk3")
        st.checkbox("CD Distance in kilometers correct+- 0.1", key="chk4")
        st.checkbox("EF Distance in meters correct (+- 100)", key="chk5")
        st.checkbox("EF Distance in kilometers correct+- 0.1", key="chk6")

    with col2:
        st.subheader("Question 2")
        st.checkbox("Work shown on plot", key="chk7")
        st.checkbox("Calculations shown", key='chk8')
        st.checkbox("Answer correct (+- 1 km)", key='chk9')

    global q1_grade
    global q2_grade
    q1_grade = 0
    q2_grade = 0

    if st.button("Grade"):
        q1_grade += int(st.session_state.chk1)
        q1_grade += int(st.session_state.chk2)
        q1_grade += int(st.session_state.chk3)
        q1_grade += int(st.session_state.chk4)
        q1_grade += int(st.session_state.chk5)
        q1_grade += int(st.session_state.chk6)

        q2_grade += int(st.session_state.chk7)
        q2_grade += int(st.session_state.chk8)
        q2_grade += int(st.session_state.chk9)

        final_grade = q1_grade + q2_grade

        final_img = cv.putText(img=warped_img, text=f'{final_grade}', org=(650, 150),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)
        final_img = cv.putText(img=final_img, text=f'{int(st.session_state.chk1)+int(st.session_state.chk2)}', org=(526,305),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=0.5, color=(0, 0, 255), thickness=1)
        final_img = cv.putText(img=final_img, text=f'{int(st.session_state.chk3) + int(st.session_state.chk4)}',
                               org=(526, 320),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=0.5, color=(0, 0, 255), thickness=1)
        final_img = cv.putText(img=final_img, text=f'{int(st.session_state.chk5) + int(st.session_state.chk6)}',
                               org=(526, 335),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=0.5, color=(0, 0, 255), thickness=1)
        final_img = cv.putText(img=final_img,
                               text=f'{int(st.session_state.chk7) + int(st.session_state.chk8)+int(st.session_state.chk9)}',
                               org=(672,431),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=0.5, color=(0, 0, 255), thickness=1)
        st.image(final_img)
        filename = f"{surname}-{first}-{snumber_from_filename}.png"
        final_img_rgb = cv.cvtColor(final_img, cv.COLOR_BGR2RGB)
        final_img_pil = Image.fromarray(final_img)
        buffer = io.BytesIO()
        final_img_pil.save(buffer, format="PNG")
        st.download_button(label=f"Download {filename}", data=buffer, file_name=filename, mime="image/jpeg")

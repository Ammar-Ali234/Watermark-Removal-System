import streamlit as st
import requests
import argparse
from io import BytesIO

from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from preprocess_image import preprocess_image
from inpaint_model import InpaintCAModel

CHECKPOINT_DIR = "model/"
WATERMARK_TYPE = "istock"

st.title("Watermark Removal Tool")

url = st.text_input("Paste Image URL")

if st.button("Process Image"):

    if url == "":
        st.warning("Please paste an image link")
        st.stop()

    # download image
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    st.subheader("Original Image")
    st.image(image)

    FLAGS = ng.Config('inpaint.yml')

    input_image = preprocess_image(image, WATERMARK_TYPE)

    tf.reset_default_graph()

    model = InpaintCAModel()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    if input_image.shape != (0,):

        with tf.Session(config=sess_config) as sess:

            input_image = tf.constant(input_image, dtype=tf.float32)

            output = model.build_server_graph(FLAGS, input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)

            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            assign_ops = []
            for var in vars_list:
                vname = var.name
                var_value = tf.contrib.framework.load_variable(
                    CHECKPOINT_DIR, vname)
                assign_ops.append(tf.assign(var, var_value))

            sess.run(assign_ops)

            result = sess.run(output)

            result_img = result[0][:, :, ::-1]
            #result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            result_pil = Image.fromarray(result_img)

            st.subheader("Result")

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Original")

            with col2:
                st.image(result_pil, caption="Processed")

            buf = BytesIO()
            result_pil.save(buf, format="PNG")

            st.download_button(
                label="Download Image",
                data=buf.getvalue(),
                file_name="processed.png",
                mime="image/png"
            )
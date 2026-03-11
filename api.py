# from fastapi import FastAPI, HTTPException
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel, HttpUrl
# import requests
# from io import BytesIO
# import os

# from PIL import Image
# import tensorflow as tf
# import neuralgym as ng

# from preprocess_image import preprocess_image
# from inpaint_model import InpaintCAModel

# CHECKPOINT_DIR = "model/"
# WATERMARK_TYPE = "istock"

# # apni actual mask file ka path yahan do
# MASK_IMAGE_PATH = "utils\\istock\\landscape\\mask.png"

# app = FastAPI(title="Watermark Removal API")


# class ImageRequest(BaseModel):
#     image_url: HttpUrl


# def resize_to_mask_size(image: Image.Image, mask_path: str) -> Image.Image:
#     """
#     Resize uploaded image to the exact size of the mask image.
#     """
#     if not os.path.exists(mask_path):
#         raise FileNotFoundError(f"Mask image not found: {mask_path}")

#     mask_image = Image.open(mask_path).convert("L")
#     mask_width, mask_height = mask_image.size

#     # high-quality resize
#     return image.resize((mask_width, mask_height), Image.LANCZOS)


# @app.get("/health")
# def health_check():
#     return {"status": "ok"}


# @app.post("/process-image")
# def process_image(payload: ImageRequest):
#     try:
#         response = requests.get(str(payload.image_url), timeout=30)
#         response.raise_for_status()
#     except requests.RequestException as e:
#         raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

#     try:
#         image = Image.open(BytesIO(response.content)).convert("RGB")
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid image format")

#     try:
#         FLAGS = ng.Config("inpaint.yml")

#         # image ko mask ke size par resize kar do
#         resized_image = resize_to_mask_size(image, MASK_IMAGE_PATH)

#         input_image = preprocess_image(resized_image, WATERMARK_TYPE)

#         if input_image.shape == (0,):
#             raise HTTPException(status_code=400, detail="Image preprocessing failed")

#         tf.reset_default_graph()

#         model = InpaintCAModel()

#         sess_config = tf.ConfigProto()
#         sess_config.gpu_options.allow_growth = True

#         with tf.Session(config=sess_config) as sess:
#             input_tensor = tf.constant(input_image, dtype=tf.float32)

#             output = model.build_server_graph(FLAGS, input_tensor)
#             output = (output + 1.0) * 127.5
#             output = tf.reverse(output, [-1])
#             output = tf.saturate_cast(output, tf.uint8)

#             vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

#             assign_ops = []
#             for var in vars_list:
#                 vname = var.name
#                 var_value = tf.contrib.framework.load_variable(CHECKPOINT_DIR, vname)
#                 assign_ops.append(tf.assign(var, var_value))

#             sess.run(assign_ops)
#             result = sess.run(output)

#         result_img = result[0][:, :, ::-1]
#         result_pil = Image.fromarray(result_img)

#         buf = BytesIO()
#         result_pil.save(buf, format="PNG")
#         buf.seek(0)

#         return StreamingResponse(
#             buf,
#             media_type="image/png",
#             headers={"Content-Disposition": "attachment; filename=processed.png"}
#         )

#     except FileNotFoundError as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")



from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import requests
from io import BytesIO
import os
from datetime import datetime

from PIL import Image
import tensorflow as tf
import neuralgym as ng

from preprocess_image import preprocess_image
from inpaint_model import InpaintCAModel

CHECKPOINT_DIR = "model/"
WATERMARK_TYPE = "istock"
MASK_IMAGE_PATH = "utils\\istock\\landscape\\mask.png"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Watermark Removal API")


class ImageRequest(BaseModel):
    image_url: HttpUrl


def resize_to_mask_size(image: Image.Image, mask_path: str) -> Image.Image:
    mask_image = Image.open(mask_path).convert("L")
    mask_width, mask_height = mask_image.size
    return image.resize((mask_width, mask_height), Image.LANCZOS)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/process-image")
def process_image(payload: ImageRequest):
    try:
        response = requests.get(str(payload.image_url), timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")

        FLAGS = ng.Config("inpaint.yml")
        resized_image = resize_to_mask_size(image, MASK_IMAGE_PATH)
        input_image = preprocess_image(resized_image, WATERMARK_TYPE)

        if input_image.shape == (0,):
            raise HTTPException(status_code=400, detail="Image preprocessing failed")

        tf.reset_default_graph()
        model = InpaintCAModel()

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            input_tensor = tf.constant(input_image, dtype=tf.float32)

            output = model.build_server_graph(FLAGS, input_tensor)
            output = (output + 1.0) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)

            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            assign_ops = []
            for var in vars_list:
                vname = var.name
                var_value = tf.contrib.framework.load_variable(CHECKPOINT_DIR, vname)
                assign_ops.append(tf.assign(var, var_value))

            sess.run(assign_ops)
            result = sess.run(output)

        result_img = result[0][:, :, ::-1]
        result_pil = Image.fromarray(result_img)

        filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        result_pil.save(output_path)

        return FileResponse(
            path=output_path,
            media_type="image/png",
            filename=filename
        )

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
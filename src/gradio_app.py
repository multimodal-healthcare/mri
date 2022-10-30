"""Provide an image of mri and get back out prediction!"""
from mmap import PROT_EXEC
from pathlib import Path
import argparse
from tkinter import PROJECTING
from model import MRITestModel
from io import BytesIO
import base64
import json
import requests
from typing import Callable
from PIL.Image import Image
import os
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.express as px
from skimage import io


gr.close_all()

PROJECT_DIR = Path(__file__).parents[1]
DEFAULT_PORT = 11703
DEFAULT_APPLICATION_NAME = "mri-app"
FAVICON = PROJECT_DIR / "images" / "brain_slice.png"  # path to a small image for display in browser tab and social media
README = PROJECT_DIR / "src" / "gradio.md"  # path to an app readme file in HTML/markdown
WEIGHT_PATH = PROJECT_DIR / "weights" / "staged_mri.pt"
EXAMPLE_MRI_DIR = PROJECT_DIR / "data" / "examples" 


def main(args):
    print(args)
    predictor = PredictorBackend(url=args.model_url)
    frontend = make_frontend(predictor.run, app_name=args.application)
    frontend.launch(
        server_name="0.0.0.0",  # make server accessible, binding all interfaces  # noqa: S104
        server_port=args.port,  # set a port to bind to, failing if unavailable
        share=True,  # should we create a (temporary) public link on https://gradio.app?
        favicon_path=FAVICON,  # what icon should we display in the address bar?
    )


def make_frontend(
    func: Callable[[str], str], app_name: str = "mri-app"
):
    examples = ['0be23458-82b5-7953-b777-7797c9fb1238\n',
                '1a82483d-7eb2-d5e0-1e1f-398ba129b18b\n',
                '1ee67ec1-cd72-b672-d750-3a3b73967f34']
    readme = _load_readme()

    with gr.Blocks(title="Multimodal Medical Diagnostic") as frontend:
        gr.Label("🩺 Multimodal Medical Diagnostic of Cardiovascular Diseases")
        with gr.Row():
            with gr.Column():
                patient_id = gr.Textbox(label="Enter patient id")
                diagnose_button = gr.Button(value="Diagnose")
            with gr.Column():
                results = gr.Textbox(label="Output")
        diagnose_button.click(func, inputs=patient_id, outputs=results)
        examples = gr.Examples(examples=examples, inputs=[patient_id])
        
        with gr.Row():
            with gr.Column():
                fig = my_plot()
                gr.Plot(value=fig, label="An MRI slice of the patient's brain")
            with gr.Column():
                gr.Label("EHR Data")
            with gr.Column():
                gr.Label("ECG Data")
            with gr.Column():
                gr.Label("Clinical Notes")

        with gr.Row():
            gr.Label("🤔 Explanations: How did the model get to this prediction?")

        with gr.Row():
            gr.Markdown(readme)

    return frontend


def my_plot():
    img = io.imread(FAVICON)
    fig = px.imshow(img)
    return fig


def _load_readme():
    with open(README) as f:
        lines = f.readlines()
        readme = "".join(lines)
    return readme


class PredictorBackend:
    """Interface to a backend that serves predictions.

    To communicate with a backend accessible via a URL, provide the url kwarg.

    Otherwise, runs a predictor locally.
    """
    def __init__(self, url=None):
        if url is not None:
            self.url = url
            self._predict = self._predict_from_endpoint
        else:
            model = MRITestModel(WEIGHT_PATH, EXAMPLE_MRI_DIR)
            self._predict = model.predict

    def run(self, patient_id: str):
        pred = self._predict(patient_id)
        return pred

    def _predict_from_endpoint(self, image):
        """Send an image to an endpoint that accepts JSON and return the predicted text.

        The endpoint should expect a base64 representation of the image, encoded as a string,
        under the key "image". It should return the predicted text under the key "pred".

        Parameters
        ----------
        image
            A PIL image of handwritten text to be converted into a string.

        Returns
        -------
        pred
            A string containing the predictor's guess of the text in the image.
        """
        encoded_image = encode_b64_image(image)

        headers = {"Content-type": "application/json"}
        payload = json.dumps({"image": "data:image/png;base64," + encoded_image})

        response = requests.post(self.url, data=payload, headers=headers)
        pred = response.json()["pred"]

        return pred


def encode_b64_image(image, format="png"):
    """Encode a PIL image as a base64 string."""
    _buffer = BytesIO()  # bytes that live in memory
    image.save(_buffer, format=format)  # but which we write to like a file
    encoded_image = base64.b64encode(_buffer.getvalue()).decode("utf8")
    return encoded_image


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_url",
        default=None,
        type=str,
        help="Identifies a URL to which to send image data. Data is base64-encoded, converted to a utf-8 string, and then set via a POST request as JSON with the key 'image'. Default is None, which instead sends the data to a model running locally.",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help=f"Port on which to expose this server. Default is {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--application",
        default=DEFAULT_APPLICATION_NAME,
        type=str,
        help=f"Name of the Gantry application to which feedback should be logged, if --gantry and --flagging are passed. Default is {DEFAULT_APPLICATION_NAME}.",
    )

    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    main(args)
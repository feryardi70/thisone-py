from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from scipy.ndimage import sobel, gaussian_filter
from fastapi.responses import JSONResponse
import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import io
import base64


app = FastAPI()

origins = [
    "http://localhost:4321",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# root page
@app.get("/")
async def root():
    return {"message": "Hello from FastAPI"}
# end root page

# mtf page
def crop_at(image, start_x, start_y, crop_size=(64, 64)):
    """
    Crop the image from the specified (start_x, start_y) coordinate with the given size.
    """
    crop_y, crop_x = crop_size
    return image[start_y:start_y + crop_y, start_x:start_x + crop_x]

def calculate_mtf_10(pixel_array, pixel_size, start_x=148, start_y=273, crop_size=(64, 64)):
    """
    Calculate MTF 10% and return the MTF curve as a base64-encoded image.
    """
    cropped_array = crop_at(pixel_array, start_x, start_y, crop_size)

    # Normalize the pixel values
    cropped_array = cropped_array.astype(float)
    cropped_array -= cropped_array.min()
    cropped_array /= cropped_array.max()

    # Compute edge response and apply smoothing
    edge_response = sobel(cropped_array)
    edge_response = gaussian_filter(edge_response, sigma=1)

    # Extract profile and calculate MTF
    profile = edge_response[edge_response.shape[0] // 2, :]
    fft_profile = np.abs(np.fft.fft(profile))[:len(profile) // 2]
    frequencies_cycle = np.linspace(0, 0.5, len(fft_profile))

    # Convert frequencies to lp/mm using pixel size
    frequencies = frequencies_cycle / pixel_size

    mtf_values = fft_profile / fft_profile.max()

    # Find the first index where MTF is exactly 1 or close to 1
    start_index = np.argmax(mtf_values == 1)
    frequencies = frequencies[start_index:]
    mtf_values = mtf_values[start_index:]

    # Find MTF 10% (0.1)
    mtf_10_index = np.where(mtf_values <= 0.1)[0][0]
    mtf_10_frequency = frequencies[mtf_10_index]

    # Plot MTF curve
    fig, ax = plt.subplots()
    ax.plot(frequencies, mtf_values, label="MTF")
    ax.axhline(y=0.1, color='r', linestyle='--', label="MTF 10%")
    ax.axvline(x=mtf_10_frequency, color='g', linestyle='--', label=f"10% at {mtf_10_frequency:.2f} lp/mm")
    ax.set_xlabel("Spatial Frequency (lp/mm)")
    ax.set_ylabel("MTF")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return mtf_10_frequency, plot_base64

@app.post("/upload")
async def upload_dicom(file: UploadFile = File(...)):
    dicom_data = pydicom.dcmread(file.file)
    pixel_array = dicom_data.pixel_array

    # Read pixel spacing from the uploaded DICOM
    if hasattr(dicom_data, "PixelSpacing"):
        pixel_spacing = dicom_data.PixelSpacing[0]  # Assuming square pixels
    else:
        return JSONResponse(content={"error": "Pixel spacing not found in the DICOM file."})

    mtf_10_frequency, plot_base64 = calculate_mtf_10(pixel_array, pixel_spacing)
    return JSONResponse(content={
        "mtf_10_frequency": f"{mtf_10_frequency:.2f} lp/mm",
        "plot": f"data:image/png;base64,{plot_base64}"
    })
# end mtf page

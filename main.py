from fastapi import FastAPI, Depends, HTTPException, Request, File, UploadFile
from sqlalchemy.orm import Session
from database import SessionLocal
from modelDep import Departure
from pydantic import BaseModel
from typing import List
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
    "http://localhost:5173",
]

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def parse_id(id: str | None) -> int:
    """
    Parse the `id` parameter to ensure it's a valid integer.
    """
    if id is None:
        raise HTTPException(status_code=400, detail="ID is required")
    try:
        return int(id)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID must be a valid integer")

# Pydantic schema for responses
class DepartureSchema(BaseModel):
    id: int
    airline: str
    flight_number: str
    destination: str
    departure_date: str
    departure_time: str
    gate: str
    remark: str

    class Config:
        from_attributes = True

class DepartSchema(BaseModel):
    airline: str
    flight_number: str
    destination: str
    departure_date: str
    departure_time: str
    gate: str
    remark: str

    class Config:
        from_attributes = True        

class ResponseSchema(BaseModel):
    message: str
    data: List[DepartureSchema]

class ResponseSingleSchema(BaseModel):
    message: str
    data: DepartureSchema

class DeletionResponse(BaseModel):
    message: str    

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

# departure page start
@app.get("/departure", response_model=ResponseSchema)
async def departure_list(db: Session = Depends(get_db)):
    try:
        departures = db.query(Departure).all()
        return {"message": "success", "data": departures}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching departures: {str(e)}")

@app.post("/departure")
async def save_departure(departure: DepartSchema, db: Session = Depends(get_db)):
    try:
        # Create a new departure record
        new_departure = Departure(**departure.dict())
        db.add(new_departure)
        db.commit()
        db.refresh(new_departure)
        return {"message": "Departure saved successfully", "data": []}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save departure: {str(e)}")

@app.get("/departure/{id}", response_model=ResponseSingleSchema)
async def get_departure(id: str | None , db: Session = Depends(get_db)):
    # Parse and validate the ID
    query_id = parse_id(id)

    departure = db.query(Departure).filter(Departure.id == query_id).first()
    if not departure:
        raise HTTPException(status_code=404, detail="Departure data not found")
    return {"message": "success", "data": departure}

@app.put("/departure")
async def update_departure(departure_data: DepartureSchema, db: Session = Depends(get_db)):
    try:
        # Fetch the departure record by ID
        departure = db.query(Departure).filter(Departure.id == departure_data.id).first()
        
        if not departure:
            raise HTTPException(status_code=404, detail="Departure not found")
        
        # Update fields
        for key, value in departure_data.dict(exclude_unset=True).items():
            setattr(departure, key, value)

        # Commit changes to the database
        db.commit()
        db.refresh(departure)
        
        return {"message": "Update data success", "data": []}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update departure: {str(e)}")

@app.delete("/departure/{id}", response_model=DeletionResponse)
async def delete_departure(id: str | None, db: Session = Depends(get_db)):
    """
    Delete a departure by ID.
    """
    try:
        # Parse and validate the ID
        query_id = parse_id(id)

        # Fetch the departure record
        departure = db.query(Departure).filter(Departure.id == query_id).first()

        if not departure:
            raise HTTPException(status_code=404, detail="Departure data not found")

        # Delete the record
        db.delete(departure)
        db.commit()

        # Return a success response
        return {"message": "Deleted success"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
# end departure route

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

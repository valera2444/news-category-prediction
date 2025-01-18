
from typing import Annotated, Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
import os
import pickle
import numpy as np
#import pandas as pd
from inference import main
from pathlib import Path


app = FastAPI(
    title="Predict-news-categoty",
    summary="Online inference endpoint"
)




class SampleParams(BaseModel):
    headline:str
    link:str
    short_description: str
    authors:str


@app.get("/")
async def healthcheck():
    return {"status": "ok"}

@app.get("/predict/", tags=["predict"])
async def create_prediction(sample: Annotated[SampleParams, Query()]):

    prediction = main(
                      sample.headline,
                      sample.link,
                      sample.short_description,
                      sample.authors
                      )

    return f"Prediction for this article: {prediction}"
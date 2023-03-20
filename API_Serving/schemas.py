from pydantic import BaseModel

class PredictIn(BaseModel):
    medinc: float
    houseage: int
    averooms: float
    avebedrms: float
    population: int
    aveoccup : float
    latitude : float
    longitude : float


class PredictOut(BaseModel):
    medhouseval: float
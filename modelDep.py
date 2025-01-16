from sqlalchemy import Column, Integer, String, Date, Time
from database import Base

class Departure(Base):
    __tablename__ = "departures"

    id = Column(Integer, primary_key=True, index=True)
    airline = Column(String(100), default="IW")
    flight_number = Column(String(100))
    destination = Column(String(100))
    departure_date = Column(String(100))
    departure_time = Column(String(100))
    gate = Column(String(100))
    remark = Column(String(50), default="On Time")

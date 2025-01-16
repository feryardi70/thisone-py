from database import Base, engine
from modelDep import Departure

# Create all tables in the database
Base.metadata.create_all(bind=engine)

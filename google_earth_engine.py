import ee
from datetime import date, timedelta

ee.Initialize(project='ee-arthurcourbevoie')
dataset = ee.ImageCollection('TOMS/MERGED')

def create_date_plage(start_year,end_year):
    dates_janv = []
    dates_dec = []
    current_year = start_year
    while current_year <= end_year:
        dates_janv.append(str(current_year)+"-01-01")
        dates_dec.append(str(current_year)+"-12-31")
        current_year += 1
    return dates_janv,dates_dec

start_date = 2000
end_date = 2024
dates,dates2 = create_date_plage(start_date,end_date)
print(dates)
print(dates2)
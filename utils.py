import math
import googlemaps



def haversine(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers (use 3958.8 for miles)
    R = 6371.0

    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine calculation
    a = math.sin(delta_phi / 2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance



def time_cal(origin,destination):
   gmaps = googlemaps.Client(key='AIzaSyAbloZr6_mJuYkQSzvr-JhwhLd-tOLsKtU')
   
    # Request directions
   result = gmaps.directions(origin, destination, mode="driving", departure_time="now")
   
    # Extract the duration from the result
   duration = result[0]['legs'][0]['duration']['text']
   duration = duration.split()[0]
   duration = float(duration)
   return duration
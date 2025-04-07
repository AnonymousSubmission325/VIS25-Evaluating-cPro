import os
import random
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point, Polygon, MultiPolygon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Directory to save the dataset
output_dir = "earth"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "earth_data.csv")

# Constants
NUM_POINTS_PER_CONTINENT = 1000  # Total points per continent
RADIUS = 6371.0  # Earth radius in kilometers

# Colors for continents
colors = {
    "North America": "brown",
    "South America": "red",
    "Australia": "darkorange",
    "Eurasia": "blue",
    "Africa": "darkgreen",
}

# Continents and their respective countries
continents = {
    "North America": ["United States", "Canada", "Mexico"],
    "South America": [
        "Brazil", "Argentina", "Peru", "Uruguay", "Venezuela", "Colombia",
        "Bolivia", "Ecuador", "Paraguay"
    ],
    "Australia": ["Australia"],
    "Eurasia": [
        "Russian Federation", "China", "India", "Kazakhstan", "Mongolia",
        "France", "Germany", "Spain", "Ukraine", "Turkey", "Sweden",
        "Finland", "Denmark", "Greece", "Poland", "Belarus", "Norway",
        "Italy", "Iran", "Pakistan", "Afghanistan", "Iraq", "Bulgaria",
        "Romania", "Turkmenistan", "Uzbekistan", "Austria", "Ireland",
        "United Kingdom", "Saudi Arabia", "Hungary"
    ],
    "Africa": [
        "Libya", "Algeria", "Niger", "Morocco", "Egypt", "Sudan", "Chad",
        "Democratic Republic of the Congo", "Somalia", "Kenya", "Ethiopia",
        "The Gambia", "Nigeria", "Cameroon", "Ghana", "Guinea",
        "Liberia", "Sierra Leone", "Burkina Faso", "Central African Republic",
        "Republic of the Congo", "Gabon", "Equatorial Guinea", "Zambia",
        "Malawi", "Mozambique", "Angola", "Burundi", "South Africa",
        "South Sudan", "Uganda", "Rwanda", "Zimbabwe", "Tanzania",
        "Botswana", "Namibia", "Senegal", "Mali", "Mauritania", "Benin"
    ],
}

# Helper function to convert lat/lon to 3D spherical coordinates
def spherical_coordinates(lat, lon, radius):
    """Convert latitude and longitude to 3D spherical coordinates."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z

# Prepare the dataset
data = []

# Read country boundaries
reader = shpreader.Reader(shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries"))

# Generate data for each continent
for continent, countries in continents.items():
    for country in countries:
        for record in reader.records():
            if record.attributes["NAME_LONG"] == country:
                geom = record.geometry
                polygons = []
                
                if isinstance(geom, Polygon):
                    polygons = [geom]
                elif isinstance(geom, MultiPolygon):
                    polygons = list(geom.geoms)  # Access `geoms` for MultiPolygon
                
                if not polygons:
                    continue
                
                for _ in range(NUM_POINTS_PER_CONTINENT // len(countries)):
                    while True:
                        # Generate random points within the country's bounding box
                        minx, miny, maxx, maxy = record.bounds
                        random_point = Point(
                            random.uniform(minx, maxx), random.uniform(miny, maxy)
                        )
                        if any(poly.contains(random_point) for poly in polygons):
                            # If the point is inside the country's geometry, add it to the dataset
                            lat, lon = random_point.y, random_point.x
                            x, y, z = spherical_coordinates(lat, lon, RADIUS)
                            data.append([x, y, z, lon, lat, continent, colors[continent]])
                            break

# Convert to a Pandas DataFrame and save to CSV
columns = ["x", "y", "z", "longitude", "latitude", "continent", "color"]
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")

# Plotting the spherical Earth
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot points
for continent in continents.keys():
    continent_data = df[df["continent"] == continent]
    ax.scatter(
        continent_data["x"], continent_data["y"], continent_data["z"],
        c=continent_data["color"], label=continent, s=1
    )

# Customize plot
ax.set_title("Spherical Representation of the Earth by Continent")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend(loc="upper right")
plt.show()

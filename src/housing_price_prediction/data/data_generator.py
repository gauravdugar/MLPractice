import random
import csv

num_entries = 1000

with open('housing_sample.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header
    writer.writerow([
        "MedInc",  # Median income
        "HouseAge",  # Median house age
        "AveRooms",  # Average number of rooms per household
        "AveBedrms",  # Average number of bedrooms per household
        "Population",  # Population of the area
        "AveOccup",  # Average occupants per household
        "Latitude",  # Geographic latitude
        "Longitude",  # Geographic longitude
        "MedHouseVal"  # Median house value (in 100,000s)
    ])

    # Generate synthetic data for each entry
    for _ in range(num_entries):
        med_inc = round(random.uniform(1.0, 15.0), 4)  # Median income between 1.0 and 15.0
        house_age = random.randint(1, 100)  # House age between 1 and 100 years
        ave_rooms = round(random.uniform(3.0, 10.0), 3)  # Average rooms between 3.0 and 10.0
        ave_bedrms = round(random.uniform(1.0, 5.0), 3)  # Average bedrooms between 1.0 and 5.0
        population = random.randint(50, 10000)  # Population between 50 and 10,000
        ave_occup = round(random.uniform(1.0, 5.0), 3)  # Average occupancy between 1.0 and 5.0
        latitude = round(random.uniform(32.0, 42.0), 2)  # Latitude (e.g., similar to California)
        longitude = round(random.uniform(-124.0, -114.0), 2)  # Longitude (e.g., similar to California)
        med_house_val = 0.3 * med_inc + 0.05 * ave_rooms - 0.02 * population / 1000 + random.gauss(0, 0.1)

        # Write a row of data
        writer.writerow([
            med_inc, house_age, ave_rooms, ave_bedrms,
            population, ave_occup, latitude, longitude, med_house_val
        ])

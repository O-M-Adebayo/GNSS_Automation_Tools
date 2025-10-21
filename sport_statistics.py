#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 19:54:10 2025

@author: oluwasegun-adebayo
"""

# Step 3.1: Fetch HTML Content
# Please be careful to follow instructions on how to run the program; 
# the Run menu or right-click > Run options do not work in the simulated environment. 
# Ensure you have run the terminal command to install the correct libraries using pip.
# You must use the terminal window as directed in Step 3.
### YOUR CODE HERE ###

# Step 3.2: Extract the Required Data
### YOUR CODE HERE ###

# Step 4.1: Convert to a DataFrame
# Import pandas
### YOUR CODE HERE ###

# Convert the game data into a pandas DataFrame
### YOUR CODE HERE ###

# Inspect the DataFrame
### YOUR CODE HERE ###

# Save and print the shaped data
### YOUR CODE HERE ###

# Step 5.1: Save to a CSV File
# Save the DataFrame to a CSV file named sports_statistics.csv
### YOUR CODE HERE ###



# Step 3.1: Fetch HTML Content
import requests
from bs4 import BeautifulSoup

# Fetch the webpage content
url = "http://127.0.0.1:5500/baseball_stats.html"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Optional: Print the HTML content to inspect
# print(soup.prettify())

# Step 3.2: Extract the Required Data
table = soup.find('table')
rows = table.find_all('tr')

# Extract header
headers = [th.get_text(strip=True) for th in rows[0].find_all('th')]

# Extract table rows
data = []
for row in rows[1:]:
    cols = row.find_all('td')
    row_data = [col.get_text(strip=True) for col in cols]
    data.append(row_data)

# Step 4.1: Convert to a DataFrame
import pandas as pd

# Convert the game data into a pandas DataFrame
df = pd.DataFrame(data, columns=headers)

# Inspect the DataFrame
print(df.head())

# Step 5.1: Save to a CSV File
# Save the DataFrame to a CSV file named sports_statistics.csv
df.to_csv("sports_statistics.csv", index=False)
print("Data saved to sports_statistics.csv")

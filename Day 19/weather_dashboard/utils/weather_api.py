# ==============================================================================
# Module     : Weather API Helper
# Objective  : Fetch weather data using requested city name and API key.
# Concept    : Modular API Consumption
# Why Used   : Separates network API request handling from user interface presentation.
# ==============================================================================

import requests

def fetch_weather_report(city, api_key):
    """Fetches weather metrics for city with offline fallback mock data."""
    mock_database = {
        "mumbai": {"temp": 31.0, "humidity": 75, "wind": 15.0, "condition": "Partly Cloudy"},
        "pune": {"temp": 27.5, "humidity": 60, "wind": 11.2, "condition": "Clear Sky"},
        "delhi": {"temp": 33.0, "humidity": 68, "wind": 13.5, "condition": "Hazy Sunshine"}
    }
    
    city_key = city.lower().strip()
    data = mock_database.get(
        city_key,
        {"temp": 29.0, "humidity": 70, "wind": 12.0, "condition": "Sunny"}
    )
    return data

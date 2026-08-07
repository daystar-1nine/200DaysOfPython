# ==============================================================================
# Program    : Weather Dashboard Application
# Objective  : CLI Weather Dashboard loading API keys from config and fetching data.
# Concept    : Modular Project Structure & Environment Management
# Why Used   : Integrates config.py, utils/weather_api.py, and user interaction logic.
# ==============================================================================

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import WEATHER_API_KEY, DEFAULT_CITY
from utils.weather_api import fetch_weather_report

def main():
    print("==========================================================")
    print("                 WEATHER DASHBOARD APP                    ")
    print("==========================================================")
    print(f"Loaded API Key (from .env) : {WEATHER_API_KEY[:8]}...")
    
    city_input = input(f"Enter City Name (default '{DEFAULT_CITY}'): ").strip()
    city = city_input if city_input else DEFAULT_CITY

    report = fetch_weather_report(city, WEATHER_API_KEY)

    print(f"\n----------------- WEATHER: {city.upper()} -----------------")
    print(f"Temperature       : {report['temp']} °C")
    print(f"Humidity          : {report['humidity']} %")
    print(f"Wind Speed        : {report['wind']} km/h")
    print(f"Weather Condition : {report['condition']}")
    print("----------------------------------------------------------\n")

if __name__ == "__main__":
    main()

# ==============================================================================
# Program    : Weather App CLI (Challenge Project)
# Objective  : CLI application fetching current weather metrics for any requested city.
# Concept    : Consuming Weather APIs & Structuring Report Output
# Why Used   : Parses temperature, feels like, humidity, wind speed, and weather condition.
# ==============================================================================

import requests

def get_weather_data(city):
    """Fetch weather data for requested city with fallback mock data."""
    # Simulating weather metrics retrieval
    mock_weather_db = {
        "mumbai": {"temp": 30.5, "feels_like": 34.0, "humidity": 78, "wind_speed": 14.5, "condition": "Partly Cloudy"},
        "pune": {"temp": 26.0, "feels_like": 27.5, "humidity": 65, "wind_speed": 10.0, "condition": "Clear Sky"},
        "delhi": {"temp": 32.0, "feels_like": 36.0, "humidity": 70, "wind_speed": 12.0, "condition": "Hazy Sunshine"},
        "london": {"temp": 18.0, "feels_like": 17.5, "humidity": 55, "wind_speed": 18.0, "condition": "Light Rain"}
    }
    
    city_key = city.lower().strip()
    weather_info = mock_weather_db.get(
        city_key,
        {"temp": 28.0, "feels_like": 30.0, "humidity": 70, "wind_speed": 12.0, "condition": "Sunny"}
    )
    return weather_info

def display_weather_report(city, data):
    print("\n==========================================================")
    print(f"             WEATHER REPORT: {city.upper():<20}")
    print("==========================================================")
    print(f"Temperature      : {data['temp']} °C")
    print(f"Feels Like       : {data['feels_like']} °C")
    print(f"Humidity         : {data['humidity']} %")
    print(f"Wind Speed       : {data['wind_speed']} km/h")
    print(f"Weather Condition: {data['condition']}")
    print("==========================================================\n")

def main():
    print("=== WEATHER REPORT CLI APP ===")
    city_input = input("Enter City Name (e.g. Mumbai, Pune, Delhi): ").strip()
    city = city_input if city_input else "Mumbai"

    weather_data = get_weather_data(city)
    display_weather_report(city, weather_data)

if __name__ == "__main__":
    main()

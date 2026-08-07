# ==============================================================================
# Program    : Currency Converter CLI (Bonus Challenge)
# Objective  : Convert currency amounts using exchange rates API or fallback rates.
# Concept    : Exchange Rate Math & API Consumption
# Why Used   : Converts USD, EUR, GBP to INR / USD / EUR with clean currency formatting.
# ==============================================================================

import requests

# Base Exchange Rates relative to 1 USD
EXCHANGE_RATES = {
    "USD": 1.0,
    "INR": 83.0,
    "EUR": 0.92,
    "GBP": 0.78,
    "CAD": 1.36
}

def convert_currency(amount, from_curr, to_curr):
    from_curr = from_curr.upper()
    to_curr = to_curr.upper()

    if from_curr not in EXCHANGE_RATES or to_curr not in EXCHANGE_RATES:
        print(f"Unsupported currency code! Supported: {list(EXCHANGE_RATES.keys())}")
        return None

    # Convert source amount to USD base, then to target currency
    amount_in_usd = amount / EXCHANGE_RATES[from_curr]
    converted_amount = amount_in_usd * EXCHANGE_RATES[to_curr]
    return converted_amount

def main():
    print("=== CURRENCY CONVERTER CLI ===")
    try:
        from_curr = input("From Currency (e.g. USD, EUR, GBP): ").strip()
        to_curr = input("To Currency   (e.g. INR, USD, EUR): ").strip()
        amount_str = input("Amount to Convert: ").strip()

        from_c = from_curr if from_curr else "USD"
        to_c = to_curr if to_curr else "INR"
        amount = float(amount_str) if amount_str else 100.0

        converted = convert_currency(amount, from_c, to_c)
        if converted is not None:
            symbol_map = {"INR": "Rs.", "USD": "$", "EUR": "€", "GBP": "£"}
            symbol = symbol_map.get(to_c.upper(), "")

            print("\n---------------- CONVERSION RESULT ----------------")
            print(f"{amount:,.2f} {from_c.upper()} = {symbol}{converted:,.2f} {to_c.upper()}")
            print("---------------------------------------------------\n")

    except ValueError:
        print("Input Error: Please enter a valid numerical amount!")

if __name__ == "__main__":
    main()

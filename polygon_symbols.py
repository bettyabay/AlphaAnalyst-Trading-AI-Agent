import requests

API_KEY = "Z4QS_cUMyOkA153ICpw7e8SE7GaRxne1"
ticker = "I:W5000"

url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
params = {"apiKey": API_KEY}

res = requests.get(url, params=params)
print(res.json())

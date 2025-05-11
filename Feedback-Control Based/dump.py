import json
import pandas as pd
with open("teststock_data_2024-01-02_to_2025-01-01.json","r") as f:
    data = json.load(f)

print(data['AAPL']['open'][1:100])
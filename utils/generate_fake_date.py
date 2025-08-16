import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

# Generate fake patient IDs
people_ids = np.random.choice(range(100), size=n)

# Generate fake dates
dates = pd.date_range(start="2024-01-01", periods=100).to_list()
actual_dates = np.random.choice(dates, size=n)

# Generate a TotalIndex score between 0 and 100
total_index = np.clip(np.random.normal(loc=50, scale=15, size=n), 0, 100)

# Binary Incident next day, probabilistically linked to index score
incident_next_day = (np.random.rand(n) < (total_index / 150)).astype(int)

df = pd.DataFrame({
    "people_id": people_ids,
    "actual_date": actual_dates,
    "TotalIndex": total_index,
    "IncidentNextDay": incident_next_day
})

df = df.sort_values(by=["people_id", "actual_date"])
df.to_csv("data/fake_structured.csv", index=False)

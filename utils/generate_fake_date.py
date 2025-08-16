import numpy as np
import pandas as pd

np.random.seed(42)

n_rows = 1000
people_ids = np.arange(1, 101)
dates = pd.date_range("2024-01-01", periods=120, freq="D")

# Column groups
activity_cols = [
    "WakeUp","MedicationCompliance","Hygene","School","TreatmentAndGroup",
    "Activity","DownTime","FreeTime","Therapy","MedAppointments",
    "Visitations","PhoneCalls","Bedtime"
]
emotion_cols = ["Happiness","Sadness","Fear","Anger"]
str_cols = [
    "Politeness","PersonalInsight","Honesty","Kindness","Cooperativeness",
    "Participation","TreatmentEngagement","AskingNeeds","Responsibility",
    "WaitingTurns","MakingTransitions","CopingSkills","StaffConnection",
    "Redirectable","Respect","Compliance","PeerRelationships"
]
special_cols = ["Boundaries","Predictability"]

# Base frame
df = pd.DataFrame({
    "people_id": np.random.choice(people_ids, size=n_rows),
    "Age": np.random.randint(12, 19, size=n_rows),
    "actual_date": np.random.choice(dates, size=n_rows),
    "ShiftNumber": np.random.choice([1,2], size=n_rows)
})

# Activities: 0–4, skewed toward 0
for col in activity_cols:
    df[col] = np.random.choice(range(5), size=n_rows, p=[0.5,0.25,0.15,0.07,0.03])

# Emotions
df["Sadness"] = np.random.choice(range(7), size=n_rows, p=[0.3,0.25,0.2,0.1,0.08,0.05,0.02])
df["Fear"]    = np.random.choice(range(7), size=n_rows, p=[0.35,0.25,0.2,0.1,0.05,0.04,0.01])
df["Anger"]   = np.random.choice(range(7), size=n_rows, p=[0.4,0.25,0.15,0.1,0.05,0.04,0.01])
df["Happiness"] = -np.random.choice(range(7), size=n_rows, p=[0.4,0.25,0.15,0.1,0.05,0.04,0.01])

# Strengths: 0–4, skewed toward higher values
for col in str_cols:
    df[col] = np.random.choice(range(5), size=n_rows, p=[0.05,0.1,0.2,0.3,0.35])

# Special
df["Boundaries"]    = np.random.choice(range(5), size=n_rows, p=[0.4,0.25,0.2,0.1,0.05])
df["Predictability"] = np.random.choice(range(6), size=n_rows, p=[0.35,0.25,0.2,0.1,0.07,0.03])

# TotalIndex = sum of all *difficulty* style items
difficulty_vars = activity_cols + ["Sadness","Fear","Anger","Boundaries","Predictability"]
# (Happiness is negative already, so add its absolute value)
df["TotalIndex"] = df[difficulty_vars].sum(axis=1) + df["Happiness"].abs()

# Incident next day probability ~ logistic on TotalIndex
prob = 1 / (1 + np.exp(-(df["TotalIndex"] - 10)/10))
df["IncidentNextDay"] = (np.random.rand(n_rows) < prob).astype(int)

df = df.sort_values(by=["people_id","actual_date","ShiftNumber"]).reset_index(drop=True)

df.to_csv("fake_structured.csv", index=False)

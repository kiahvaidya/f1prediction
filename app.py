from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import fastf1
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

with open("formula1_model.pkl", "rb") as f: model = pickle.load(f)

driver_enc   = LabelEncoder()
team_enc     = LabelEncoder()
compound_enc = LabelEncoder()
event_enc    = LabelEncoder()

train_df = pd.read_csv("formula1.csv")
driver_enc.fit(train_df["Driver"])
team_enc.fit(train_df["Team"])
compound_enc.fit(train_df["Compound"])
event_enc.fit(train_df["EventName"])

fastf1.Cache.enable_cache("cache_dir")  

def fetch_and_preprocess(gp_name, year):
    
    try:
        session = fastf1.get_session(int(year), gp_name, 'Q')
        session.load()
        res = session.results.copy()
        laps = session.laps

        laps = laps.dropna(subset=["Driver", "LapTime"])

        idx_fast = laps.groupby("Driver")["LapTime"].idxmin()

        fast_laps = laps.loc[idx_fast, ["Driver","LapTime","Compound","Stint"]].copy()
        fast_laps["QualifyingTime"] = fast_laps["LapTime"].dt.total_seconds()

        quali = pd.DataFrame({"Driver": res["Abbreviation"],"Team": res["TeamName"]}).merge(
            fast_laps[["Driver","QualifyingTime","Compound","Stint"]], on="Driver", how="left"
        )
        
        # Replace NaN or inf with a large number or average
        quali["QualifyingTime"] = quali["QualifyingTime"].fillna(quali["QualifyingTime"].mean())
        quali["QualifyingTime"] = quali["QualifyingTime"].replace([np.inf, -np.inf], quali["QualifyingTime"].mean())


        
        quali["StartPosition"] = quali["QualifyingTime"].rank(method="first").astype(int)

        quali["EventName"] = gp_name
        quali["EventEncoded"] = event_enc.transform(quali["EventName"])

        quali["DriverEncoded"]   = driver_enc.transform(quali["Driver"])
        quali["TeamEncoded"]     = team_enc.transform(quali["Team"])
        quali["CompoundEncoded"] = compound_enc.transform(quali["Compound"])
       
        quali["Year"] = int(year)


        features_for_model = quali[["Stint","Year","QualifyingTime","StartPosition", "DriverEncoded","CompoundEncoded","TeamEncoded","EventEncoded"]]
        return quali, features_for_model

    except Exception as e:
        print("Error fetching session:", e)
        return None, None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    gp_name = ""
    year = ""

    if request.method == "POST":
        gp_name = request.form.get("gp_name")
        year = request.form.get("year")


        df_event, X = fetch_and_preprocess(gp_name, year)
        if X is not None and not X.empty:
            preds = model.predict(X)
            df_event['PredictedRank'] = preds
            df_event['PredictedRank'] = df_event['PredictedRank'].rank(method='first').astype(int)
            podium = df_event.sort_values("PredictedRank").head(3)
            prediction = podium[["Driver","Team","PredictedRank"]].to_dict(orient='records')


    return render_template("index.html", prediction=prediction, gp_name=gp_name, year=year)

if __name__ == "__main__":
    app.run(debug=True)

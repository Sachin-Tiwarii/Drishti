"""Optional evaluation script for the synthetic demo dataset."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

df=pd.read_csv("dataset/grouped_week.csv")
df["sin_week"]=np.sin(2*np.pi*df.iso_week/52); df["cos_week"]=np.cos(2*np.pi*df.iso_week/52)
features=["AREA","LAT","LON","iso_week","sin_week","cos_week"]
tr=df[df.iso_year<=2024]; te=df[df.iso_year==2025]
models={"XGBoost":XGBRegressor(n_estimators=140,max_depth=4,learning_rate=.05,objective="reg:squarederror",random_state=42),
"Random Forest":RandomForestRegressor(n_estimators=140,max_depth=10,min_samples_leaf=2,random_state=42),
"KNN":Pipeline([("scaler",StandardScaler()),("knn",KNeighborsRegressor(n_neighbors=7,weights="distance"))])}
for name,m in models.items():
 m.fit(tr[features],tr.crime_count); p=m.predict(te[features])
 print(f"{name}: MAE={mean_absolute_error(te.crime_count,p):.2f}, RMSE={np.sqrt(mean_squared_error(te.crime_count,p)):.2f}, R2={r2_score(te.crime_count,p):.3f}")

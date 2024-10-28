import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import xgboost as xgb

# load the dataset (csv file)

df = pd.read_csv("G:\\Education\\BigData Project\\ODI_Match_Data_New.csv", low_memory=False)

# Computing Total runs ball by ball

df['innings_runs'] = df.groupby(['match_id', 'innings'])['runs_off_bat'].cumsum() + df.groupby(['match_id', 'innings'])['extras'].cumsum()

# Computing total wickets ball by ball

tw = df.iloc[:,[0,4,18]]
tw.loc[:,'wicket_type'] = tw['wicket_type'].fillna(0)

wicket_cnt=[]
cnt=0
tw['wicket_type'] = tw['wicket_type'].fillna(0)
w=1
for i in range(len(tw)):
    if tw['innings'][i]!=w:
        w=2
        cnt=0
    if tw['wicket_type'][i]!=0:
        cnt+=1
    wicket_cnt.append(cnt)
df['Innings_wickets'] = wicket_cnt

# Computing Total Score Per Ennings

df['total_score'] = df.groupby(['match_id', 'innings'])['innings_runs'].transform('last').astype(int)

for rIndex, rRow in df.iterrows():
    if df['ball'][rIndex] > 5.0:
        print(df['ball'][rIndex])


# computinf Runs in Last 5 Overs

for rIndex, rRow in df.iterrows():
    if df['ball'][rIndex] > 5.0:
        d = df['innings_runs'][rIndex] - df['innings_runs'][rIndex-30]
        df.at[rIndex,'runs_last_5_overs'] = d
    else:
        df.at[rIndex,'runs_last_5_overs'] = df['innings_runs'][rIndex]


# computinf Wickets in Last 5 Overs

for wIndex, wRow in df.iterrows():
    if df['ball'][wIndex] > 5.0:
        d = df['Innings_wickets'][wIndex] - df['Innings_wickets'][wIndex-30]
        df.at[wIndex,'wickets_last_5_overs'] = d
    else:
        df.at[wIndex,'wickets_last_5_overs'] = df['Innings_wickets'][wIndex]

# removing irrerelevent features

irr = ['match_id', 'season', 'start_date', 'venue', 'innings',
       'striker', 'non_striker', 'bowler',
       'runs_off_bat', 'extras', 'wides', 'noballs', 'byes', 'legbyes',
       'penalty', 'wicket_type', 'player_dismissed', 'other_wicket_type',
       'other_player_dismissed', 'cricsheet_id']

print(f'Before Removing Irrelevant Columns : {df.shape}')

df = df.drop(irr, axis=1)
print(f'After Removing Irrelevant Columns : {df.shape}')


# Removing Non-Consistent Teams

my_teams = ['Afghanistan', 'Australia', 'Bangladesh','England', 'India', 'Ireland','New Zealand', 'Pakistan', 'South Africa', 'Sri Lanka',
               'West Indies', 'Zimbabwe']

print(f'Before Removing Inconsistent Teams : {df.shape}')

df = df[(df['batting_team'].isin(my_teams)) & (df['bowling_team'].isin(my_teams))]
print(f'After Removing Irrelevant Columns : {df.shape}')

print(f"Consistent Teams : \n{df['batting_team'].unique()}")

# Remove First 5 Overs of every match

print(f'Before Removing Overs : {df.shape}')

df = df[df['ball'] >= 5.0]
print(f'After Removing Overs : {df.shape}')

# Save the cleaned data as CSV
df.to_csv(f"ODI_train_data_new.csv", index=False)

# Load the cleaned Data.
df2 = pd.read_csv('ODI_train_data_new.csv', low_memory=False)

from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
le = LabelEncoder()

# Encode specified columns
df2[['batting_team', 'bowling_team']] = df2[['batting_team', 'bowling_team']].apply(le.fit_transform)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define the ColumnTransformer with OneHotEncoder for the specified columns
column_transformer = ColumnTransformer(
    [('encoder', OneHotEncoder(), ['batting_team', 'bowling_team'])],
    remainder='passthrough'
)

# Apply the transformer to the DataFrame and convert the result to an array
df2_array = column_transformer.fit_transform(df2).toarray()

# Print the shape of the resulting array
print(df2_array.shape)

# Get the feature names after transformation
encoder_feature_names = list(
    column_transformer.named_transformers_['encoder'].get_feature_names_out(input_features=['batting_team', 'bowling_team'])
)

# Combine the encoder feature names with the remaining feature names
feature_names = encoder_feature_names + ['ball', 'innings_runs', 'Innings_wickets', 'total_score', 'runs_last_5_overs', 'wickets_last_5_overs']

# Print the combined feature names
print(feature_names)

df = pd.DataFrame(df2_array, columns=feature_names)

features = df.drop(columns='total_score')
labels = df['total_score']

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.30, shuffle=True)
print(f"Training Set : {train_features.shape}\nTesting Set : {test_features.shape}")

models = dict()

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
# Train Model
tree.fit(train_features, train_labels)

# Evaluate Model
train_score_tree = str(tree.score(train_features, train_labels) * 100)
test_score_tree = str(tree.score(test_features, test_labels) * 100)
print(f'Train Score : {train_score_tree[:5]}%\nTest Score : {test_score_tree[:5]}%')
models["tree"] = test_score_tree

from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
print("---- Decision Tree Regressor - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mae(test_labels, tree.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_labels, tree.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, tree.predict(test_features)))))


xgboost_model = xgb.XGBRegressor()
xgboost_model.fit(train_features, train_labels)

train_score_xgb = str(xgboost_model.score(train_features, train_labels)*100)
test_score_xgb = str(xgboost_model.score(test_features, test_labels)*100)
print(f'Train Score : {train_score_xgb[:5]}%\nTest Score : {test_score_xgb[:5]}%')
models["xgboost_model"] = test_score_xgb

print("---- xgboost_model - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mae(test_labels, xgboost_model.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_labels, xgboost_model.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, xgboost_model.predict(test_features)))))


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
# Train Model
forest.fit(train_features, train_labels)

train_score_forest = str(forest.score(train_features, train_labels) * 100)
test_score_forest = str(forest.score(test_features, test_labels) * 100)
print(f'Train Score : {train_score_forest[:5]}%\nTest Score : {test_score_tree[:5]}%')
models["tree"] = test_score_forest


print("---- Random Forest Regression - Model Evaluation ----")
print("Mean Absolute Error (MAE): {}".format(mae(test_labels, forest.predict(test_features))))
print("Mean Squared Error (MSE): {}".format(mse(test_labels, forest.predict(test_features))))
print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, forest.predict(test_features)))))


def score_predict(batting_team, bowling_team, runs, wickets, overs, runs_last_5_overs, wickets_last_5_overs, model=forest):
    prediction_array = []
  # Batting Team
    if batting_team == 'Afghanistan':
        prediction_array = prediction_array + [1,0,0,0,0,0,0,0,0,0,0,0]
    elif batting_team == 'Australia':
        prediction_array = prediction_array + [0,1,0,0,0,0,0,0,0,0,0,0]
    elif batting_team == 'Bangladesh':
        prediction_array = prediction_array + [0,0,1,0,0,0,0,0,0,0,0,0]
    elif batting_team == 'England':
        prediction_array = prediction_array + [0,0,0,1,0,0,0,0,0,0,0,0]
    elif batting_team == 'India':
        prediction_array = prediction_array + [0,0,0,0,1,0,0,0,0,0,0,0]
    elif batting_team == 'Ireland':
        prediction_array = prediction_array + [0,0,0,0,0,1,0,0,0,0,0,0]
    elif batting_team == 'New Zealand':
        prediction_array = prediction_array + [0,0,0,0,0,0,1,0,0,0,0,0]
    elif batting_team == 'Pakistan':
        prediction_array = prediction_array + [0,0,0,0,0,0,0,1,0,0,0,0]
    elif batting_team == 'South Africa':
        prediction_array = prediction_array + [0,0,0,0,0,0,0,0,1,0,0,0]
    elif batting_team == 'Sri Lanka':
        prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,1,0,0]
    elif batting_team == 'West Indies':
        prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,1,0]
    elif batting_team == 'Zimbabwe':
        prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,1]
  # Bowling Team
    if bowling_team == 'Afghanistan':
        prediction_array = prediction_array + [1,0,0,0,0,0,0,0,0,0,0,0]
    elif bowling_team == 'Australia':
        prediction_array = prediction_array + [0,1,0,0,0,0,0,0,0,0,0,0]
    elif bowling_team == 'Bangladesh':
        prediction_array = prediction_array + [0,0,1,0,0,0,0,0,0,0,0,0]
    elif bowling_team == 'England':
        prediction_array = prediction_array + [0,0,0,1,0,0,0,0,0,0,0,0]
    elif bowling_team == 'India':
        prediction_array = prediction_array + [0,0,0,0,1,0,0,0,0,0,0,0]
    elif bowling_team == 'Ireland':
        prediction_array = prediction_array + [0,0,0,0,0,1,0,0,0,0,0,0]
    elif bowling_team == 'New Zealand':
        prediction_array = prediction_array + [0,0,0,0,0,0,1,0,0,0,0,0]
    elif bowling_team == 'Pakistan':
        prediction_array = prediction_array + [0,0,0,0,0,0,0,1,0,0,0,0]
    elif bowling_team == 'South Africa':
        prediction_array = prediction_array + [0,0,0,0,0,0,0,0,1,0,0,0]
    elif bowling_team == 'Sri Lanka':
        prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,1,0,0]
    elif bowling_team == 'West Indies':
        prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,1,0]
    elif bowling_team == 'Zimbabwe':
        prediction_array = prediction_array + [0,0,0,0,0,0,0,0,0,0,0,1]

    prediction_array = prediction_array + [overs, runs, wickets, runs_last_5_overs, wickets_last_5_overs]
    prediction_array = np.array([prediction_array])
    pred = model.predict(prediction_array)
    return int(round(pred[0]))



def score_predict(batting_team, bowling_team, runs, wickets, overs, runs_last_5_overs, wickets_last_5_overs):
    # Dummy logic for score prediction (Replace with your model's logic)
    predicted_score = runs + (50 - wickets * 5) + (5 - wickets_last_5_overs) * 10
    return predicted_score

# Title of the Web App
st.title("Cricket Score Prediction")

# Dropdowns for Batting and Bowling Teams
teams = ['Australia', 'Bangladesh', 'England', 'India', 'New Zealand', 
         'Pakistan', 'South Africa', 'Sri Lanka', 'West Indies', 'Zimbabwe']

batting_team = st.selectbox("Select Batting Team", teams)
bowling_team = st.selectbox("Select Bowling Team", teams)

# Input fields for other match details
overs = st.number_input("Overs Completed", min_value=0.0, max_value=50.0, step=0.1)
runs = st.number_input("Current Runs", min_value=0)
wickets = st.number_input("Wickets Lost", min_value=0, max_value=10)
runs_last_5_overs = st.number_input("Runs in Last 5 Overs", min_value=0)
wickets_last_5_overs = st.number_input("Wickets in Last 5 Overs", min_value=0, max_value=5)

# Button to trigger prediction
if st.button("Predict Score"):
    score = score_predict(batting_team, bowling_team, runs, wickets, overs, 
                          runs_last_5_overs, wickets_last_5_overs)
    st.success(f"Predicted Score: {score}")


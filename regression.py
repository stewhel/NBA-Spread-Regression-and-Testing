import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge

pd.options.mode.chained_assignment = None

# Load datasets from https://www.basketball-reference.com/leagues/NBA_2022_games-october.html

df1 = pd.read_csv('October2021.csv')
df2 = pd.read_csv('November2022.csv')
df = pd.concat([df1, df2])

# Rename the columns

df.rename(columns={ 
    df.columns[0]: "date",
    df.columns[2]: "vis",
    df.columns[3]: "vis_pts",
    df.columns[4]: "home",
    df.columns[5]: "home_pts"
                  }, inplace = True)

df = df[['date', 'vis', 'vis_pts', 'home', 'home_pts']]

# Add in a "Home Point difference" so we can get columns for home win or home loss

df['home_pt_diff'] = df['home_pts'] - df['vis_pts']
df['home_win'] = np.where(df['home_pt_diff'] > 0, 1, 0)
df['home_loss'] = np.where(df['home_pt_diff'] < 0, 1, 0)

# Create dummy values to convert a categorial variable, the team, into an indicator variable

df_home = pd.get_dummies(df['home'], dtype=np.int64)
df_vis = pd.get_dummies(df['vis'], dtype= np.int64)

# Create training model

df_model = df_home.sub(df_vis)
df_model['home_pt_diff'] = df['home_pt_diff']
df_train = df_model

# Conduct ridge regression. Ridge regression helps avoid overfitting the data to the training set.
# A least squares regression's slope is min(sum of the squared residuals). Ridge is min(sum of squared resideuals + alpha + slope square)

lr = Ridge(alpha = 0.001)
x = df_train.drop(['home_pt_diff'], axis = 1)
y = df_train['home_pt_diff']
             
lr.fit(x, y)

# The rating is the regression coefficient, or how many points above average the team will expect to score

df_ratings = pd.DataFrame(data={'team': x.columns, 'rating': lr.coef_})
df_ratings.sort_values('rating')

# Make a dictionary for each teams rating

df_ratings = df_ratings.set_index('team')
key = df_ratings.to_dict()
key = key['rating']

# Odds Data --- https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm
# Load the odds dataset. Data for each game took up two rows. I made new columns and used =MOD(ROW(),2) to get 0s and 1s,then filtered for 1s to remove every other row

df_odds = pd.read_csv('OddsData22_clean_post_november.csv')
df_odds = df_odds[['home', 'vis', 'home_tot', 'vis_tot', 'home_ml', 'vis_ml']]

# Create a dictionary matching the team names

names = {
    'Brooklyn': 'Brooklyn Nets',
    'Milwaukee': 'Milwaukee Bucks',
    'GoldenState': 'Golden State Warriors',
    'LALakers': 'Los Angeles Lakers',
    'Indiana': 'Indiana Pacers',
    'Charlotte': 'Charlotte Hornets',
    'Chicago': 'Chicago Bulls',
    'Detroit': 'Detroit Pistons',
    'Washington': 'Washington Wizards',
    'Toronto': 'Toronto Raptors',
    'Boston': 'Boston Celtics',
    'NewYork': 'New York Knicks',
    'Cleveland': 'Cleveland Cavaliers',
    'Memphis': 'Mephis Grizzlies',
    'Philadelphia': 'Philadelphia 76ers',
    'NewOrleans': 'New Orleans Pelicans',
    'Houston': 'Houston Rockets',
    'Minnesota': 'Minnesota Timberwolves',
    'Orlando': 'Orlando Magic',
    'SanAntonio': 'San Antonio Spurs',
    'OklahomaCity': 'Oklahoma City Thunder',
    'Utah': 'Utah Jazz',
    'Sacramento': 'Sacramento Kings',
    'Portland': 'Portland Trailblazers',
    'Denver': 'Denver Nuggets',
    'Phoenix': 'Pheonix Suns',
    'Dallas': 'Dallas Mavericks',
    'Atlanta': 'Atlanta Hawks',
    'Miami': 'Miami Heat',
    'LAClippers': 'Los Angeles Clippers',
    'Golden State': 'Golden State Warriors'
}

df_odds['home'] = df_odds['home'].map(names)
df_odds['vis'] = df_odds['vis'].map(names)

# Now that the names match, map the ratings

df_odds['home_rat'] = df_odds['home'].map(key)
df_odds['vis_rat'] = df_odds['vis'].map(key)

# Convert betting lines to integer, and then convert from American to decimal odds

df_odds['home_ml'] = df_odds['home_ml'].astype(int)
df_odds['vis_ml'] = df_odds['vis_ml'].astype(int)
df_odds['vis_ml'] = np.where(df_odds['vis_ml'] > 0, (df_odds['vis_ml'] / 100) + 1,  (100 / abs(df_odds['vis_ml']) + 1))
df_odds['home_ml'] = np.where(df_odds['home_ml'] > 0, (df_odds['home_ml'] / 100) + 1,  (100 / abs(df_odds['home_ml']) + 1))

# Bet based on who has a higher rating. Win 1212 bets, lose 1434

df_odds['bet_home']  = np.where(df_odds['home_rat'] > df_odds['vis_rat'] , 1, 0)
df_odds['home_win'] = np.where(df_odds['home_tot'] > df_odds['vis_tot'] , 1, 0)

df_odds['bet_win'] = np.where(df_odds['bet_home'] == df_odds['home_win'], 1, 0)
df_odds['bet_lose'] = np.where(df_odds['bet_home'] != df_odds['home_win'] , 1, 0)

df_odds['bet_odds'] = np.where(df_odds['bet_home'] == 1 , df_odds['home_ml'], df_odds['vis_ml'])
df_odds['winnings'] = np.where(df_odds['bet_win'] == 1, (df_odds['bet_odds'] * 100) - 100, -100)

print("Bets won: " + df_odds['bet_win'].sum())
print("Bets lost: " + df_odds['bet_lose'].sum())
print("Winnings: " + df_odds['winnings'].sum())


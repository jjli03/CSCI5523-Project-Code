import pandas as pd
import matplotlib.pyplot as plt

# Load NBA rookies dataset
nba_rookies = pd.read_csv("NBA_Rookies_InCollegeData.csv")

# Load collegiate players dataset
collegiate_players = pd.read_csv("NBA_makers_LastYearOnly_CollegeBasketballData.csv")

# Calculate correlation between points columns
correlation = nba_rookies['PTSpg'].corr(collegiate_players['pts'])

print("Correlation between NBA rookie points and collegiate player points:", correlation)

# Drop rows with missing values in 'PTSpg' and 'pts' columns
nba_rookies_clean = nba_rookies.dropna(subset=['PTSpg'])
collegiate_players_clean = collegiate_players.dropna(subset=['pts'])

# Ensure both arrays have the same length
min_length = min(len(nba_rookies_clean), len(collegiate_players_clean))
nba_rookies_clean = nba_rookies_clean[:min_length]
collegiate_players_clean = collegiate_players_clean[:min_length]

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(nba_rookies_clean['PTSpg'], collegiate_players_clean['pts'], alpha=0.5)
plt.title("NBA Rookie Points vs Collegiate Player Points")
plt.xlabel("NBA Rookie Points per Game (PTSpg)")
plt.ylabel("Collegiate Player Points")
plt.grid(True)
plt.show()

# ---------- Assists ----------

# Calculate correlation between points columns
correlation = nba_rookies['ASTpg'].corr(collegiate_players['ast'])

print("Correlation between NBA rookie points and collegiate player assists:", correlation)

# Drop rows with missing values in 'PTSpg' and 'pts' columns
nba_rookies_clean = nba_rookies.dropna(subset=['ASTpg'])
collegiate_players_clean = collegiate_players.dropna(subset=['ast'])

# Ensure both arrays have the same length
min_length = min(len(nba_rookies_clean), len(collegiate_players_clean))
nba_rookies_clean = nba_rookies_clean[:min_length]
collegiate_players_clean = collegiate_players_clean[:min_length]

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(nba_rookies_clean['ASTpg'], collegiate_players_clean['ast'], alpha=0.5)
plt.title("NBA Rookie assists vs Collegiate Player Assists")
plt.xlabel("NBA Rookie assists per Game (ASTpg)")
plt.ylabel("Collegiate Player assists")
plt.grid(True)
plt.show()

# ---------- Rebounds ----------

correlation = nba_rookies['TRBpg'].corr(collegiate_players['treb'])

print("Correlation between NBA rookie points and collegiate player rebounds:", correlation)

# Drop rows with missing values in 'PTSpg' and 'pts' columns
nba_rookies_clean = nba_rookies.dropna(subset=['TRBpg'])
collegiate_players_clean = collegiate_players.dropna(subset=['treb'])

# Ensure both arrays have the same length
min_length = min(len(nba_rookies_clean), len(collegiate_players_clean))
nba_rookies_clean = nba_rookies_clean[:min_length]
collegiate_players_clean = collegiate_players_clean[:min_length]

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(nba_rookies_clean['TRBpg'], collegiate_players_clean['treb'], alpha=0.5)
plt.title("NBA Rookie rebounds vs Collegiate Player Rebounds")
plt.xlabel("NBA Rookie rebounds per Game (TRBpg)")
plt.ylabel("Collegiate Player rebounds")
plt.grid(True)
plt.show()
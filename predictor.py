from pathlib import Path
import math
import pandas as pd
import numpy as np
from scipy.stats import poisson

DATA_FILE = Path(__file__).parent / "data" / "matches.csv"
MAX_GOALS = 6
RECENT_MATCHES = 5
HISTORICAL_SEASONS = 2


def load_matches(path=DATA_FILE):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Season"])
    return df.sort_values("Date").reset_index(drop=True)


def split_seasons(matches):
    seasons = sorted(matches["Season"].dropna().unique())
    if len(seasons) < HISTORICAL_SEASONS + 1: 
        raise ValueError("Dataset needs two completed seasons plus the current season.")
    current = seasons[-1]
    historical = seasons[-(HISTORICAL_SEASONS + 1):-1]
    return historical, current


def recent_form(matches, team, before_date=None):
    _, current_season = split_seasons(matches)
    data = matches[matches["Season"] == current_season].copy()
    if before_date is not None:
        before_date = pd.Timestamp(before_date)
        if before_date.tzinfo is None:
            before_date = before_date.tz_localize("UTC")
        data = data[data["Date"] < before_date]

    team_matches = data[(data["HomeTeam"] == team) | (data["AwayTeam"] == team)].sort_values("Date").tail(RECENT_MATCHES)
    gf, ga, points = [], [], 0

    for _, row in team_matches.iterrows():
        if row["HomeTeam"] == team:
            goals_for, goals_against = row["FTHG"], row["FTAG"]
        else:
            goals_for, goals_against = row["FTAG"], row["FTHG"]
        gf.append(goals_for)
        ga.append(goals_against)
        points += 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0

    n = len(team_matches)
    return {
        "matches": n,
        "goals_for": np.mean(gf) if gf else 0.0,
        "goals_against": np.mean(ga) if ga else 0.0,
        "points_per_game": points / n if n else 0.0,
    }


def team_season_stats(matches, team, home=None):
    if home is True:
        data = matches[matches["HomeTeam"] == team]
        gf, ga = data["FTHG"], data["FTAG"]
    elif home is False:
        data = matches[matches["AwayTeam"] == team]
        gf, ga = data["FTAG"], data["FTHG"]
    else:
        home_data = matches[matches["HomeTeam"] == team]
        away_data = matches[matches["AwayTeam"] == team]
        gf = pd.concat([home_data["FTHG"], away_data["FTAG"]])
        ga = pd.concat([home_data["FTAG"], away_data["FTHG"]])

    return {
        "matches": len(gf),
        "goals_for": gf.mean() if len(gf) else 0.0,
        "goals_against": ga.mean() if len(ga) else 0.0,
    }


def expected_goals(matches, home_team, away_team, before_date=None):
    historical_seasons, current_season = split_seasons(matches)
    historical = matches[matches["Season"].isin(historical_seasons)].copy()

    # For a current-season prediction, use only completed current-season matches before the fixture.
    current = matches[matches["Season"] == current_season].copy()
    if before_date is not None:
        before_date = pd.Timestamp(before_date)
        if before_date.tzinfo is None:
            before_date = before_date.tz_localize("UTC")
        current = current[current["Date"] < before_date]

    home_stats = team_season_stats(historical, home_team, home=True)
    away_stats = team_season_stats(historical, away_team, home=False)
    home_recent = recent_form(matches, home_team, before_date)
    away_recent = recent_form(matches, away_team, before_date)

    home_attack = 0.70 * home_stats["goals_for"] + 0.30 * home_recent["goals_for"]
    away_defence = 0.70 * away_stats["goals_against"] + 0.30 * away_recent["goals_against"]
    away_attack = 0.70 * away_stats["goals_for"] + 0.30 * away_recent["goals_for"]
    home_defence = 0.70 * home_stats["goals_against"] + 0.30 * home_recent["goals_against"]

    league_home_goals = historical["FTHG"].mean() if len(historical) else 1.4
    league_away_goals = historical["FTAG"].mean() if len(historical) else 1.1

    home_attack_strength = home_attack / league_home_goals if league_home_goals else 1
    away_defensive_weakness = away_defence / league_away_goals if league_away_goals else 1
    away_attack_strength = away_attack / league_away_goals if league_away_goals else 1
    home_defensive_weakness = home_defence / league_home_goals if league_home_goals else 1

    home_xg = league_home_goals * home_attack_strength * away_defensive_weakness
    away_xg = league_away_goals * away_attack_strength * home_defensive_weakness
    return float(np.clip(home_xg, 0.05, 4.5)), float(np.clip(away_xg, 0.05, 4.5))


def predict_match(matches, home_team, away_team, before_date=None):
    home_xg, away_xg = expected_goals(matches, home_team, away_team, before_date)
    score_probs = {}
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            score_probs[(h, a)] = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)

    home_win = sum(p for (h, a), p in score_probs.items() if h > a)
    draw = sum(p for (h, a), p in score_probs.items() if h == a)
    away_win = sum(p for (h, a), p in score_probs.items() if h < a)
    total = home_win + draw + away_win

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_xg": home_xg, 
        "away_xg": away_xg,
        "home_win": home_win / total,
        "draw": draw / total,
        "away_win": away_win / total,
        "most_likely_score": max(score_probs, key=score_probs.get),
    }


def backtest(matches):
    _, current_season = split_seasons(matches)
    current_matches = matches[matches["Season"] == current_season].sort_values("Date")
    correct = total = 0

    for _, row in current_matches.iterrows():
        history = current_matches[current_matches["Date"] < row["Date"]]
        if len(history) < 50:
            continue
        result = predict_match(matches, row["HomeTeam"], row["AwayTeam"], row["Date"])
        predicted = "H" if result["home_win"] >= max(result["draw"], result["away_win"]) else "D" if result["draw"] >= result["away_win"] else "A"
        correct += predicted == row["FTR"]
        total += 1

    return correct / total if total else 0.0, total


def print_prediction(result):
    print("\\n" + "=" * 55)
    print(f"{result['home_team']} vs {result['away_team']}")
    print("=" * 55)
    print(f"Expected goals: {result['home_team']} {result['home_xg']:.2f} - {result['away_team']} {result['away_xg']:.2f} {result['away_team']}")
    print(f"Home win : {result['home_win']:.1%}")
    print(f"Draw     : {result['draw']:.1%}")
    print(f"Away win : {result['away_win']:.1%}")
    print(f"Most likely score: {result['most_likely_score'][0]}-{result['most_likely_score'][1]}")


if __name__ == "__main__":
    matches = load_matches()
    historical, current = split_seasons(matches)
    print(f"Historical seasons: {historical}")
    print(f"Current season: {current}")
    print(f"Recent form: latest {RECENT_MATCHES} current-season fixtures")

    home_team = input("Enter home team: ")
    away_team = input("Enter away team: ")

    home_team = home_team.strip()
    away_team = away_team.strip()

    team_names = matches["HomeTeam"].unique()

    home_team = next((t for t in team_names if t.lower() == home_team.lower()), home_team)
    away_team = next((t for t in team_names if t.lower() == away_team.lower()), away_team)

    result = predict_match(matches, home_team, away_team)

    print_prediction(result) 

    # accuracy, n = backtest(matches)
    # print(f"\\nBacktest accuracy: {accuracy:.1%} ({n} matches)")
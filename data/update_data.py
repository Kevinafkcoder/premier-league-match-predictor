from pathlib import Path
import requests
import pandas as pd

DATA_FILE = Path(__file__).parent / "matches.csv"
SEASON_URLS = {
    "2024/2025": "https://fixturedownload.com/feed/json/epl-2024",
    "2025/2026": "https://fixturedownload.com/feed/json/epl-2025",
    "2026/2027": "https://fixturedownload.com/feed/json/epl-2026",
}


def fetch_season(season, url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    rows = response.json()

    out = []
    for row in rows:
        if row.get("HomeTeamScore") is None or row.get("AwayTeamScore") is None:
            continue
        home_goals = int(row["HomeTeamScore"])
        away_goals = int(row["AwayTeamScore"])
        result = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
        out.append({
            "Date": row["DateUtc"],
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "FTHG": home_goals,
            "FTAG": away_goals,
            "FTR": result,
            "Season": season,
        })

    return pd.DataFrame(out)


def main():
    frames = [fetch_season(season, url) for season, url in SEASON_URLS.items()]
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).sort_values("Date")
    df.to_csv(DATA_FILE, index=False)
    print(f"Saved {len(df)} completed matches to {DATA_FILE}")
    print(df.groupby("Season").size())


if __name__ == "__main__":
    main()

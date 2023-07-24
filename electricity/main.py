from datetime import datetime
import pandas as pd
from meteostat import Hourly, Stations
from functools import lru_cache
from typing import Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Phoenix Sky Harbor Airport
STATION_ID = "72278"
TIME_ZONE = "US/Arizona"


@lru_cache
def get_station(latitude: float, longitude: float):
    stations = Stations()
    stations = stations.nearby(latitude, longitude)
    return stations.fetch(1)


def get_weather_data(start: datetime, end: datetime):
    data = Hourly(STATION_ID, start, end)
    return data.fetch().tz_localize("etc/UTC").tz_convert(TIME_ZONE).tz_localize(None)


def str_to_datetime(
    start_date: str, end_date: str, usage_data: pd.DataFrame
) -> Tuple[datetime, datetime]:
    start = min(usage_data["Date_Time"])
    end = max(usage_data["Date_Time"])
    if start_date:
        start = datetime.fromisoformat(start_date)
    if end_date:
        end = datetime.fromisoformat(end_date)
    return start, end


def scatter(df: pd.DataFrame, x: str, y: str, hover: str):
    fig = px.scatter(df, x=x, y=y, hover_data=hover)
    fig.show()


def double_y(df: pd.DataFrame, x: str, y1: str, y2: str):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=df[x], y=df[y1], name=y1),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df[x], y=df[y2], name=y2),
        secondary_y=True,
    )
    fig.show()


def add_wet_bulb_temp(df: pd.DataFrame) -> pd.DataFrame:
    temp = df["temp"]
    rh = df["rhum"]
    df["wet_bulb_temp"] = (
        temp * np.arctan(0.152 * (rh + 8.3136) ** 0.5)
        + np.arctan(temp + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * (rh ** (3 / 2)) * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return df


def main(csv_path: str, start_date: str, end_date: str):
    usage = pd.read_csv(csv_path, parse_dates=[["Date", "Time"]])
    start, end = str_to_datetime(start_date, end_date, usage)
    weather = get_weather_data(start, end)
    combined = pd.merge(usage, weather, left_on="Date_Time", right_on="time")
    combined = add_wet_bulb_temp(combined)
    scatter(combined, "temp", "Usage(kWh)", "Date_Time")
    scatter(combined, "wet_bulb_temp", "Usage(kWh)", "Date_Time")
    double_y(combined, "Date_Time", "Usage(kWh)", "temp")
    double_y(combined, "Date_Time", "Usage(kWh)", "wet_bulb_temp")
    breakpoint()
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str)
    parser.add_argument("--start-date", type=str, default="")
    parser.add_argument("--end-date", type=str, default="")
    args = parser.parse_args()

    main(**vars(args))

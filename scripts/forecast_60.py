#!/usr/bin/env python3
"""Daily predictions for the next N days from 'today'; optional validation on final 60 days."""

from waimea_forecast.cli import main_forecast_60

if __name__ == "__main__":
    main_forecast_60()

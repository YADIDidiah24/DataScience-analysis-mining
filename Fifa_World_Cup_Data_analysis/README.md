# Football World Cup Analysis - 2022 Prediction

This project analyzes historical FIFA World Cup data to predict the outcome of matches for the 2022 World Cup.

## Code Structure

The code is written in Python and utilizes several libraries including:

- `pandas`: Data manipulation and analysis
- `beautifulsoup4`: Web scraping
- `requests`: Downloads web content
- `scipy`: Statistical functions
- `pickle`: Saves and loads data objects

The code can be broadly divided into the following sections:

### Data Acquisition

- Scrapes historical World Cup data (1930-2018) and 2022 data from Wikipedia tables.
- Saves the scraped data to CSV files.

### Data Cleaning

- Removes irrelevant data (e.g., walk-over matches).
- Cleans the score data (removes extra characters, converts to numerical values).
- Calculates home and away goals from the score.
- Fills missing values and converts data types.
- Calculates total goals scored per match.
- Saves the cleaned data to a CSV file.

### Data Analysis

- Calculates team strength based on historical goal scoring and conceding data.

### Match Prediction

- Defines a function `predict_points` that uses Poisson distribution to predict the probability of each team winning, drawing, or losing based on their team strengths.
- Predicts points for each team in the group stage.

### Group Stage Simulation

- Uses the predicted points to create a standings table for each group.

### Knockout Stage Simulation

- Reads the group winners and runners-up from the group stage tables.
- Populates the knockout stage fixtures with the predicted teams.
- Defines functions `get_winner` and `update_table` to simulate matches and update the knockout stage brackets.
- Simulates the knockout stages by predicting winners based on team strengths.

### Saving Results

- Saves the predicted group stage standings using pickle.

**Note:**

- The code relies on web scraping from Wikipedia, which may be subject to changes in website structure.
- The prediction model is based on historical data and may not be entirely accurate.

I hope this readme provides a clear understanding of the code's functionality!

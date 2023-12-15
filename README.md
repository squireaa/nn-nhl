# nn-nhl
Predicts the outcome of NHL games based on trained neural networks

# Usage
- USAGE RIGHT NOW IS LIMITED
- All models are currently built, but the off-season of the sport is limiting use at the moment
- Next steps:
    - Use p-values to figure out the probability the sample tests reflect the accuracy for the full dataset
    - Test the models accuracy against unseen 2022-23 data once that data is released
    - Impliment a way to gather data real time once the season starts to automatically make predictions
    - Add an injury report to the data, which can help fine tune accuracy, especially when a key player is missing from a lineup
    - Create a flask-based web tool to act as a front-end to display the predictions in real time

# Summary
- This project aimed to investigate discrepancies between sportsbook evaluations and data-backed evaluations as it pertains to sportsbook odds in the NHL
- Data was gathered through various APIs that provided access to numerous types of data that could be used in training a neural network
- Data was then organized into forms usable by the neural network
- Several types of training were done on the dataframes to optimize both performance and accuracy
- Results were then rigorously tested
- Future: models will be used to give real-time predictions for live games

# Results
- The accuracy for the moneyline odds is solidly in the 90% accuracy range, which could have huge implications if that accuracy holds throughout the next season
- The data shows that under-over betting does adhere to the stereotype that under-over bets are a 50/50 gamble
- Spread data was ~95% as accurate as what's thought feasibly possible for neural-networks to predict

# Graphs
![image](https://github.com/squireaa/nn-nhl/blob/main/nnnhlpicture.PNG)

# Pricing Analysis

This project uses a simulation to predict the outcomes of a membership model where on pays a monthly price to unlock perks/discounts with each trip. From analysis of SaMo riders, there is evidence that suggests that memberships will help with retention and reduce churn. While such a program can be beneficial for longer term revenue, the price point needs to be one that does not increase the risk of potential revenue loss based on a market's YoY revenue.


Benefits:
- Customer retention/loyalty
- Ridership
- Sustained revenue


## Probability Distributions

Inputs of each discrete event were modeled after historical data. Goodness of fit of probability distributions were done using P-P/Q-Q plots as well as various scores relevant to each respective distribution.


## Model Iterations and Descriptions

| Model Version | Description |
|----------|----------|
|   V1  |   This is the first simulation that was created. Feedback had outcomes being too bullish so we want to model new rider behavior by those that take on a membership. Per Edwin's feedback, we probably also only want to have one tier.   |
|   V2  |   This model will now only have 1 tier for membership and will allow a higher probability that a member will try to makeup for the monthly fee with number of rides equating the unlock fee. It will also model riders who do not signup for a membership even if they ride enough to justify it. Membership will have only $0 unlock fee   |
|   main.py  |   Is an exact copy of V2   |



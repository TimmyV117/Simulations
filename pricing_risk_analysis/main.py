import numpy as np
from scipy.stats import truncnorm
import pandas as pd
import os
import yaml
import copy
from tqdm import tqdm
import logging
import argparse

# Constants
N_ITERATIONS = int(10e3)
SIM_VERISON = 'v2'
SEED_N = 1
BOOLEAN_ASSIGNMENT_THRESHOLD = 0.51  # if probability dist is over this number, assign True
MARKET_NAME = 'ANON_MARKET1'
ANON_MARKET1_UNLOCK_FEE = 1.0  # USD
ANON_MARKET1_PRICE_PER_MINUTE = 0.39  # USD
SAR_UNLOCK_FEE = 3  # USD
SAR_PRICE_PER_MINUTE = 0.39  # USD
SAR_MINUTES_FREE = 5
ANON_MARKET1_AVG_MAU = 1900  # avg monthly active users
ANON_MARKET1_AVG_MAU_STD = 100  # standard deviation monthly active users
ANON_MARKET2_AVG_MAU = 2000  # avg monthly active users
ANON_MARKET2_AVG_MAU_STD = 100  # standard deviation monthlyactive users
MEDIAN_MONTHLY_RIDES = 3
MIN_RIDES = 1  # Minimum number of rides all customers can take
MAX_RIDES = 4  # Max number of rides all customers can take -- expected that
ANON_MARKET1_MEMBERSHIP_UNLOCK_REDUCTION = -1  # take a dollar off
ANON_MARKET1_MEDIAN_RIDE_DURATION_MINUTES = 8.0
SAR_MEMBERSHIP_UNLOCK_REDUCTION = -3
TIER1 = 1
TIER2 = 2
TIER3 = 3
T1_MONTHLY_PRICE = 6.99  # USD -- original
T2_MONTHLY_PRICE = 9.99  # USD
T3_MONTHLY_PRICE = 14.99  # USD
T1_UNLOCK_BREAKEVEN = int(np.ceil(T1_MONTHLY_PRICE))
T2_UNLOCK_BREAKEVEN = int(np.ceil(T2_MONTHLY_PRICE))
T3_UNLOCK_BREAKEVEN = int(np.ceil(T3_MONTHLY_PRICE))
T1_MEMBERSHIP_PCT_DISCOUNT = 0
T2_MEMBERSHIP_PCT_DISCOUNT = 0.05  # 5% off entire trip
T3_MEMBERSHIP_PCT_DISCOUNT = 0.15  # 15% off entire trip
PCT_OF_BREAKEVEN_RIDES = 0.7  # at least 80% of breakeven rides needs to be taken to be considered late signup for standard riders

parser = argparse.ArgumentParser()
parser.add_argument("--test_price", type=float, help="$XX.XX Test price you want to run the simulation with")
parser.add_argument("--it", type=int, help="Number of iterations to run simulation", default=N_ITERATIONS)
args = parser.parse_args()

T1_MONTHLY_PRICE = args.test_price
T1_UNLOCK_BREAKEVEN = int(np.ceil(T1_MONTHLY_PRICE))

logging.basicConfig(filename=f'{SIM_VERISON}_log.log', encoding='utf-8', level=logging.DEBUG)

# NOTE: This will only have one tier and will be more bullish on people taking more rides

def monthly_active_users(market_name):
    if market_name == 'ANON_MARKET1':
        mau = np.random.normal(ANON_MARKET1_AVG_MAU, ANON_MARKET1_AVG_MAU_STD)
    elif market_name == 'ANON_MARKET2':
        mau = np.random.normal(ANON_MARKET2_AVG_MAU, ANON_MARKET2_AVG_MAU_STD)
    else:
        mau = np.nan
        
    return mau


def new_signups_percentage():
    return np.random.beta(7, 93)

# NOTE: we'll init sim by estimating how many customers will be in play
# then we'll estimate if they'll become a member and what tier

# functions for _uptake will be made to figure out if people will sign up initially at introduction
def pct_existing_customer_uptake():
    return np.random.beta(2, 98)

def pct_new_customer_uptake():
    return np.random.beta(8, 92)

# This is a func that will be applied for existing customers that rode at least within 1 ride of breakeven ride
def existing_rider_signsup_after_enough_rides():
    choice_array = [True, False]
    pr_choice = [0.998, 0.002]
    
    return np.random.choice(choice_array, p=pr_choice)
    # return True
 
# NOTE: we'll multiply the uptake pcts times the list of customers and
# assign state of True or False if they take it
# but now which tier will they choose? Only applies to ones that get assigned a state of True

# NOTE: keep track of new customers as True or False
def probability_tier_1():
    return np.random.beta(4, 6)

def probability_tier_2():
    return np.random.beta(3, 7)

def probability_tier_3():
    return np.random.beta(2, 8)

def tier_chosen():
    # in this version we're only going to have 1 tier
    # NOTE: for membership pricing here, edit the T1 constants
    # pr = [probability_tier_1(), probability_tier_2(), probability_tier_3()]
    return TIER1

def initial_price_for_ride(market, minutes):
    if market=='ANON_MARKET1':
        ride_price = ANON_MARKET1_UNLOCK_FEE + (ANON_MARKET1_PRICE_PER_MINUTE * minutes)
    if market=='SAR':
        ride_price = SAR_UNLOCK_FEE + (SAR_PRICE_PER_MINUTE * minutes)
    
    return ride_price

def membership_price_adjustment(market, tier, minutes):
    adjusted_ride_price = 0.0
    if market=='ANON_MARKET1':
        
        if tier==1:
            adjusted_ride_price = (ANON_MARKET1_PRICE_PER_MINUTE * minutes)
        if tier==2:
            adjusted_ride_price = (ANON_MARKET1_PRICE_PER_MINUTE * minutes)
        if tier==3:
            adjusted_ride_price = (ANON_MARKET1_PRICE_PER_MINUTE * minutes)
            
    if market=='SAR':
        
        if tier==1:
            adjusted_ride_price = (SAR_PRICE_PER_MINUTE * minutes)
        if tier==2:
            adjusted_ride_price = (SAR_PRICE_PER_MINUTE * minutes)
        if tier==3:
            adjusted_ride_price = (SAR_PRICE_PER_MINUTE * minutes)
    
    return adjusted_ride_price
        
# run this only for month 0
# gets the probability that a new signup with a membership will be active next month
def probability_membership_new_signup_active_next_month():
    # Set the parameters
    minimum = .4
    maximum = .68
    median = .6

    # Calculate the standard deviation
    std = (maximum - minimum) / 4.0

    # Calculate the lower and upper bounds for truncation
    a = (minimum - median) / std
    b = (maximum - median) / std

    # Generate the truncated normal distribution
    distribution = truncnorm(a, b, loc=median, scale=std)
    pr = float(distribution.rvs(1))
    
    return pr

# run this only for month 0
# gets the probability that an existing rider with a membership will be active next month
def probabillity_membership_existing_active_next_month():
     # Set the parameters
    minimum = .25
    maximum = .6
    median = .53

    # Calculate the standard deviation
    std = (maximum - minimum) / 4.0

    # Calculate the lower and upper bounds for truncation
    a = (minimum - median) / std
    b = (maximum - median) / std

    # Generate the truncated normal distribution
    distribution = truncnorm(a, b, loc=median, scale=std)
    pr = float(distribution.rvs(1))
    
    return pr

# run this only for month 0
# gets the probability that a standard rider will be active next month
def probability_standard_active_next_month():
    return np.random.normal(0.18, 0.1)

# anything with diminished will be reserved for customers from month 0 to month 2
# run this only for month 2 on month 0 cohort
# def probability_membership_new_signup_active_next_month_0_2():
#     return np.random.normal(0.55, 0.025)

# # run this only for month 2 on month 0 cohort
# def probabillity_membership_existing_active_next_month_0_2():
#     return np.random.normal(.50, .01)

# # run this only for month 2 on month 0 cohort
# def probability_standard_active_next_month_0_2():
#     return np.random.normal(0.2, 0.025)

# run this only for month 1 cohort
# gets the probability that a new signup with a membership will be active next month
def probability_membership_new_signup_active_next_month1():
    # Set the parameters
    minimum = .4
    maximum = .68
    median = .6

    # Calculate the standard deviation
    std = (maximum - minimum) / 4.0

    # Calculate the lower and upper bounds for truncation
    a = (minimum - median) / std
    b = (maximum - median) / std

    # Generate the truncated normal distribution
    distribution = truncnorm(a, b, loc=median, scale=std)
    pr = float(distribution.rvs(1))
    
    return pr


# run this only for month 1 cohort
# gets the probability that an existing member with a membership will be active next month
def probabillity_membership_existing_active_next_month1():
    # Set the parameters
    minimum = .4
    maximum = .6
    median = .53

    # Calculate the standard deviation
    std = (maximum - minimum) / 4.0

    # Calculate the lower and upper bounds for truncation
    a = (minimum - median) / std
    b = (maximum - median) / std

    # Generate the truncated normal distribution
    distribution = truncnorm(a, b, loc=median, scale=std)
    pr = float(distribution.rvs(1))
    
    return pr

# run this only for month 1 cohort
# gets the probability that a standard rider will be active next month
def probability_standard_active_next_month1():
    return np.random.normal(0.18, 0.1)


# run this only for month 2 cohort
# gets the probability that a new signup with a membership will be active next month
def probability_membership_new_signup_active_next_month2():
    # Set the parameters
    minimum = .4
    maximum = .68
    median = .6

    # Calculate the standard deviation
    std = (maximum - minimum) / 4.0

    # Calculate the lower and upper bounds for truncation
    a = (minimum - median) / std
    b = (maximum - median) / std

    # Generate the truncated normal distribution
    distribution = truncnorm(a, b, loc=median, scale=std)
    pr = float(distribution.rvs(1))
    
    return pr


# run this only for month 2 cohort
# gets the probability that an existing member with a membership will be active next month
def probabillity_membership_existing_active_next_month2():
    # Set the parameters
    minimum = .4
    maximum = .6
    median = .53

    # Calculate the standard deviation
    std = (maximum - minimum) / 4.0

    # Calculate the lower and upper bounds for truncation
    a = (minimum - median) / std
    b = (maximum - median) / std

    # Generate the truncated normal distribution
    distribution = truncnorm(a, b, loc=median, scale=std)
    pr = float(distribution.rvs(1))
    
    return pr

# run this only for month 2 cohort
# gets the probability that a standard rider will be active next month
def probability_standard_active_next_month2():
    return np.random.normal(0.18, 0.1)

# Assuming everyone will follow a gamma distribution no matter what case
def apply_ride_price(array_length):
    return np.random.gamma(4.7, 1.2, array_length)

# Actually, instead of a price following gamma, let's do pricing based on minutes
# ANON_MARKET1 and SAR follow about the same distribution
# this has a median of around 8 or 9 minutes and goes up to about 40 minutes, which is 95th percentile of data inside of fct_rides
def apply_ride_duration_minutes_standard(array_length):
    return np.clip(np.random.gamma(3, 3.2, array_length), a_min=2.5, a_max=None)   # median 8-9 minutes

def apply_ride_duration_minutes_membership(array_length):
    return np.clip(np.random.gamma(2.95, 3.1, array_length), a_min=ANON_MARKET1_MEDIAN_RIDE_DURATION_MINUTES * np.random.beta(3.8, 6.2), a_max=None)  # Median 11-12 minutes

# TODO: we may be bullish on utilization when people buy a membership
# If someone buys it, they should use it -- someone that doesn't ride a lot wouldn't buy it -- let's look at high probability tier uptake happens
# the probability that someone will become a member is if their initial ride probability is within a certain percentage of breaking even
# new functions to initialize number of rides for standard and membership riders
def apply_number_of_rides_standard(array_length):
    # initial_rides = 1 + np.arange(5)
    # pr_initial_rides = [.05, .3, .5, .1, .05]  # TODO: rethink this distribution to model those who may take a lot of rides but not signup
    lam = MEDIAN_MONTHLY_RIDES + 0.1  # NOTE: This is the distribution we went with to model low pr for higher than 4 rides
    return np.clip(np.random.poisson(lam, size=array_length), a_min=1, a_max=None)

def apply_number_of_rides_membership(array_length):
    initial_rides = np.random.poisson(T1_UNLOCK_BREAKEVEN, array_length)
    initial_rides = np.clip(initial_rides, a_min=T1_UNLOCK_BREAKEVEN - 1, a_max=None)
    
    return initial_rides

# obsolete in v2 since we are modeling the scenario where we know people will really utilize this
def rider_tries_to_make_up_for_monthly_price():
    # will apply to initial rides if rides < break even unlock fee
    choice_array = [True, False]
    pr_choice = [0.23, 0.77]
    
    return np.random.choice(choice_array, p=pr_choice)


# NOTE: So then how many rides will each customer take?
# We assume monthly median of 3
# use this function to add 0 to 3 additional rides
# think of this as a lighter version of people trying to make up for monthly price
# heavier probability on not taking additional rides
def additional_monthly_rides_from_membership(array_length):
    # additional_rides = np.arange(4)
    # pr_additional_rides = [0.1, 0.4, 0.35, 0.15]
    
    # return np.random.choice(additional_rides, p=pr_additional_rides, size=array_length)
    return np.random.poisson(1.8, size=array_length)
    

def roll_the_dice_month0(market_name):
    _mau0 = int(monthly_active_users(market_name))
    _new_signups_percentage0 = new_signups_percentage()
    _existing_customer_uptake_percentage0 = pct_existing_customer_uptake()
    _new_customer_uptake_percentage0 = pct_new_customer_uptake()
    _probability_membership_new_signup_active_next_month0 = probability_membership_new_signup_active_next_month()
    _probabillity_membership_existing_active_next_month0 = probabillity_membership_existing_active_next_month()
    _probability_standard_active_next_month0 = probability_standard_active_next_month()
    
    return _mau0, _new_signups_percentage0, _existing_customer_uptake_percentage0, _new_customer_uptake_percentage0, _probability_membership_new_signup_active_next_month0, _probabillity_membership_existing_active_next_month0, _probability_standard_active_next_month0


def roll_the_dice_month1(market_name):
    _mau1 = int(monthly_active_users(market_name))
    _new_signups_percentage1 = new_signups_percentage()
    _existing_customer_uptake_percentage1 = pct_existing_customer_uptake()
    _new_customer_uptake_percentage1 = pct_new_customer_uptake()
    _probability_membership_new_signup_active_next_month1 = probability_membership_new_signup_active_next_month()
    _probabillity_membership_existing_active_next_month1 = probabillity_membership_existing_active_next_month()
    _probability_standard_active_next_month1 = probability_standard_active_next_month()
    
    return _mau1, _new_signups_percentage1, _existing_customer_uptake_percentage1, _new_customer_uptake_percentage1, _probability_membership_new_signup_active_next_month1, _probabillity_membership_existing_active_next_month1, _probability_standard_active_next_month1

# TODO: resolve this function and see why it won't let you call the new_signups_percentage with float error
# NOTE: the float error was probably occurring because we were re-defining the function as the var name
# and so the value could not be called bc it was no longer a function but instead what was being spit out by the probaiblity distribution
def roll_the_dice_month2(market_name):
    _mau2 = int(monthly_active_users(market_name))
    _new_signups_percentage2 = new_signups_percentage()
    _existing_customer_uptake_percentage2 = pct_existing_customer_uptake()
    _new_customer_uptake_percentage2 = pct_new_customer_uptake()
    _probability_membership_new_signup_active_next_month2 = probability_membership_new_signup_active_next_month2()
    _probabillity_membership_existing_active_next_month2 = probabillity_membership_existing_active_next_month2()
    _probability_standard_active_next_month2 = probability_standard_active_next_month2()
    
    return _mau2, _new_signups_percentage2, _existing_customer_uptake_percentage2, _new_customer_uptake_percentage2, _probability_membership_new_signup_active_next_month2, _probabillity_membership_existing_active_next_month2, _probability_standard_active_next_month2

iteration_results_summary = list()
for i in tqdm(range(args.it)):
    seed = np.random.seed(i)
    iteration = i
    month = 0
    
    # STEP 1: Roll the dice for month 0
    mau_month0, \
    new_signups_percentage_month0, \
    existing_customer_uptake_percentage_month0, \
    new_customer_uptake_percentage_month0, \
    probability_membership_new_signup_active_next_month0, \
    probabillity_membership_existing_active_next_month0, \
    probability_standard_active_next_month0 = roll_the_dice_month0(MARKET_NAME)
    
    # STEP 2: Create the customer indices
    # assumes that initial MAU is estimated from previous month
    existing_customers_month0 = mau_month0
    new_signups_month0 = int(existing_customers_month0 * new_signups_percentage_month0)
    total_customer_list = np.arange((existing_customers_month0 + new_signups_month0)).tolist()  # NOTE: this needs to be turned into a list


    # STEP 3: Existing customer or new customer sign up
    existing_customer_month_list = copy.copy(total_customer_list)
    existing_customer_month_list[:mau_month0] = [True for i in range(len(existing_customer_month_list[:mau_month0]))]
    existing_customer_month_list[mau_month0:] = [False for i in range(len(existing_customer_month_list[mau_month0:]))]


    # STEP 4: Work out of a dataframe to make this easier
    results_df0 = pd.DataFrame()
    results_df0['iteration'] = np.repeat(iteration, len(total_customer_list))  # TODO: copy over this change to the other months
    results_df0['month'] = np.repeat(month, len(total_customer_list))
    results_df0['customers'] = total_customer_list
    results_df0['is_existing_customer'] = existing_customer_month_list
    
    
    # STEP 5: Assign the membership signup boolean
    results_df0['membership_signed_up'] = False
    existing_customers = results_df0.loc[results_df0['is_existing_customer']==True, 'customers']
    n_existing_customers_chosen = int(len(existing_customers) * existing_customer_uptake_percentage_month0)
    existing_customers_chosen = np.random.choice(existing_customers, 
                                                 size=n_existing_customers_chosen, 
                                                 replace=False)
    
    results_df0.loc[results_df0['customers'].isin(existing_customers_chosen), 'membership_signed_up'] = True
    
    new_customers = results_df0.loc[results_df0['is_existing_customer']==False, 'customers']
    n_new_customers_chosen = int(len(new_customers) * new_customer_uptake_percentage_month0)
    new_customers_chosen = np.random.choice(new_customers,
                                            size=n_new_customers_chosen,
                                            replace=False)
    results_df0.loc[results_df0['customers'].isin(new_customers_chosen), 'membership_signed_up'] = True
    
    # STEP 6: For people who signed up, which tier did they choose?
    # everyone chooses just 1 tier in v2
    results_df0['tier_chosen'] = np.nan  # init tier
    results_df0['membership_revenue'] = 0.0  # init membership revenue
    array_length = results_df0.loc[results_df0['membership_signed_up']==True].shape[0]
    results_df0.loc[results_df0['membership_signed_up']==True, 'tier_chosen'] = [tier_chosen() for i in range(array_length)]
    
    # STEP 7: How many rides did each person take this month for standard and membership?
    results_df0['rides_taken'] = 1  # init rides taken
    # array_length = len(results_df0['rides_taken'])
    # print(f'Array length for rides taken init: {array_length}')
    # how many rides for standard riders?
    # creating another column called rides_taken_control: rides taken in a world where there are no memberships
    array_length = results_df0.shape[0]
    results_df0['rides_taken_control'] = apply_number_of_rides_standard(array_length)
    # those who did not sign up from the jump are assigned the control set of rides
    results_df0.loc[results_df0['membership_signed_up']==False, 'rides_taken'] = results_df0.loc[results_df0['membership_signed_up']==False, 'rides_taken_control'].to_list()
    
    # TODO: create an
    array_length = results_df0.loc[results_df0['membership_signed_up']==True, 'rides_taken'].shape[0]
    results_df0.loc[results_df0['membership_signed_up']==True, 'rides_taken'] = apply_number_of_rides_membership(array_length)
    # print(f'Sample of rides applied array: {results_df0["rides_taken"].head(5)}')
    # STEP 7A: What is initial number of rides if membership riders decide to try and make up for monthly price with unlock fee waived?
    # TODO: This subroutine is probably obsolete
    # for idx, row in results_df0.loc[results_df0['membership_signed_up']==True].iterrows():
    #     # TODO: finish the adjustment routine to account for people wanting to make up for the monthly price
    #     _tier_chosen = row['tier_chosen']
    #     initial_rides = row['rides_taken']
    #     will_make_attempt = rider_tries_to_make_up_for_monthly_price()
        
    #     if _tier_chosen == TIER1:
    #         results_df0.at[idx,'membership_revenue'] = T1_MONTHLY_PRICE
    #     elif _tier_chosen == TIER2:
    #         results_df0.at[idx,'membership_revenue'] = T2_MONTHLY_PRICE
    #     elif _tier_chosen == TIER3:
    #         results_df0.at[idx,'membership_revenue'] = T3_MONTHLY_PRICE
        
    #     if will_make_attempt:
    #         if _tier_chosen == TIER1 and initial_rides < T1_UNLOCK_BREAKEVEN:
    #             results_df0.at[idx, 'rides_taken'] = T1_UNLOCK_BREAKEVEN - np.random.choice([0, 1, 2, 3], p=[.1, .2, .2, .5])
    #         elif _tier_chosen == TIER2 and initial_rides < T2_UNLOCK_BREAKEVEN:
    #             results_df0.at[idx, 'rides_taken'] = T2_UNLOCK_BREAKEVEN - np.random.choice([0, 1, 2, 3, 4], p=[.01, .09, .15, .25, .5])
    #         elif _tier_chosen == TIER3 and initial_rides < T3_UNLOCK_BREAKEVEN:
    #             results_df0.at[idx, 'rides_taken'] = T3_UNLOCK_BREAKEVEN - (int(10 * np.random.beta(6, 4)))
    #         else:
    #             continue
    #     else:
    #         continue
    
    # STEP 8: How many additional rides did a membership customer take this month and switch high riding standard riders to membership
    # TODO: This is probably where might want to signup stragglers
    # results_df0['existing_customer_late_signup'] = False
    membership_signup_threshold = T1_UNLOCK_BREAKEVEN * PCT_OF_BREAKEVEN_RIDES   # 80% of breakeven point
    # print(T1_UNLOCK_BREAKEVEN)
    # print(PCT_OF_BREAKEVEN_RIDES)
    # print(membership_signup_threshold)
    riders_likely_to_signup = results_df0.loc[(results_df0['membership_signed_up']==False) & (results_df0['rides_taken_control'] >= membership_signup_threshold), 'customers'].tolist()
    for idx, row in results_df0.loc[results_df0['customers'].isin(riders_likely_to_signup)].iterrows():
        late_signup = existing_rider_signsup_after_enough_rides()
        
        if late_signup:
            results_df0.at[idx, 'existing_customer_late_signup'] = True
            results_df0.at[idx, 'membership_signed_up'] = True
            results_df0.at[idx, 'tier_chosen'] = tier_chosen()
            results_df0.at[idx, 'membership_revenue'] = T1_MONTHLY_PRICE
    
    # TODO: find the customers that were late signups and then apply membership pricing to their additional rides
    
    results_df0['additional_rides_taken'] = 0
    array_length = results_df0[results_df0['membership_signed_up']==True].shape[0]
    results_df0.loc[results_df0['membership_signed_up']==True, 'additional_rides_taken'] = additional_monthly_rides_from_membership(array_length).tolist()
    # print(f'Sample of Additional rides taken: {results_df0["additional_rides_taken"].head(5)}')
    results_df0['total_rides_taken'] = results_df0['rides_taken'] + results_df0['additional_rides_taken']
    # print(f'Sample of total_rides_taken: {results_df0["total_rides_taken"]}')
    
    # STEP 9: Based on the monthly rides taken, assign the total minutes per trip where probability distribution is called for each trip taken
    # and get the ride revenue based on the state of the customer
    
    results_df0['total_control_ride_minutes'] = 0.0
    results_df0['control_ride_minutes_array'] = 0
    results_df0['total_control_ride_revenue'] = 0.0
    results_df0['control_ride_revenue_array'] = 0
    
    results_df0['total_standard_minutes'] = 0.0
    # results_df0['standard_minutes_array'] = 0.0
    results_df0['total_membership_minutes'] = 0.0
    results_df0['membership_ride_minutes_array'] = 0.0
    
    results_df0['initial_standard_ride_revenue'] = 0.0
    results_df0['initial_membership_ride_revenue'] = 0.0
    results_df0['adjusted_standard_ride_revenue'] = 0.0
    results_df0['adjusted_membership_ride_revenue'] = 0.0
    
    results_df0['initial_standard_ride_revenue_array'] = 0.0
    results_df0['initial_membership_ride_revenue_array'] = 0.0
    results_df0['adjusted_standard_ride_revenue_array'] = 0.0
    results_df0['adjusted_membership_ride_revenue_array'] = 0.0
    
    # results_df0['initial_ride_revenue_array'] = 0
    # results_df0['adjusted_revenue_array'] = 0
    
    results_df0.reset_index(drop=True, inplace=True)
    # print(results_df0.head(5))
    # TODO: Get the results straightened out for both standard riders and membership riders
    
    for idx, row in results_df0.iterrows():
        num_rides = row['total_rides_taken']
        num_control_rides = row['rides_taken_control']
        _tier_chosen = row['tier_chosen']
        _signed_up = row['membership_signed_up']
        _late_signup = row['existing_customer_late_signup']
        
        if _signed_up:
            results_df0.at[idx, 'membership_revenue'] = T1_MONTHLY_PRICE
        
        control_ride_minutes_array = apply_ride_duration_minutes_standard(num_control_rides)
        control_ride_revenue_array = [initial_price_for_ride(MARKET_NAME, ride_minute) for ride_minute in control_ride_minutes_array]
        
        control_ride_minutes = np.sum(control_ride_minutes_array)
        control_ride_revenue = np.sum(control_ride_revenue_array)
        
        results_df0.at[idx, 'total_control_ride_minutes'] = control_ride_minutes
        results_df0.at[idx, 'control_ride_minutes_array'] = str(control_ride_minutes_array.tolist())
        
        results_df0.at[idx, 'total_control_ride_revenue'] = control_ride_revenue
        results_df0.at[idx, 'control_ride_revenue_array'] = str(control_ride_revenue_array)
        
        # special case for late signups
        if _late_signup:
            num_rides_pre_signup = row['rides_taken_control']
            num_rides_post_signup = row['additional_rides_taken']
            
            try:
                late_signup_standard_ride_minutes_array = apply_ride_duration_minutes_standard(num_rides_pre_signup)
                late_signup_membership_ride_minutes_array = apply_ride_duration_minutes_membership(num_rides_post_signup)
                late_signup_full_ride_minutes_array = np.concatenate((late_signup_standard_ride_minutes_array,
                                                                      late_signup_membership_ride_minutes_array))
            except Exception as e:
                logging.warning(e)
                logging.warning(f'Issue on late signup rider in iteration {iteration} and month {month}')
            
            # This is the pricing for modeled minutes ridden per ride for standard riders before membership pricing
            initial_standard_ride_revenue_array = [initial_price_for_ride(MARKET_NAME, ride_minute) for ride_minute in late_signup_standard_ride_minutes_array]
            
            # This is the adjusted membership price for modeled minutes ridden per ride for standard riders
            adjusted_standard_ride_revenue_array = [membership_price_adjustment(MARKET_NAME, _tier_chosen, ride_minute) for ride_minute in late_signup_standard_ride_minutes_array]
            
            # This is the pricing for modeled minutes ridden per ride for membership riders before membership pricing
            initial_membership_ride_revenue_array = [initial_price_for_ride(MARKET_NAME, ride_minute) for ride_minute in late_signup_membership_ride_minutes_array]
            
            # This is the adjusted membership price for modeled minutes ridden per ride for membership riders
            adjusted_membership_ride_revenue_array = [membership_price_adjustment(MARKET_NAME, _tier_chosen, ride_minute) for ride_minute in late_signup_membership_ride_minutes_array]

            # aggregations of the above arrays   
            # for late signups, these will be used for pre and post membership signup
            total_standard_minutes = np.sum(late_signup_standard_ride_minutes_array)
            total_membership_minutes = np.sum(late_signup_membership_ride_minutes_array)
            total_initial_standard_ride_revenue = np.sum(initial_standard_ride_revenue_array)  # ride revenue if rider was standard
            total_adjusted_standard_ride_revenue = np.sum(adjusted_standard_ride_revenue_array)  # ride revenue if rider rode like standard rider with membership pricing
            total_initial_membership_ride_revenue = np.sum(initial_membership_ride_revenue_array)
            total_adjusted_membership_ride_revenue = np.sum(adjusted_membership_ride_revenue_array)
            
            # assignment in the colulmns in the results dataframe
            results_df0.at[idx, 'initial_standard_ride_revenue'] = total_initial_standard_ride_revenue
            results_df0.at[idx, 'initial_standard_ride_revenue_array'] = str(initial_standard_ride_revenue_array)
            results_df0.at[idx, 'adjusted_standard_ride_revenue'] = total_adjusted_standard_ride_revenue
            results_df0.at[idx, 'adjusted_standard_ride_revenue_array'] = str(adjusted_standard_ride_revenue_array)
            
            results_df0.at[idx, 'initial_membership_ride_revenue'] = total_initial_membership_ride_revenue
            results_df0.at[idx, 'initial_membership_ride_revenue_array'] = str(initial_membership_ride_revenue_array)
            results_df0.at[idx, 'adjusted_membership_ride_revenue'] = total_adjusted_membership_ride_revenue
            results_df0.at[idx, 'adjusted_membership_ride_revenue_array'] = str(adjusted_membership_ride_revenue_array)
            
            results_df0.at[idx, 'total_standard_minutes'] = total_standard_minutes
            results_df0.at[idx, 'total_membership_minutes'] = total_membership_minutes
            results_df0.at[idx, 'standard_ride_minutes_array'] = str(late_signup_standard_ride_minutes_array.tolist())
            results_df0.at[idx, 'membership_ride_minutes_array'] = str(late_signup_membership_ride_minutes_array.tolist())
        
        else:
            
            # business as usual for everyone else who isn't a late signup
            try:
                standard_ride_minutes_array = apply_ride_duration_minutes_standard(num_rides)
                membership_ride_minutes_array = apply_ride_duration_minutes_membership(num_rides)
                # print(idx)
                # print(f'What does the ride minutes array look like? {ride_minutes_array}')
            except Exception as e:
                logging.warning(e)
                logging.warning(f'Issue with num_rides in iteration {iteration} and month {month}')
                continue
            
            # This is the pricing for modeled minutes ridden per ride for standard riders before membership pricing
            initial_standard_ride_revenue_array = [initial_price_for_ride(MARKET_NAME, ride_minute) for ride_minute in standard_ride_minutes_array]
            
            # This is the adjusted membership price for modeled minutes ridden per ride for standard riders
            adjusted_standard_ride_revenue_array = [membership_price_adjustment(MARKET_NAME, _tier_chosen, ride_minute) for ride_minute in standard_ride_minutes_array]
            
            # This is the pricing for modeled minutes ridden per ride for membership riders before membership pricing
            initial_membership_ride_revenue_array = [initial_price_for_ride(MARKET_NAME, ride_minute) for ride_minute in membership_ride_minutes_array]
            
            # This is the adjusted membership price for modeled minutes ridden per ride for membership riders
            adjusted_membership_ride_revenue_array = [membership_price_adjustment(MARKET_NAME, _tier_chosen, ride_minute) for ride_minute in membership_ride_minutes_array]

            # aggregations of the above arrays   
            total_standard_minutes = np.sum(standard_ride_minutes_array)
            total_membership_minutes = np.sum(membership_ride_minutes_array)
            total_initial_standard_ride_revenue = np.sum(initial_standard_ride_revenue_array)  # ride revenue if rider was standard
            total_adjusted_standard_ride_revenue = np.sum(adjusted_standard_ride_revenue_array)  # ride revenue if rider rode like standard rider with membership pricing
            total_initial_membership_ride_revenue = np.sum(initial_membership_ride_revenue_array)
            total_adjusted_membership_ride_revenue = np.sum(adjusted_membership_ride_revenue_array)
            
            # assignment in the colulmns in the results dataframe
            results_df0.at[idx, 'initial_standard_ride_revenue'] = total_initial_standard_ride_revenue
            results_df0.at[idx, 'initial_standard_ride_revenue_array'] = str(initial_standard_ride_revenue_array)
            results_df0.at[idx, 'adjusted_standard_ride_revenue'] = total_adjusted_standard_ride_revenue
            results_df0.at[idx, 'adjusted_standard_ride_revenue_array'] = str(adjusted_standard_ride_revenue_array)
            
            results_df0.at[idx, 'initial_membership_ride_revenue'] = total_initial_membership_ride_revenue
            results_df0.at[idx, 'initial_membership_ride_revenue_array'] = str(initial_membership_ride_revenue_array)
            results_df0.at[idx, 'adjusted_membership_ride_revenue'] = total_adjusted_membership_ride_revenue
            results_df0.at[idx, 'adjusted_membership_ride_revenue_array'] = str(adjusted_membership_ride_revenue_array)
            
            results_df0.at[idx, 'total_standard_minutes'] = total_standard_minutes
            results_df0.at[idx, 'total_membership_minutes'] = total_membership_minutes
            results_df0.at[idx, 'standard_ride_minutes_array'] = str(standard_ride_minutes_array.tolist())
            results_df0.at[idx, 'membership_ride_minutes_array'] = str(membership_ride_minutes_array.tolist())
        
        # adding this here just to double check everyone gets assigned a membership price
        _signed_up = row['membership_signed_up']
        
        if _signed_up:
            results_df0.at[idx, 'membership_revenue'] = T1_MONTHLY_PRICE
        
    
        
        # NOTE: Don't need to include this since we spelled it out up above for each customer type
        # if _signed_up == True:
        #     results_df0.at[idx, 'adjusted_revenue'] = total_adjusted_ride_revenue
        #     results_df0.at[idx, 'adjusted_revenue_array'] = str(adjusted_ride_revenue_array)
        # else:
        #     results_df0.at[idx, 'adjusted_revenue'] = total_initial_ride_revenue
        #     results_df0.at[idx, 'adjusted_revenue_array'] = str(initial_ride_revenue_array)
        
    save_filepath = f'./data/{SIM_VERISON}_{T1_MONTHLY_PRICE}_price_results'
    try:
        os.makedirs(save_filepath)
    except FileExistsError as e:
        pass
    
    results_df0.to_csv(f'{save_filepath}/i{iteration}_m{month}_results.csv', index=False)
     
     # FOR NOW, let's forget about subsequent months   
    ################# END MONTH 0  #################
    # ################# ################# #################
    # NOW LET'S LOOK AT MONTH 1 (Second Month)
#     month = 1
    
#     # how many of the new membership customers from previous month (month 0) be active this month (month 1)?
#     num_existing_and_membership_signup = results_df0.loc[(results_df0['is_existing_customer']==True) & (results_df0['membership_signed_up']==True)].shape[0]
#     num_new_and_membership_signup = results_df0.loc[(results_df0['is_existing_customer']==False) & (results_df0['membership_signed_up']==True)].shape[0]
#     num_standard_riders = results_df0.loc[results_df0['membership_signed_up']==False].shape[0]
    
#     active_customers_into_month1 = int(np.sum([(num_standard_riders * probability_standard_active_next_month0),
#                                                (num_new_and_membership_signup * probability_membership_new_signup_active_next_month0),
#                                                (num_existing_and_membership_signup * probabillity_membership_existing_active_next_month0)]))
    
#     # NOTE: Assuming that these active customers will be in addition to the historical MAU count
#     # which says that memberships will add to the historical MAU + new signups we expect
#     # new signups percentage will be applied to the mau selected from the generator
    
#     mau_month1, \
#     new_signups_percentage_month1, \
#     existing_customer_uptake_percentage_month1, \
#     new_customer_uptake_percentage_month1, \
#     probability_membership_new_signup_active_next_month1, \
#     probabillity_membership_existing_active_next_month1, \
#     probability_standard_active_next_month1 = roll_the_dice_month1(MARKET_NAME)
    
#      # STEP 2: Create the customer indices
#     # assumes that initial MAU is estimated from previous month
#     num_existing_customers_month1 = int(mau_month1) + active_customers_into_month1
#     new_signups_month1 = int(num_existing_customers_month1 * new_signups_percentage_month1)
#     total_customer_list_month1 = np.arange(num_existing_customers_month1 + new_signups_month1).tolist()
    
#     # STEP 3: Existing customer or new customer sign up
#     existing_customer_month_list = copy.copy(total_customer_list_month1)
#     existing_customer_month_list[:num_existing_customers_month1] = [True for i in range(len(existing_customer_month_list[:num_existing_customers_month1]))]
#     existing_customer_month_list[num_existing_customers_month1:] = [False for i in range(len(existing_customer_month_list[num_existing_customers_month1:]))]
            
    
#     # STEP 4: Work out of a dataframe to make this easier
#     results_df1 = pd.DataFrame()
#     results_df1['iteration'] = np.repeat(iteration, len(total_customer_list_month1))
#     results_df1['month'] = np.repeat(month, len(total_customer_list_month1))
#     results_df1['customers'] = total_customer_list_month1
#     results_df1['is_existing_customer'] = existing_customer_month_list
    
#     # STEP 5: Assign the membership signup boolean
#     # this needs to change a bit to keep the membership status for those that signed up for it month 0
#     results_df1['membership_signed_up'] = False
    
#       # here is the line of code to adjust this for pre-existing membership people
#     # this is meant to be from month0...don't change
#     # this looks at people who are existing but have been assigned as false for signing up
#     num_active_membership_riders_month0 = int(np.sum([(num_new_and_membership_signup * abs(probability_membership_new_signup_active_next_month0)),
#                                                       (num_existing_and_membership_signup * abs(probabillity_membership_existing_active_next_month0))]))
#     counter = 1
#     for idx, row in results_df1.loc[(results_df1['is_existing_customer']==True) & (results_df1['membership_signed_up']==False)].iterrows():
#         if counter <= num_active_membership_riders_month0:
#             row['membership_signed_up'] = True
#             counter += 1
#         else:
#             continue
   
#     # the above adjustment routine was brought upwards so that we don't double select customers that get assigned to signup for the membership
#     existing_customers = results_df1.loc[(results_df1['is_existing_customer']==True) & (results_df1['membership_signed_up']==False), 'customers']
#      # End of adjustment for existing membership riders
#     n_existing_customers_chosen = int(len(existing_customers) * existing_customer_uptake_percentage_month1)
#     existing_customers_chosen = np.random.choice(existing_customers, 
#                                                  size=n_existing_customers_chosen, 
#                                                  replace=False)
    
#     results_df1.loc[results_df1['customers'].isin(existing_customers_chosen), 'membership_signed_up'] = True
                                                                                             
  
    
#     new_customers = results_df1.loc[results_df1['is_existing_customer']==False, 'customers']
#     n_new_customers_chosen = int(len(new_customers) * new_customer_uptake_percentage_month1)
#     new_customers_chosen = np.random.choice(new_customers,
#                                             size=n_new_customers_chosen,
#                                             replace=False)
#     results_df1.loc[results_df1['customers'].isin(new_customers_chosen), 'membership_signed_up'] = True
    
#     # STEP 6: For people who signed up, which tier did they choose?
#     # given that the selection for who would be active from memberships is random, we should be okay resetting the tier selections for the next month
#     # NOTE: Audit that assumption ^^
#     results_df1['tier_chosen'] = np.nan
#     results_df1['membership_revenue'] = 0.0
#     array_length = results_df1.loc[results_df1['membership_signed_up']==True].shape[0]
#     results_df1.loc[results_df1['membership_signed_up']==True, 'tier_chosen'] = [tier_chosen() for i in range(array_length)]
    
#     # STEP 7: How many rides did each person take this month?
#     results_df1['rides_taken'] = 0
#     array_length = results_df1.shape[0]
#     results_df1['rides_taken'] = apply_number_of_rides(array_length)
    
#     # STEP 7A: What is initial number of rides if membership riders decide to try and make up for monthly price with unlock fee waived?
#     for idx, row in results_df0.loc[results_df0['membership_signed_up']==True].iterrows():
#         # TODO: finish the adjustment routine to account for people wanting to make up for the monthly price
#         _tier_chosen = row['tier_chosen']
#         initial_rides = row['rides_taken']
#         will_make_attempt = rider_tries_to_make_up_for_monthly_price()
        
#         if _tier_chosen == TIER1:
#             results_df1.at[idx,'membership_revenue'] = T1_MONTHLY_PRICE
#         elif _tier_chosen == TIER2:
#             results_df1.at[idx,'membership_revenue'] = T2_MONTHLY_PRICE
#         elif _tier_chosen == TIER3:
#             results_df1.at[idx,'membership_revenue'] = T3_MONTHLY_PRICE
        
#         if will_make_attempt:
#             if _tier_chosen == TIER1 and initial_rides < T1_UNLOCK_BREAKEVEN:
#                 results_df1.at[idx, 'rides_taken'] = T1_UNLOCK_BREAKEVEN - np.random.choice([0, 1, 2, 3], p=[.1, .2, .2, .5])
#             elif _tier_chosen == TIER2 and initial_rides < T2_UNLOCK_BREAKEVEN:
#                 results_df1.at[idx, 'rides_taken'] = T2_UNLOCK_BREAKEVEN - np.random.choice([0, 1, 2, 3, 4], p=[.01, .09, .15, .25, .5])
#             elif _tier_chosen == TIER3 and initial_rides < T3_UNLOCK_BREAKEVEN:
#                 results_df1.at[idx, 'rides_taken'] = T3_UNLOCK_BREAKEVEN - (int(10 * np.random.beta(6, 4)))
#             else:
#                 continue
#         else:
#             continue
    
#     # STEP 8: How many additional rides did a membership customer take this month?
#     results_df1['additional_rides_taken'] = 0
#     array_length = len(results_df1[results_df1['membership_signed_up']==True])
#     results_df1.loc[results_df1['membership_signed_up']==True, 'additional_rides_taken'] = additional_monthly_rides_from_membership(array_length)
    
#     results_df1['total_rides_taken'] = results_df1['rides_taken'] + results_df1['additional_rides_taken']
    
    
#     # STEP 9: Based on the monthly rides taken, assign the total minutes per trip where probability distribution is called for each trip taken
#     # and get the ride revenue based on the state of the customer
    
#     results_df1['initial_ride_revenue'] = 0.0
#     results_df1['initial_ride_revenue_array'] = 0
#     results_df1['adjusted_revenue'] = 0.0
#     results_df1['adjusted_revenue_array'] = 0
#     results_df1['total_minutes'] = 0.0
#     results_df1['ride_minutes_array'] = 0
    
#     for idx, row in results_df1.iterrows():
#         num_rides = row['total_rides_taken']
#         _tier_chosen = row['tier_chosen']
#         _signed_up = row['membership_signed_up']
        
#         try:
#             ride_minutes_array = apply_ride_duration_minutes(num_rides)
#         except Exception as e:
#             # logging.warning(e)
#             # logging.warning(f'Issue with num_rides in iteration {iteration} and month {month}')
#             continue
#         initial_ride_revenue_array = [initial_price_for_ride(MARKET_NAME, ride_minute) for ride_minute in ride_minutes_array]
#         adjusted_ride_revenue_array = [membership_price_adjustment(MARKET_NAME, _tier_chosen, ride_minute) for ride_minute in ride_minutes_array]
        
#         total_minutes = np.sum(ride_minutes_array)
#         total_initial_ride_revenue = np.sum(initial_ride_revenue_array)
#         total_adjusted_ride_revenue = np.sum(adjusted_ride_revenue_array)
        
#         results_df1.at[idx, 'initial_ride_revenue'] = total_initial_ride_revenue
#         results_df1.at[idx, 'initial_ride_revenue_array'] = str(initial_ride_revenue_array)
        
#         results_df1.at[idx, 'total_minutes'] = total_minutes
#         results_df1.at[idx, 'ride_minutes_array'] = str(ride_minutes_array.tolist())
        
#         if _signed_up == True:
#             results_df1.at[idx, 'adjusted_revenue'] = total_adjusted_ride_revenue
#             results_df1.at[idx, 'adjusted_revenue_array'] = str(adjusted_ride_revenue_array)
#         else:
#             results_df1.at[idx, 'adjusted_revenue'] = total_initial_ride_revenue
#             results_df1.at[idx, 'adjusted_revenue_array'] = str(initial_ride_revenue_array)
        
        
    
#     # results_df1.to_csv(f'./data/i{iteration}_m{month}_results.csv', index=False)
    
#      ################# END MONTH 1  #################
#     # ################# ################# #################
#     # NOW LET'S LOOK AT MONTH 2 (Third Month)
    
#     month = 2
    
#     # how many of the new membership customers from first month (month 0) be active this month (month 2)?
#     # we should probably roll the dice again to understand the month 0 effects on month 2
# #     num_existing_and_membership_signup = results_df0.iloc[results_df0['is_existing_customer']==True & results_df0['membership_signed_up']==True].shape[0]
# #     num_new_and_membership_signup = results_df0.iloc[results_df0['is_existing_customer']==False & results_df0['membership_signed_up']==True].shape[0]
# #     num_standard_riders = results_df0.iloc[results_df0['membership_signed_up']==False]
    
# #     active_customers_into_month2 = int(np.sum([(num_standard_riders * probability_standard_active_next_month0()),
# #                                                (num_new_and_membership_signup * probability_membership_new_signup_active_next_month0()),
# #                                                (num_existing_and_membership_signup * probabillity_membership_existing_active_next_month0())]))
# # NOTE: Maybe we don't need to model the diminished probability of cohort from month 0
    
#     # how many of the new membership customers from previous month (month 1) be active this month (month 2)?
#     num_existing_and_membership_signup = results_df1.loc[(results_df1['is_existing_customer']==True) & (results_df1['membership_signed_up']==True)].shape[0]
#     num_new_and_membership_signup = results_df1.loc[(results_df1['is_existing_customer']==False) & (results_df1['membership_signed_up']==True)].shape[0]
#     num_standard_riders = results_df1.loc[results_df1['membership_signed_up']==False].shape[0]
    
#     active_customers_into_month2 = int(np.sum([(num_standard_riders * probability_standard_active_next_month1),
#                                                (num_new_and_membership_signup * probability_membership_new_signup_active_next_month1),
#                                                (num_existing_and_membership_signup * probabillity_membership_existing_active_next_month1)]))
    
    
#     # NOTE: Assuming that these active customers will be in addition to the historical MAU count
#     # which says that memberships will add to the historical MAU + new signups we expect
#     # new signups percentage will be applied to the mau selected from the generator
    
#     mau_month2, \
#     new_signups_percentage_month2, \
#     existing_customer_uptake_percentage_month2, \
#     new_customer_uptake_percentage_month2, \
#     _, \
#     _, \
#     _ = roll_the_dice_month2(MARKET_NAME)
    
#      # STEP 2: Create the customer indices
#     # assumes that initial MAU is estimated from previous month
#     num_existing_customers_month2 = int(mau_month2) + active_customers_into_month2
#     new_signups_month2 = int(num_existing_customers_month2 * new_signups_percentage_month2)
#     total_customer_list_month2 = np.arange(num_existing_customers_month2 + new_signups_month2).tolist()
    
#     # STEP 3: Existing customer or new customer sign up
#     existing_customer_month_list = copy.copy(total_customer_list_month2)
#     existing_customer_month_list[:num_existing_customers_month2] = [True for i in range(len(existing_customer_month_list[:num_existing_customers_month2]))]
#     existing_customer_month_list[num_existing_customers_month2:] = [False for i in range(len(existing_customer_month_list[num_existing_customers_month2:]))]
            
    
#     # STEP 4: Work out of a dataframe to make this easier
#     results_df2 = pd.DataFrame()
#     results_df2['iteration'] = np.repeat(iteration, len(total_customer_list_month2))
#     results_df2['month'] = np.repeat(month, len(total_customer_list_month2))
#     results_df2['customers'] = total_customer_list_month2
#     results_df2['is_existing_customer'] = existing_customer_month_list
    
#     # STEP 5: Assign the membership signup boolean
#     # this needs to change a bit to keep the membership status for those that signed up for it month 0
#     results_df2['membership_signed_up'] = False
    
#       # here is the line of code to adjust this for pre-existing membership people
#     # this is meant to be from month0...don't change
#     # this looks at people who are existing but have been assigned as false for signing up
#     num_active_membership_riders_month2 = int(np.sum([(num_new_and_membership_signup * probability_membership_new_signup_active_next_month1),
#                                                       (num_existing_and_membership_signup * probabillity_membership_existing_active_next_month1)]))
#     counter = 1
#     for idx, row in results_df2.loc[(results_df2['is_existing_customer']==True) & (results_df2['membership_signed_up']==False)].iterrows():
#         if counter <= num_active_membership_riders_month0:
#             row['membership_signed_up'] = True
#             counter += 1
#         else:
#             continue
   
#     # the above adjustment routine was brought upwards so that we don't double select customers that get assigned to signup for the membership
#     existing_customers = results_df2.loc[(results_df2['is_existing_customer']==True) & (results_df2['membership_signed_up']==False), 'customers']
#      # End of adjustment for existing membership riders
#     n_existing_customers_chosen = int(len(existing_customers) * existing_customer_uptake_percentage_month2)
#     existing_customers_chosen = np.random.choice(existing_customers, 
#                                                  size=n_existing_customers_chosen, 
#                                                  replace=False)
    
#     results_df2.loc[results_df2['customers'].isin(existing_customers_chosen), 'membership_signed_up'] = True
                                                                                             
  
    
#     new_customers = results_df2.loc[results_df2['is_existing_customer']==False, 'customers']
#     n_new_customers_chosen = int(len(new_customers) * new_customer_uptake_percentage_month2)
#     new_customers_chosen = np.random.choice(new_customers,
#                                             size=n_new_customers_chosen,
#                                             replace=False)
#     results_df2.loc[results_df2['customers'].isin(new_customers_chosen), 'membership_signed_up'] = True
    
#     # STEP 6: For people who signed up, which tier did they choose?
#     # given that the selection for who would be active from memberships is random, we should be okay resetting the tier selections for the next month
#     # NOTE: Audit that assumption ^^
#     results_df2['tier_chosen'] = np.nan
#     results_df2['membership_revenue'] = 0.0
#     array_length = results_df2.loc[results_df2['membership_signed_up']==True].shape[0]
#     results_df2.loc[results_df2['membership_signed_up']==True, 'tier_chosen'] = [tier_chosen() for i in range(array_length)]
    
#     # STEP 7: How many rides did each person take this month?
#     results_df2['rides_taken'] = 0
#     array_length = results_df2.shape[0]
#     results_df2['rides_taken'] = apply_number_of_rides(array_length)
    
#     # STEP 7A: What is initial number of rides if membership riders decide to try and make up for monthly price with unlock fee waived?
#     for idx, row in results_df2.loc[results_df2['membership_signed_up']==True].iterrows():
#         # TODO: finish the adjustment routine to account for people wanting to make up for the monthly price
#         _tier_chosen = row['tier_chosen']
#         initial_rides = row['rides_taken']
#         will_make_attempt = rider_tries_to_make_up_for_monthly_price()
        
#         if _tier_chosen == TIER1:
#             results_df2.at[idx, 'membership_revenue'] = T1_MONTHLY_PRICE
#         elif _tier_chosen == TIER2:
#             results_df2.at[idx, 'membership_revenue'] = T2_MONTHLY_PRICE
#         elif _tier_chosen == TIER3:
#             results_df2.at[idx, 'membership_revenue'] = T3_MONTHLY_PRICE
        
#         if will_make_attempt:
#             if _tier_chosen == TIER1 and initial_rides < T1_UNLOCK_BREAKEVEN:
#                 results_df2.at[idx, 'rides_taken'] = T1_UNLOCK_BREAKEVEN - np.random.choice([0, 1, 2, 3], p=[.1, .2, .2, .5])
#             elif _tier_chosen == TIER2 and initial_rides < T2_UNLOCK_BREAKEVEN:
#                 results_df2.at[idx, 'rides_taken'] = T2_UNLOCK_BREAKEVEN - np.random.choice([0, 1, 2, 3, 4], p=[.01, .09, .15, .25, .5])
#             elif _tier_chosen == TIER3 and initial_rides < T3_UNLOCK_BREAKEVEN:
#                 results_df2.at[idx, 'rides_taken'] = T3_UNLOCK_BREAKEVEN - (int(10 * np.random.beta(6, 4)))
#             else:
#                 continue
#         else:
#             continue
    
#     # STEP 8: How many additional rides did a membership customer take this month?
#     results_df2['additional_rides_taken'] = 0
#     array_length = len(results_df2[results_df2['membership_signed_up']==True])
#     results_df2.loc[results_df2['membership_signed_up']==True, 'additional_rides_taken'] = additional_monthly_rides_from_membership(array_length)
    
#     results_df2['total_rides_taken'] = results_df2['rides_taken'] + results_df2['additional_rides_taken']
    
    
#     # STEP 9: Based on the monthly rides taken, assign the total minutes per trip where probability distribution is called for each trip taken
#     # and get the ride revenue based on the state of the customer
    
#     results_df2['initial_ride_revenue'] = 0.0
#     results_df2['initial_ride_revenue_array'] = 0
#     results_df2['adjusted_revenue'] = 0.0
#     results_df2['adjusted_revenue_array'] = 0
#     results_df2['total_minutes'] = 0.0
#     results_df2['ride_minutes_array'] = 0
    
#     for idx, row in results_df2.iterrows():
#         num_rides = row['total_rides_taken']
#         _tier_chosen = row['tier_chosen']
#         _signed_up = row['membership_signed_up']
        
#         try:
#             ride_minutes_array = apply_ride_duration_minutes(num_rides)
#         except Exception as e:
#             logging.warning(e)
#             logging.warning(f'Issue with num_rides in iteration {iteration} and month {month}')
#             continue
        
#         initial_ride_revenue_array = [initial_price_for_ride(MARKET_NAME, ride_minute) for ride_minute in ride_minutes_array]
#         adjusted_ride_revenue_array = [membership_price_adjustment(MARKET_NAME, _tier_chosen, ride_minute) for ride_minute in ride_minutes_array]
        
#         total_minutes = np.sum(ride_minutes_array)
#         total_initial_ride_revenue = np.sum(initial_ride_revenue_array)
#         total_adjusted_ride_revenue = np.sum(adjusted_ride_revenue_array)
        
#         results_df2.at[idx, 'initial_ride_revenue'] = total_initial_ride_revenue
#         results_df2.at[idx, 'initial_ride_revenue_array'] = str(initial_ride_revenue_array)
        
#         results_df2.at[idx, 'total_minutes'] = total_minutes
#         results_df2.at[idx, 'ride_minutes_array'] = str(ride_minutes_array.tolist())
        
#         if _signed_up == True:
#             results_df2.at[idx, 'adjusted_revenue'] = total_adjusted_ride_revenue
#             results_df2.at[idx, 'adjusted_revenue_array'] = str(adjusted_ride_revenue_array)
#         else:
#             results_df2.at[idx, 'adjusted_revenue'] = total_initial_ride_revenue
#             results_df2.at[idx, 'adjusted_revenue_array'] = str(initial_ride_revenue_array)
        
        
    
#     # results_df2.to_csv(f'./data/i{iteration}_m{month}_results.csv', index=False)
    # TODO: let's work on the agg stats results for each iteration
    # thinking each row is an iteration and then each col is appended with month index suffix
    
    
    # calculating revenue revenue under a world with memberships
    late_signup_ride_revenue_before_signing = results_df0.loc[results_df0['existing_customer_late_signup']==True, 'initial_standard_ride_revenue'].sum()
    late_signup_ride_revenue_after_signing = results_df0.loc[results_df0['existing_customer_late_signup']==True, 'adjusted_membership_ride_revenue'].sum()
    total_late_signup_ride_revenue = np.sum([late_signup_ride_revenue_before_signing, 
                                            late_signup_ride_revenue_after_signing])
    
    other_membership_ride_revenue = results_df0.loc[((results_df0['membership_signed_up']==True) & (results_df0['existing_customer_late_signup']==False)), 'adjusted_membership_ride_revenue'].sum()
    
    # calculating total monthly fee revenue
    monthly_membership_fee_revenue = results_df0.loc[results_df0['membership_signed_up']==True, 'membership_revenue'].sum()
    
    total_membership_revenue = np.sum([total_late_signup_ride_revenue,
                                      other_membership_ride_revenue,
                                      monthly_membership_fee_revenue])
    
    # closer look at those who breakeven -- excludes membership fees
    total_membership_revenue_riders_over_breakeven = results_df0.loc[((results_df0['membership_signed_up']==True) & (results_df0['total_rides_taken'] >= T1_UNLOCK_BREAKEVEN)), 'adjusted_membership_ride_revenue'].sum()
    total_standard_revenue_riders_over_breakeven = results_df0.loc[((results_df0['membership_signed_up']==False) & (results_df0['total_rides_taken'] >= T1_UNLOCK_BREAKEVEN)), 'initial_standard_ride_revenue'].sum()
    
    total_membership_ride_mintues_riders_over_breakeven = results_df0.loc[((results_df0['membership_signed_up']==True) & (results_df0['total_rides_taken'] >= T1_UNLOCK_BREAKEVEN)), 'total_membership_minutes'].sum()
    total_standard_ride_mintues_riders_over_breakeven = results_df0.loc[((results_df0['membership_signed_up']==False) & (results_df0['total_rides_taken'] >= T1_UNLOCK_BREAKEVEN)), 'total_standard_minutes'].sum()
    
    num_membership_rides_riders_over_breakeven = results_df0.loc[((results_df0['membership_signed_up']==True) & (results_df0['total_rides_taken'] >= T1_UNLOCK_BREAKEVEN)), 'total_rides_taken'].sum()
    num_standard_rides_riders_over_breakeven = results_df0.loc[((results_df0['membership_signed_up']==False) & (results_df0['total_rides_taken'] >= T1_UNLOCK_BREAKEVEN)), 'total_rides_taken'].sum()
    
    iteration_results = {
        'iteration': iteration,
        'mau_month0': mau_month0,
        'total_customers_month0': results_df0.shape[0],
        'new_signups_percentage_month0': new_signups_percentage_month0,
        'existing_customer_uptake_percentage_month0': existing_customer_uptake_percentage_month0,
        'total_late_signups': results_df0.loc[results_df0['existing_customer_late_signup']==True].shape[0],
        'new_customer_uptake_percentage_month0': new_customer_uptake_percentage_month0,
        'probability_membership_new_signup_active_next_month0': probability_membership_new_signup_active_next_month0,
        'probabillity_membership_existing_active_next_month0': probabillity_membership_existing_active_next_month0,
        'probability_standard_active_next_month0': probability_standard_active_next_month0,
        # 'tier_count_month0': str(results_df0['tier_chosen'].value_counts()),

        # 'mau_month1': mau_month1,
        # 'total_customers_month1': results_df1.shape[0],
        # 'active_customers_into_month1': active_customers_into_month1,
        # 'new_signups_percentage_month1': new_signups_percentage_month1,
        # 'existing_customer_uptake_percentage_month1': existing_customer_uptake_percentage_month1,
        # 'new_customer_uptake_percentage_month1': new_customer_uptake_percentage_month1,
        # 'probability_membership_new_signup_active_next_month1': probability_membership_new_signup_active_next_month1,
        # 'probabillity_membership_existing_active_next_month1': probabillity_membership_existing_active_next_month1,
        # 'probability_standard_active_next_month1': probability_standard_active_next_month1,
        # # 'tier_count_month1': str(results_df1['tier_chosen'].value_counts()),

        # 'mau_month2': mau_month2,
        # 'total_custoemrs_month2': results_df2.shape[0],
        # 'active_customers_into_month2': active_customers_into_month2,
        # 'new_signups_percentage_month2': new_signups_percentage_month2,
        # 'existing_customer_uptake_percentage_month2': existing_customer_uptake_percentage_month2,
        # 'new_customer_uptake_percentage_month2': new_customer_uptake_percentage_month2,
        # 'probability_membership_new_signup_active_next_month2': probability_membership_new_signup_active_next_month2,
        # 'probabillity_membership_existing_active_next_month2': probabillity_membership_existing_active_next_month2,
        # 'probability_standard_active_next_month2': probability_standard_active_next_month2,
        # 'tier1_count_month2': str(results_df2['tier_chosen'].value_counts()),
        'counterfactual_ride_revenue_month0': results_df0['total_control_ride_revenue'].sum(),
        'counterfactual_number_rides_taken_month0': results_df0['rides_taken_control'].sum(),
        'total_membership_revenue_month0': total_membership_revenue,  # this will be ride revenue with membership fees, late signups, and all other members
        'treatment_world_number_rides_taken_month0': results_df0['rides_taken'].sum(),
        'num_customers_riding_over_breakeven': results_df0.loc[results_df0['total_rides_taken'] >= T1_UNLOCK_BREAKEVEN].shape[0],
        'total_membership_revenue_riders_over_breakeven': total_membership_revenue_riders_over_breakeven,
        'total_standard_revenue_riders_over_breakeven': total_standard_revenue_riders_over_breakeven,
        'total_membership_ride_mintues_riders_over_breakeven': total_membership_ride_mintues_riders_over_breakeven,
        'total_standard_ride_mintues_riders_over_breakeven': total_standard_ride_mintues_riders_over_breakeven,
        'num_membership_rides_riders_over_breakeven': num_membership_rides_riders_over_breakeven,
        'num_standard_rides_riders_over_breakeven': num_standard_rides_riders_over_breakeven
        # 'ride_revenue_wo_memberships_month0': results_df0['initial_ride_revenue'].sum(),
        # 'ride_revenue_adjusted_month0': results_df0['adjusted_revenue'].sum() + results_df0['membership_revenue'].sum(),
        # 'ride_revenue_wo_memberships_month1': results_df1['initial_ride_revenue'].sum(),
        # 'ride_revenue_adjusted_month1': results_df1['adjusted_revenue'].sum() + results_df1['membership_revenue'].sum(),
        # 'ride_revenue_wo_memberships_month2': results_df2['initial_ride_revenue'].sum(),
        # 'ride_revenue_adjusted_month2': results_df2['adjusted_revenue'].sum() + results_df2['membership_revenue'].sum(),
    }
    
    iteration_results_summary.append(iteration_results)

iteration_results_summary_df = pd.DataFrame(iteration_results_summary)
iteration_results_summary_df.to_csv(f'./data/{SIM_VERISON}_{T1_MONTHLY_PRICE}_price_iteration_results_summary.csv', index=False)
# NOTE: you commented out the writing out of resutls csvs since you had an error with iteration results summary before
# NOTE: you also commented out the logging
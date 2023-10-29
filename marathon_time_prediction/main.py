import pandas as pd
import numpy as np
from numpy.random import seed, RandomState
import datetime


# Desired start time
intended_start_time = np.datetime64('2018-09-28T06:00')
current_time = intended_start_time
night_time = np.datetime64('2018-09-28T22:00')
next_day_morning = np.datetime64('2018-09-29T07:00')

# Accounting for morning delay following normal distribution
def start_delay_min():
    delay = np.random.randint(0, 15)
    
    return delay


# actual start time in the morning
def actual_start_time():
    actual_start_time = intended_start_time + np.timedelta64(start_delay_min(), 'm')
    
    return actual_start_time


# state information for each mile...what happened in each mile? 0 = didn't happen, 1 = occurred
def take_water_break():
    return np.random.randint(0, 2)

def water_break_duration():
    wbduration = np.int(np.random.normal(3, .5))
    
    return wbduration


def take_bathroom_break():
    return np.random.randint(0, 2)

def bathroom_break_duration():
    bbduration = np.int(np.random.normal(4, 2))
    
    return bbduration


def engage_crowd():
    return np.random.randint(0, 2)


def crowd_engagement_duration():
    ceduration = np.int(np.random.normal(15, 1))
    
    return ceduration


def quick_pace_rate():
    return np.int(np.random.normal(10, 1.3))


def medium_pace_rate():
    return np.int(np.random.normal(13, 1.36))


def finishing_pace_rate():
    return np.int(np.random.normal(14, 3.2))


# Running 1000 different scenarios to see when we would end up in Columbus
scenarios = list((range(1,1001)))
start_time_list = []
miles = (list(range(1,143)))
scenario_index = []
mile_index = []
pace = []
waterduration = []
bathroombreakduration = []
crowdengagedduration = []
arrival_time = []

def main():
    seed_num = 0
    for scenario in scenarios:
        seed_num +=1
        seed(seed_num)
        current_time = actual_start_time()
        start_time_list.append(current_time)

        for mile in miles:

            mile_index.append(mile)
            scenario_index.append(scenario)

            water_break_taken = take_water_break()
            bathroom_break_taken = take_bathroom_break()
            crowd_engaged = engage_crowd()

            if mile < 10:
                pace.append(quick_pace_rate())
            elif mile >= 10 and mile < 80:
                pace.append(medium_pace_rate())
            else:
                pace.append(finishing_pace_rate())

            if water_break_taken == 1 and mile%3==0:
                waterduration.append(water_break_duration())
            else:
                waterduration.append(0)

            if bathroom_break_taken == 1 and (mile % 6 or mile % 9) == 0:
                bathroombreakduration.append(bathroom_break_duration())
            else:
                bathroombreakduration.append(0)

            if crowd_engaged==1 and mile % 6  == 0 and ((current_time > intended_start_time and current_time < night_time) or current_time > next_day_morning) :
                crowdengagedduration.append(crowd_engagement_duration())
            else:
                crowdengagedduration.append(0)

            mile_duration = np.sum([pace[mile-1],
                                    waterduration[mile-1],
                                    bathroombreakduration[mile-1],
                                    crowdengagedduration[mile-1],
                                    ])

            current_time += np.timedelta64(np.int(mile_duration), 'm')
            arrival_time.append(current_time)


    scenario_results = {'scenario': scenarios,
                        'start_time': start_time_list}

    sim_results = {'scenario': scenario_index,
                'mile': mile_index,
                'pace': pace,
                'water_break_duration': waterduration,
                'bathroom_break_duration': bathroombreakduration,
                'crowd_duration': crowdengagedduration,
                'arrival': arrival_time}
    
    scenario_results_df = pd.DataFrame(scenario_results)
    sim_results_df = pd.DataFrame(sim_results)
    
    scenario_results_df.to_csv('./data/results/scenario_results.csv', index=False)
    sim_results_df.to_csv('./data/resutsl/simulaton_results.csv', index=False)
    
    

if __name__ == '__main__':
    main()
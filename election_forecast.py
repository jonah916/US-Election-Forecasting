"""

This script makes a preliminary attempt to "reverse engineer" FiveThirtyEight's 2020 election model and forecast the results of that presidential election. 
It does so by aggregating data from 4,200 pre-election polls (found here: https://projects.fivethirtyeight.com/polls-page/data/president_polls_historical.csv), 
applying various weights, and simulating state elections.

FiveThirtyEight (FTE) is extremely transparent with its methodology, so my approach mostly followed the descriptions here:  
https://fivethirtyeight.com/features/how-fivethirtyeights-2020-presidential-forecast-works-and-whats-different-because-of-covid-19/.

FTE's model has three steps: 1) Collect and adjust polls, 2) Combine polls with demographic and economic data, and 3) Account for uncertainty and simulate the election
thousands of times. This script performs Step 1, calculating adjusted weighted polling averages for the major candidates (Joe Biden and Donald Trump) and using them
to simulate the resuts of an election.

The script processes and adjusts the polls in two general ways: selecting the best version of polls, and weighting certain polls more or less depending on a few
characteristics. Sometimes, a poll will have multiple versions for different populations -- likely voters, registered voters, all voters, or all adults. The most
representative polls are thought to be those that survey likely voters. The worst are those surveying all adults, regardless of voter status. Details on how the
script selects poll versions are included in comments in the script. The factors that affect a poll's weight are: the quality of the pollster, as graded by FTE;
the number of polls the pollster has conducted; the sample size; and how recently it occurred.

Despite omitting Steps 2 and 3 of FTE's model, this approach produces fairly accurate results. In most states, each candidate's weighted polling averages are close
to FTE's projections. In the simulation I demonstrate, you'll notice a few unusual results, such as Biden polling at 28.96% in his home state of Delaware (this is due
to 'other' candidates being over-represented in Delaware state polls) and large polling errors leading to Biden pulling off an upset in Montana and Trump upsetting in
Washington. In reality, such large errors in solidly red or blue states would be extremely unlikely, and if they did occur they would likely signal a major systematic
error in the polls, since state polling errors tend to be correlated. This approach does not take such correlations into account.

Please allow about 15 seconds for this script to run. The output is the results of simulated elections in all 50 states plus Washington, D.C. 
The winner of the election is printed at the very end.

Thanks for reading!

"""
# ----- Load libraries -----

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from datetime import datetime
import time
from scipy.stats import t
import random

# ----- Import data -----

print('~~~~~ now importing poll data ~~~~~')
fte = pd.read_csv('https://projects.fivethirtyeight.com/polls-page/data/president_polls_historical.csv',
                 dtype = {'sponsor_candidate': str, 'sponsor_candidate_party': str, 'tracking': str, 'notes': str})
print('~~~~~ data imported successfully ~~~~~\n')
print(fte.head()) # make sure the data imported correctly
time.sleep(2)

# ----- Define functions -----

# Two of the poll weights depend on pollster qualities, so we need to have easy access to each pollster's FTE grade and the number of polls they conducted
# This function outputs a dataframe with each pollsters count of polls (n_polls) and pollster rating (FTE_grade)
def pollsterInfo(polls, grades):
    # get number of polls by each pollster as well as each pollster's grade
    pollster_info = polls.groupby(['pollster']).agg({'poll_id':lambda x: x.nunique(),
                                                     'fte_grade': 'max'}).reset_index()
    pollster_info.columns = ['pollster','n_polls','fte_grade']
    
    # each poll is weighted by 1/sqrt(n) where n is the number of polls conducted by that pollster 
    # (see https://fivethirtyeight.com/features/the-death-of-polling-is-greatly-exaggerated/)
    pollster_info['pollster_n_weight'] = 1/np.sqrt(pollster_info.n_polls)
    
    # each poll is weighted according to FTE's rating of the pollster
    # the weights assigned to each grade are assigned below in main()
    pollster_info['pollster_rating_weight'] = [grades[grade] for grade in pollster_info.fte_grade]
    
    return pollster_info
    
    
# A number of pre-processing steps are necessary before the polls can be aggregated
def preProcessPolls(polls, grade_weights):
    processed_polls = polls.copy()
    
    # change dates from string to datetime
    processed_polls.loc[:,'start_date'] = pd.to_datetime(polls['start_date'])
    processed_polls.loc[:,'end_date'] = pd.to_datetime(polls['end_date'])
    
    # limit columns
    keep_cols =['poll_id','pollster_id','pollster','sponsors','display_name',
                'fte_grade','methodology','state','start_date','end_date',
                'question_id','sample_size','population','internal',
                'partisan','party','candidate_name','pct']
    processed_polls = processed_polls.loc[:,keep_cols]
    
    # create 'Other' candidate for all except the two major candidates
    major_cand_idx = processed_polls.candidate_name.isin(['Joe Biden','Donald Trump'])
    processed_polls.loc[~major_cand_idx, 'candidate_name'] = 'Other'
    processed_polls.loc[~major_cand_idx, 'party'] = 'OTHER'

    # if a poll had multiple versions for for likely voters, registered voters, etc., then we need to select the 'best' version
    # we'll do this by using ranked order created from the population variable: likely voters (lv) > registered voters (rv) > voters (v) > adults (a)
    pop_cats = CategoricalDtype(categories = ['a','v','rv','lv'], ordered = True)
    processed_polls['population'] = processed_polls['population'].astype(pop_cats)

    # remove polls from F-rated pollsters
    processed_polls = processed_polls[processed_polls.fte_grade != 'F']
    
    # some polls conducted as a collaboration by multiple pollsters are unrated
    # if any pollster involved has a solo grade, we assign that grade to the collaboration
    # if multiple pollsters involved have solo grades, we take the average of their grades and assign it to their collaboration
    processed_polls.loc[processed_polls.pollster == 'Reconnect Research/Roanoke College','fte_grade'] = polls.loc[polls.pollster =='Roanoke College','fte_grade'].iloc[0]
    processed_polls.loc[processed_polls.pollster == 'Benenson Strategy Group/GS Strategy Group','fte_grade'] = 'B/C' # comes from https://github.com/fivethirtyeight/data/blob/master/pollster-ratings/2020/pollster-ratings.csv
    processed_polls.loc[processed_polls.pollster == 'YouGov Blue/Data for Progress','fte_grade'] = polls.loc[polls.pollster =='Data for Progress','fte_grade'].iloc[0]
    processed_polls.loc[processed_polls.pollster == 'Global Strategy Group/Data for Progress','fte_grade'] = 'B-' # average of B and B/C
    processed_polls.loc[processed_polls.pollster == 'Montana State University Bozeman/University of Denver','fte_grade'] = 'B/C' # comes from https://github.com/fivethirtyeight/data/blob/master/pollster-ratings/2020/pollster-ratings.csv
    
    # remove the remaining polls whose pollsters are unrated
    processed_polls.dropna(subset=['fte_grade'], inplace=True)

    # get dataframe with pollster ratings and number of polls conducted
    pollster_info = pollsterInfo(processed_polls, grade_weights)
    
    return processed_polls, pollster_info
    
    
# Select the 'best' version of each poll using the ranked CategoricalDtype variable created in prior step
def filterPollVersions(polls):
    # get preferred version for each poll
    poll_pops = polls.groupby('poll_id',sort=False).population.max()
    poll_pops_df = pd.DataFrame({'poll_id':polls.poll_id.unique(),'population_to_use':poll_pops}).reset_index(drop=True)
    
    # only keep desired version for each poll
    desired_versions = polls.merge(poll_pops_df,on='poll_id')
    idx = desired_versions.population == desired_versions.population_to_use
    polls_filtered = polls[idx.values]
    
    return polls_filtered


# Polls with more people are weighted more heavily
def getSampleSizeWeight(sample_size): 
    weight = np.sqrt(sample_size/600) #based on https://fivethirtyeight.com/features/polls-now-weighted-by-sample-size/
    return weight


# Polls conducted early in the election cycle are included but get significantly less weight
# We include a forecast_date argument so we can see how the forecast has changed over the course of the cycle
def recencyWeight(poll_date, forecast_date):
    forecast_date_fmt = datetime.strptime(forecast_date, '%m/%d/%y')
    delta = forecast_date_fmt - poll_date
    days_since_poll = delta.days
    weeks_since_poll = delta.days//7
    weight = 0.95**weeks_since_poll # y=0.95^x; adapted from https://fivethirtyeight.com/features/the-death-of-polling-is-greatly-exaggerated/
    return weight
    
    
# Add a column to each poll for each of the 4 weights discussed above
def getWeights(polls, pollster_info, forecast_date):
    polls_with_weights = polls.copy()
    
    sample_size_weights = [getSampleSizeWeight(sample) for sample in polls.sample_size]
    pollster_n_weights = [pollster_info.loc[pollster_info['pollster'] == pollster, 'pollster_n_weight'].iloc[0] for pollster in polls.pollster]
    pollster_rating_weight = [pollster_info.loc[pollster_info['pollster'] == pollster, 'pollster_rating_weight'].iloc[0] for pollster in polls.pollster]
    recency_weight = [recencyWeight(date, forecast_date) for date in polls.end_date]

    # add weights to the polls dataset
    polls_with_weights.loc[:,'sample_size_weight'] = sample_size_weights
    polls_with_weights.loc[:,'pollster_n_weight'] = pollster_n_weights
    polls_with_weights.loc[:,'pollster_rating_weight'] = pollster_rating_weight
    polls_with_weights.loc[:,'recency_weight'] = recency_weight
    
    return polls_with_weights


# Multiply the weights from the previous step with each polling percentage to calculate weighted percentages 
def calculateWeightedAverage(polls_with_weights):
    polls_with_weighted_avgs = polls_with_weights.copy()
    
    weighted_avg = polls_with_weighted_avgs.loc[:,'pct':'recency_weight'].product(axis=1)
    # Add weighted percentages to the dataset
    polls_with_weighted_avgs.loc[:,'weighted_pct'] = weighted_avg
    
    return polls_with_weighted_avgs
    
    
# Take the weighted average for each candidate and standardize them so they sum to 100
def standardizeWeightedAverage(polls_with_weighted_averages):
    forecast = polls_with_weighted_averages.groupby(['candidate_name'])['weighted_pct'].mean()
    standardized_forecast = forecast/sum(forecast)
    return standardized_forecast.sort_values(ascending = False)
    
    
# Apply all of the above steps on one state at a time
def _part1_getPollingAverage(state, polls, pollster_info, forecast_date):

    # filter polls to state of interest
    state_polls = polls.loc[polls.state == state,] 
    
    # filter polls to on or before the date of the forecast
    state_polls = state_polls.loc[state_polls.end_date <= forecast_date,]
    
    # if there are multiple versions of a poll (rv/lv/v/a), only use the best one
    best_state_polls = filterPollVersions(state_polls) 
    
    # calculate weights
    polls_with_weights = getWeights(best_state_polls, pollster_info, forecast_date)
    
    # get weighted percentage for each poll
    polls_with_weighted_avg = calculateWeightedAverage(polls_with_weights)
    
    # get standardized weighted average for each candidate
    standardized_forecast = standardizeWeightedAverage(polls_with_weighted_avg)
                                                                                          
    return standardized_forecast

# ----- Simulate election -----
  
# Simulate the election results in each state, add up the electoral college votes, and determine a winner
def simulateSingleElection(polls, pollster_info, forecast_date, electoralVotes, marginOfError, verbose=False):
    Biden_win_count = 0
    Trump_win_count = 0
    Biden_votes_won = 0
    Trump_votes_won = 0
    for i in range(electoralVotes.shape[0]):
        
        state = electoralVotes.state[i] # iterate over each state
        votes = electoralVotes.votes[i] # and its number of electoral college votes
        
        # get each candidate's weighted average in the current state
        state_polling_avg = _part1_getPollingAverage(state, polls, pollster_info, forecast_date)
        Biden_state_polling_avg = float(state_polling_avg[state_polling_avg.index == 'Joe Biden']) * 100
       
        # generate a polling error from a t-distribution with 10 degrees of freedom
        polling_error = t.rvs(df=10)/2*marginOfError
        
        # forecast Biden's actual vote share
        Biden_state_vote_share = Biden_state_polling_avg + polling_error
        
        # update running tally of states won and electoral college votes won
        if Biden_state_vote_share > 50:
            Biden_win_count += 1
            Biden_votes_won += votes
        else:
            Trump_win_count += 1
            Trump_votes_won += votes
            
        if verbose:
            print('Now simulating '+state)
            print('Biden polling average: '+str(np.round(Biden_state_polling_avg,2)))
            print('Polling error: '+str(np.round(polling_error,2)))
            print('Biden vote share: '+str(np.round(Biden_state_vote_share,2)))
            print('Biden states won: '+str(Biden_win_count))
            print('Trump states won: '+str(Trump_win_count))
            print('Electoral vote tally:\nBiden: '+str(Biden_votes_won)+', Trump: '+str(Trump_votes_won))
            print('-----------------------------\n')
    
    # Biden victory
    if Biden_votes_won > Trump_votes_won:
        if verbose:
            print('Biden wins the election\n')
        return 'Biden'
    # Electoral college tie
    elif Biden_votes_won == Trump_votes_won:
        if verbose:
            print('It\'s a tie\n')
        return 'Tie'
    # Trump victory
    else:
        if verbose:
            print('Trump wins the election\n')
        return 'Trump'
        
    
# Will not run for the sake of time
# To simulate multiple elections and get a percentage forecast, simply run simulateSingleElection() n times,
# count how many wins Biden and Trump each had, and divide by n
def simulateMultipleElections(n, polls, pollster_info, forecast_date, electoralVotes, marginOfError, verbose=False):
    election_results = []
    for i in range(n):
        election_results.append(simulateSingleElection(polls, pollster_info, forecast_date, electoralVotes, marginOfError, verbose=False))
    p_Biden = election_results.count('Biden')/n
    p_Trump = election_results.count('Trump')/n
    p_Tie = election_results.count('Tie')/n
    if verbose:
        print(election_results)
    return p_Biden, p_Trump, p_Tie, election_results


# Execute program
def main():
    # create dictionary of FTE grade weights
    grade = ['A+', 'A', 'A-', 'A/B',
             'B+', 'B', 'B-', 'B/C',
             'C+', 'C', 'C-', 'C/D',
             'D+', 'D', np.nan]
    weight = [4.3, 4.0, 3.7, 3.5,
              3.3, 3.0, 2.7, 2.5,
              2.3, 2.0, 1.7, 1.5,
              1.3, 1.0, 0]
    grade_weights = dict(zip(grade,weight))

    # create dataframe of states and their electoral college votes
    state_names = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia',
                   'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
                   'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
                   'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island',
                   'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
    votes = [9, 3, 11, 6, 55, 9, 7, 3, 3, 29,
            16, 4, 4, 20, 11, 6, 6, 8, 8, 4,
            10, 11, 16, 10, 6, 10, 3, 5, 6, 4,
            14, 5, 29, 15, 3, 18, 7, 7, 20, 4,
            9, 3, 11, 38, 6, 3, 13, 12, 5, 10, 3]

    electoralVotes = pd.DataFrame({'state': state_names,
                                   'votes': votes})
    
    # pre-process polls
    processed_polls, pollster_info = preProcessPolls(polls = fte,
                                                     grade_weights = grade_weights)
    # simulate one election in all states + DC
    np.random.seed(100)
    simulateSingleElection(polls = processed_polls,
                           pollster_info = pollster_info,
                           forecast_date = '11/03/20', # election day 2020
                           electoralVotes = electoralVotes,
                           marginOfError = 10,
                           verbose=True)
    
if __name__ == "__main__":
    main()

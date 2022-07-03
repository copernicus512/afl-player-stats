
"""
AFL Machine Learning Project

Created on Mon Jun  7 08:31:14 2021

@author: Robert Carter

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def print_data_types(dataset):
    """Generate a table of data types contained within the dataset.
    Returns: print table of types and counts.

    """

    data_types = {'Categorical': 'object',
                  'Numeric Int': 'int64',
                  'Numeric Float': 'float64',
                  'Boolean': 'bool'}
    
    pad_len = max(len(name) for name in data_types.keys())
    table_hdr = '{:{width}}  |   Nr Features'.format('Data Type', width=pad_len)
    print('\n' + table_hdr)
    print('-' * len(table_hdr))
    for key, value in data_types.items():
        nr_feats = len(dataset.dtypes[dataset.dtypes == value])
        
        print_1 = '{:{width}}  |  '.format(key, width=pad_len)
        print_2 = '{}'.format(nr_feats)
        print(print_1 + print_2)


def calculate_game_score(nr_goals=0, nr_behinds=0):
    """Parameters
    ----------
    nr_goals : TYPE, optional
        DESCRIPTION. The default is 0.
    nr_behinds : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    total_score : TYPE
        DESCRIPTION.

    """
    total_score = nr_behinds + (nr_goals * 6)
    return total_score


# Don't limit screen output for dataframes
pd.options.display.max_columns = None

# Check if current working directory isset to project working directory
cur_dir = os.getcwd()

root_dir = "C:\\Users\\desca\\OneDrive"
project_dir = "AFL Project"
working_dir = f'{root_dir}\\{project_dir}'

if cur_dir != working_dir:
    os.chdir(working_dir)
    print("Working directory changed to " + working_dir)

header_row = 0
import_file = f'{working_dir}\\player_game_stats.csv'
stats_df = pd.read_csv(import_file, header=header_row)

col_names = list(stats_df.columns)

# Remove the first column which contains R row index
stats_df.drop(labels=col_names[0], axis=1, inplace=True)
col_names = list(stats_df.columns)

# Replace some of the original feature names
replace_names = ['round.roundNumber', 'venue.name', 'home.team.club.name',
                 'away.team.club.name', 'player.jumperNumber',
                 'player.player.position', 'clearances.centreClearances',
                 'clearances.stoppageClearances',
                 'clearances.totalClearances', 'player.givenName',
                 'player.surname', 'team.name']
replace_index = [col_names.index(i) for i in replace_names]       
new_names = ['roundNumber', 'venue', 'homeTeam', 'awayTeam', 'playerNumber',
             'playerPosition', 'centreClearances', 'stoppageClearances',
             'totalCLearances', 'firstName', 'lastName', 'teamName']
for list_ind, val in enumerate(replace_index):
    col_names[val] = new_names[list_ind]
stats_df.columns = col_names

# Convert field teamStatus to home (=1) vs away (=0) indicator field
stats_df['teamStatus'] = np.where(stats_df['teamStatus'] == 'home', 1, 0)
stats_df['teamMatchup'] = (stats_df['homeTeam'] + "_vs_" +
                           stats_df['awayTeam'])
stats_df.drop(labels=['homeTeam', 'awayTeam'], axis=1, inplace=True)

# drop any column containing all NAs
stats_df.dropna(axis=1, how='all', inplace=True)

# identify any rows where more than X% of elements are NAs and drop rows
na_limit = 0.3
na_count = stats_df.isna().sum(axis=1)
stats_df.drop(labels=na_count[na_count > na_limit].index,
              axis=0,
              inplace=True)
# should be clean dataset
stats_df.info()

# Summarise feature types
print_data_types(stats_df)

# stats_df.hist(bins=20, figsize=(10, 10))
# plt.show()

# Aggregate player stats to team totals

team_totals = (stats_df
               .groupby(['season', 'roundNumber', 'teamName'])
                .agg({'goals': 'sum', 'behinds': 'sum',
                      'teamMatchup': 'statistics.mode'})
                .rename(columns={'goals': 'Goals', 'behinds': 'Behinds',
                                 'teamMatchup': 'Match'})
                )
team_totals.reset_index(inplace=True)
team_totals['Points'] = calculate_game_score(team_totals['Goals'],
                                              team_totals['Behinds'])

afl_year = 2022
year_stats = stats_df.query('season == @afl_year')

test = (year_stats
        # .loc[:, ['teamName', 'firstName', 'lastName', 'disposals']]
        .groupby(['teamName', 'firstName', 'lastName'])
        .agg(avg_disposals = ('disposals', np.mean),
             avg_goals = ('goals', np.mean),
             avg_behinds = ('behinds', np.mean),
             avg_tackles = ('tackles', np.mean))
        .sort_values(by='avg_disposals', ascending=False)
        )



# plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import pandas as pd
import numpy as np

import capstone_2 as cap 

global main_characters
main_characters = ['JERRY', 'GEORGE', 'ELAINE', 'KRAMER']

global more_characters
more_characters = ['JERRY', 'GEORGE', 'ELAINE', 'KRAMER', 'NEWMAN', 'MORTY', 'HELEN', 'FRANK']

global main_writers
main_writers = ['Larry David', 'Larry David, Jerry Seinfeld', 'Larry Charles', 
                'Peter Mehlman', 'Tom Gammill, Max Pross', 'Gregg Kavet, Andy Robin',
                'Alec Berg, Jeff Schaffer', 'Spike Feresten ', 'Jennifer Crittenden']

def plot_lines_spoken(df_scripts):
    df_scripts = df_scripts.copy()

    fig = plt.figure(figsize=(12,6))

    ax = sns.countplot(x="Character", 
                    data=df_scripts, 
                    order=pd.value_counts(df_scripts['Character']).iloc[:13].index,
                    palette="Set2")

    ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
    ax.set_title("Total Lines Spoken By Main Characters - All Seasons")
    plt.grid(linestyle="--", color=(0.75, 0.75, 0.75, 0.1))

    plt.show()

def create_colormap(cmap_name, chars):
    cmap = mpl.cm.get_cmap(cmap_name)
    
    cmap_dict = {}
    for i in range(len(chars)):
        character = chars[i]
        cmap_dict[character] = cmap(i)

    return cmap_dict

def df_lines_per_season(df_scripts):
    df = df_scripts.copy()
    
    # Group by Season and Character to count total lines per character per season
    df_lines_per_season = df.groupby(['Season', 'Character'], as_index=False).Dialogue.count()
    
    # Sum lines per season to get total values so can calculate fraction
    total_lines_per_season = df_lines_per_season.groupby('Season').Dialogue.sum()
    
    # Join lines per season with total lines
    df_lines_per_season = df_lines_per_season.merge(pd.DataFrame(total_lines_per_season),
                                        left_on='Season',
                                        right_index=True)
    
    df_lines_per_season.columns = ['Season', 'Character', 'Lines',
                             'Lines_in_Season']
    
    # Calcualte % of total lines/season by character
    df_lines_per_season['Percent_of_Total_Lines'] = (df_lines_per_season.Lines /
                                      df_lines_per_season.Lines_in_Season)

    return df_lines_per_season

def plot_character_usage(df_line_counts):
    df = df_line_counts.copy()
    
    fig = plt.figure(figsize=(12,6))
    
    for character in main_characters:

        character_df = df[df['Character'] == character]
        x = character_df['Season']
        y = character_df['Percent_of_Total_Lines']
        
        plt.scatter(x, y, label=character.title(), color=main_char_colors[character], lw=2)
        plt.plot(x, y, label='_nolegend_', color=main_char_colors[character], lw=3)

    plt.ylabel('% of Total Lines')
    plt.xlabel('Season')
    plt.title('Main Character Lines of Dialogue by Season', loc='center')

    plt.grid(linestyle="--", color=(0.75, 0.75, 0.75, 0.1))
    plt.legend()
    plt.show()

def plot_character_usage_more(df_line_counts):
    df = df_line_counts.copy()
    
    fig = plt.figure(figsize=(12,6))
    
    for character in more_characters:
        mpl.rc('image', cmap='Set3')
        
        plt.cmap = "Set3"
        character_df = df[df['Character'] == character]
        x = character_df['Season']
        y = character_df['Percent_of_Total_Lines']
        
        # mpl.rcParams['axes.color_cycle']
        plt.scatter(x, y, label=character.title(), color=more_char_colors[character], lw=2)
        plt.plot(x, y, label='_nolegend_', color=more_char_colors[character], lw=2.5)

    
    plt.ylabel('% of Total Lines')
    plt.xlabel('Season')
    plt.title('Lines Spoken by Main Characters - by Season', loc='center')

    plt.grid(linestyle="--", color=(0.75, 0.75, 0.75, 0.1))
    plt.legend()
    plt.show()

def df_writers_per_season(df_docs_by_ep):
    df = df_docs_by_ep.copy()
    
    # Group by Season and Writers to count total lines per character per season
    df_eps_per_writer = df.groupby(['Season', 'Writers'], as_index=False).Episode.count()
    
    # Sum episodes per season to get total values so can calculate fraction
    total_eps_per_season = df_eps_per_writer.groupby('Season').Episode.sum()
    
    # Join episdoes per season with writers
    df_eps_per_writer = df_eps_per_writer.merge(pd.DataFrame(total_eps_per_season),
                                        left_on='Season',
                                        right_index=True)
    
    df_eps_per_writer.columns = ['Season', 'Writers', 'Episodes_Written',
                             'Episodes_in_Season']
    
    # Calculate % of total episodes/season by writer
    df_eps_per_writer['Percent_of_Total_Episodes'] = (df_eps_per_writer.Episodes_Written /
                                      df_eps_per_writer.Episodes_in_Season)

    return df_eps_per_writer

def plot_writers(df_eps_per_writer):
    df = df_eps_per_writer.copy()
    
    fig = plt.figure(figsize=(12,6))
    
    for writers in main_writers:
        character_df = df[df['Writers'] == writers]
        x = character_df['Season']
        y = character_df['Percent_of_Total_Episodes']
        
        plt.scatter(x, y, label=writers.title())
        plt.plot(x, y, label='_nolegend_', lw=2.5)

    plt.grid(linestyle="--", color=(0.75, 0.75, 0.75, 0.1))
    plt.ylabel('% of Episodes Written')
    plt.xlabel('Season')
    plt.legend()
    plt.title('Primary Episode Writers - by Season', loc='center')
    plt.show()

def plot_total_episodes_writer(df_eps_per_writer):
    df_eps_per_writer = df_eps_per_writer.copy()
    df_eps_per_writer_sum = df_eps_per_writer.groupby(['Writers']).Episodes_Written.sum().reset_index()
    df_eps_per_writer_sum  = df_eps_per_writer_sum.sort_values(by='Episodes_Written', ascending=False)

    fig = plt.figure(figsize=(20,20))
    plt.title("Total Episodes Written", fontsize=22)
    ax = sns.barplot(x='Episodes_Written', y='Writers', data=df_eps_per_writer_sum,ci=None)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.grid(linestyle="--", color=(0.75, 0.75, 0.75, 0.1))

    plt.show()

def df_lines_chars_per_season(df_docs_by_ep):
    df = df_docs_by_ep.copy()
    
    df['Script_Length'] = df.Dialogue.str.len()
    
    # Group by Season to count total lines of Dialogue
    df_lines_per_season = df.groupby(['Season'], as_index=False) \
                         .Lines_of_Dialogue.agg({'Average_Lines': np.mean, 
                                                 'Total_Lines': np.sum, 
                                                 'Number_of_Episodes': len})
    
    df_chars_per_season = df.groupby(['Season'], as_index=False) \
                         .Script_Length.agg({'Average_Chars': np.mean,
                                             'Total_Chars': np.sum})
            
    # Join lines_per_season per season with chars_per_season
    df_lines_chars_per_season = df_lines_per_season.merge(df_chars_per_season,
                                        left_on='Season',
                                        right_on='Season')
        
    return df_lines_chars_per_season

def plot_lines_chars_bar(df_lines_per_season):
    fig = plt.figure(figsize=(12,6))

    ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

    width = 0.45

    df_lines_per_season.Average_Lines.plot(kind='bar', color='orange', 
                                           ax=ax, width=width, position=1, 
                                           label="Average Lines")
    df_lines_per_season.Average_Chars.plot(kind='bar', color='blue', 
                                           ax=ax2, width=width, position=0, 
                                           label="Average Characters")

    plt.title('Average Lines of Dialogue and Characters in Script per Episode - by Season', loc='center')

    ax.set_ylabel('Average Lines of Dialogue per Script', color='orange')
    ax2.set_ylabel('Average Characters per Script', color='b')

    ax.legend(loc='upper left')
    ax.set_ylim(200, 400)
    ax2.legend(loc='upper right')
    plt.xlabel('Season')
    ax.set_xlabel('Season')
    ax2.set_ylim(10000, 20000)
    ax.set_xticklabels(range(1,10),rotation=0)

    plt.grid(linestyle="--", color=(0.75, 0.75, 0.75, 0.1))

    plt.show()
    
if __name__ == "__main__":
    # load in data 
    df_info, df_scripts = cap.load_data()
    df_docs_by_ep = cap.agg_dialogue_by_episode(df_scripts, df_info)

    # create color maps for each character list
    main_char_colors = create_colormap('Set2', main_characters)
    more_char_colors = create_colormap('Set2', more_characters)

    df_line_counts = df_lines_per_season(df_scripts)
    print(df_line_counts.head())

    #plot_character_usage(df_line_counts)
    #plot_character_usage_more(df_line_counts)

    df_eps_per_writer = df_writers_per_season(df_docs_by_ep)
    # plot_writers(df_eps_per_writer)
    # plot_total_episodes_writer(df_eps_per_writer)

    df_lines_chars_per_season = df_lines_chars_per_season(df_docs_by_ep)
    plot_lines_chars_bar(df_lines_chars_per_season)




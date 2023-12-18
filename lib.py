import os
import sqlite3
import pandas as pd
import pickle
import numpy as np

def flatten_list(lst):
    return [x for xs in lst for x in xs]

def flatten_obs(df, max_objs, times):
    new_cols = [f"{c}_{idx}"
                for idx in range(max_objs)
                for c in df.columns[1:]]
    new_cols = ["time"] + new_cols
    obs   = []
    for tm in times:
        cur = df[df["time"] == tm]
        vals = cur.values[:, 1:]
        vals = vals[:max_objs, :]

        if cur.shape[0] < max_objs:
            padding_val = max_objs - vals.shape[0]
            padding = np.zeros((padding_val, vals.shape[1]))
            vals = np.vstack((vals, padding))
    
        # Flatten
        new_vals = np.hstack(vals)

        # Append
        obs.append(new_vals)

    # Combine
    obs   = np.vstack(obs)
    times = np.expand_dims(np.array(times), axis=1)
    obs   = np.hstack((times, obs))
    obs_df = pd.DataFrame(data=obs, columns=new_cols)

    return obs_df

def add_distances(original_df, player_df):
    # Step 1: Filter out Player's data
    player_df_data = player_df[['time', 'pos_x', 'pos_z']]

    # Step 2: Merge with the original DataFrame on 'time'
    merged_df = original_df.merge(player_df_data, on='time', suffixes=('', '_player'))

    # Step 3: Calculate Euclidean distance
    merged_df['distance_from_player_x'] = abs(
        merged_df["pos_x_player"] - merged_df["pos_x"])
    merged_df['distance_from_player_z'] = abs(
        merged_df["pos_z_player"] - merged_df["pos_z"])
    merged_df['distance_from_player'] = np.sqrt(
        (merged_df['pos_x'] - merged_df['pos_x_player'])**2 +
        (merged_df['pos_z'] - merged_df['pos_z_player'])**2)
    merged_df = merged_df.drop(columns=["pos_x_player", "pos_z_player"])

    return merged_df

def swap_columns(df, col1, col2):
    """
    Swap two columns in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame in which to swap columns.
    col1 (str): The name of the first column to swap.
    col2 (str): The name of the second column to swap.
    """
    temp = df[col1].copy()
    df[col1] = df[col2]
    df[col2] = temp
    return df

def find_aa_target(row, auto_attacks_df):
    target_x, target_z = row["pos_x"], row["pos_z"]
    
    # For each champ pos, check against missile positions
    for _, aa in auto_attacks_df.iterrows():
        end_pos_x, end_pos_z = aa["end_pos_x"], aa["end_pos_z"]
        if target_x == end_pos_x and target_z == end_pos_z:
            return True
    return False

def remove_empty(lst):
    return [x for x in lst if x]

def get_names(DB_REPLAYS_DIR):
    if not os.path.exists("NAMES.pkl"):
        NAMES = {
            "CHAMP_NAMES": set(),
            "SUMMONER_NAMES": set(),
            "SPELL_NAMES": set(),
            "MISSILE_NAMES": set(),
            "MINION_NAMES": set(),
            "TURRET_NAMES": set(),
            "MONSTER_NAMES": set(),
            "MISSILE_SPELL_NAMES": set()
        }
        for i, replay_fi in enumerate(os.listdir(DB_REPLAYS_DIR)):
            print(i+1, len(os.listdir(DB_REPLAYS_DIR)))

            replay_path    = os.path.join(DB_REPLAYS_DIR, replay_fi)
            con            = sqlite3.connect(replay_path)

            champ_names    = pd.read_sql("SELECT name FROM champs;", con)
            summoner_names = pd.read_sql("SELECT d_name, f_name FROM champs;", con)
            missile_names = pd.read_sql(
                "SELECT missile_name, spell_name FROM missiles;", con)
            spell_names = pd.read_sql(
                "SELECT q_name, w_name, e_name, r_name FROM champs;", con)
            minion_names   = pd.read_sql(
                "SELECT name FROM minions;", con)
            turret_names   = pd.read_sql(
                "SELECT name FROM turrets;", con)
            monster_names  = pd.read_sql(
                "SELECT name FROM monsters;", con)

            champ_names    = set(remove_empty(champ_names["name"]))
            summoner_names = set(
                remove_empty(summoner_names["d_name"]) + \
                remove_empty(summoner_names["f_name"]))
            missile_spell_names = set(
                remove_empty(missile_names["spell_name"]))
            missile_names  = set(
                remove_empty(missile_names["missile_name"]))
            spell_names    = set(
                remove_empty(spell_names["q_name"]) + \
                remove_empty(spell_names["w_name"]) + \
                remove_empty(spell_names["e_name"]) + \
                remove_empty(spell_names["r_name"]))
            minion_names   = set(remove_empty(minion_names["name"]))
            turret_names   = set(remove_empty(turret_names["name"]))
            monster_names  = set(remove_empty(monster_names["name"]))

            NAMES["CHAMP_NAMES"]         |= champ_names
            NAMES["SUMMONER_NAMES"]      |= summoner_names
            NAMES["SPELL_NAMES"]         |= spell_names
            NAMES["MISSILE_NAMES"]       |= missile_names
            NAMES["MINION_NAMES"]        |= minion_names
            NAMES["TURRET_NAMES"]        |= turret_names
            NAMES["MONSTER_NAMES"]       |= monster_names
            NAMES["MISSILE_SPELL_NAMES"] |= missile_spell_names

        with open("NAMES.pkl", "wb+") as f:
            pickle.dump(NAMES, f)

    else:
        with open("NAMES.pkl", "rb") as f:
            NAMES = pickle.load(f)

    return NAMES
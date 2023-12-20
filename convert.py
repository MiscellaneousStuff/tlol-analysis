"""
Change list:
- Normalisation
- Auto attacks
- Player movement
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import concurrent.futures

from lib import *

DB_REPLAYS_DIR = "/Users/joe/Downloads/DB-2"
NUMPY_REPLAYS_DIR = "/Users/joe/Downloads/NP-2"
DB_REPLAYS = os.listdir(DB_REPLAYS_DIR)

NAMES = get_names(DB_REPLAYS_DIR)

AUTO_ATTACK_TARGETS = ["CHAMPS", "TURRETS", "MINIONS", "MISSILES", "MONSTERS", "OTHER"]
GAME_OBJECT_LIST    = ["champs", "turrets", "minions", "missiles", "monsters"]
MAX_OBJS            = [10, 30, 30, 30, 30]
MAX_WORKERS         = 4
MIN_IDX             = 0
MAX_IDX             = 10000

def dataframe_preprocessing(
        replay_db_path,
        NAMES,
        GAME_OBJECT_LIST):
    
    CHAMP_LIST          = list(NAMES["CHAMP_NAMES"])
    SUMMONER_NAMES      = list(NAMES["SUMMONER_NAMES"])
    SPELL_NAMES         = list(NAMES["SPELL_NAMES"])
    MISSILE_NAMES       = list(NAMES["MISSILE_NAMES"])
    MISSILE_SPELL_NAMES = list(NAMES["MISSILE_SPELL_NAMES"])
    MINION_NAMES        = list(NAMES["MINION_NAMES"])
    TURRET_NAMES        = list(NAMES["TURRET_NAMES"])
    MONSTER_NAMES       = list(NAMES["MONSTER_NAMES"])

    # Load dataframes
    con = sqlite3.connect(replay_db_path)
    df_s = {
        obj:pd.read_sql(f"SELECT * FROM {obj};", con)\
            .drop(labels=["game_id"], axis=1) for obj in GAME_OBJECT_LIST}

    # Clean `missiles_df`
    df_s["missiles"] = df_s["missiles"].drop(labels=["name", "src_idx", "dst_idx"], axis=1)

    # Convert champ names into idx
    df_s["champs"]           = df_s["champs"][df_s["champs"]["name"].astype(bool)]
    df_s["champs"]["name"]   = df_s["champs"]["name"].apply(lambda d: CHAMP_LIST.index(d))
    df_s["champs"]["d_name"] = df_s["champs"]["d_name"].apply(lambda d: SUMMONER_NAMES.index(d))
    df_s["champs"]["f_name"] = df_s["champs"]["f_name"].apply(lambda d: SUMMONER_NAMES.index(d))
    
    # Convert champ spells into idx
    df_s["champs"]["q_name"] = df_s["champs"]["q_name"].apply(lambda d: SPELL_NAMES.index(d))
    df_s["champs"]["w_name"] = df_s["champs"]["w_name"].apply(lambda d: SPELL_NAMES.index(d))
    df_s["champs"]["e_name"] = df_s["champs"]["e_name"].apply(lambda d: SPELL_NAMES.index(d))
    df_s["champs"]["r_name"] = df_s["champs"]["r_name"].apply(lambda d: SPELL_NAMES.index(d))

    # Convert missile missile names and spell names into idx
    df_s["missiles"] = df_s["missiles"][df_s["missiles"]["missile_name"] != '']
    df_s["missiles"]["missile_name"] = df_s["missiles"]["missile_name"].apply(
        lambda d: MISSILE_NAMES.index(d))
    df_s["missiles"]["spell_name"] = df_s["missiles"]["spell_name"].apply(
        lambda d: MISSILE_SPELL_NAMES.index(d))

    # Convert minion and turret names into idx
    df_s["minions"]["name"]  = df_s["minions"]["name"].apply(lambda d: MINION_NAMES.index(d))
    df_s["turrets"]["name"]  = df_s["turrets"]["name"].apply(lambda d: TURRET_NAMES.index(d))
    df_s["monsters"]["name"] = df_s["monsters"]["name"].apply(lambda d: MONSTER_NAMES.index(d))

    # Data cleaning for all game object dataframes
    for obj in GAME_OBJECT_LIST:
        if obj != "missiles":
            df_s[obj] = df_s[obj].drop_duplicates(
                subset=["time", "name"])
        else:
            df_s[obj] = df_s[obj].drop_duplicates(
                subset=["time", "missile_name"])
        df_s[obj] = df_s[obj][
            df_s[obj]["time"] > 15]

    # Data normalisation for `champs_df`
    df_s["champs"].loc[df_s["champs"]['q_cd'] < 0, 'q_cd'] = 0
    df_s["champs"].loc[df_s["champs"]['w_cd'] < 0, 'w_cd'] = 0
    df_s["champs"].loc[df_s["champs"]['e_cd'] < 0, 'e_cd'] = 0
    df_s["champs"].loc[df_s["champs"]['r_cd'] < 0, 'r_cd'] = 0
    df_s["champs"].loc[df_s["champs"]['d_cd'] < 0, 'd_cd'] = 0
    df_s["champs"].loc[df_s["champs"]['f_cd'] < 0, 'f_cd'] = 0

    return df_s, con
    
def build_obs(
        df_s,
        NAMES,
        GAME_OBJECT_LIST,
        champs_scaler,
        missiles_scaler,
        turrets_scaler,
        minions_scaler,
        monsters_scaler,
        MAX_OBJS):
    
    COLUMNS_TO_SCALE = {
        "champs":   ['time', 'hp', 'max_hp', 'mana', 'max_mana', 'armor', 'mr', 'ad',
            'ap', 'level', 'atk_range', 'visible', 'team', 'pos_x', 'pos_z',
            'q_cd', 'w_cd', 'e_cd', 'r_cd', 'd_cd', 'f_cd', 'distance_from_player_x',
            'distance_from_player_z', 'distance_from_player'],
        "other":  ['time', 'hp', 'max_hp', 'mana', 'max_mana', 'armor', 'mr', 'ad',
            'ap', 'level', 'atk_range', 'visible', 'team', 'pos_x', 'pos_z',
            'distance_from_player_x', 'distance_from_player_z',
            'distance_from_player'],
        "missiles":  ['time', 'start_pos_x', 'start_pos_z',
            'end_pos_x', 'end_pos_z', 'pos_x', 'pos_z', 'distance_from_player_x',
            'distance_from_player_z', 'distance_from_player']
    }
    
    CHAMP_LIST          = list(NAMES["CHAMP_NAMES"])

    player_df = df_s["champs"][df_s["champs"]["name"] == CHAMP_LIST.index("Ezreal")]

    # Init `replay_df`
    replay_df = pd.DataFrame()
    times = df_s["champs"].drop_duplicates(subset=['time'])["time"]
    replay_df["time"] = times

    # Add Distance Between Local Player and All Game Objects
    for obj in GAME_OBJECT_LIST:
        df_s[obj] = add_distances(df_s[obj], player_df)

    # Save pre-normalised positions
    champ_pos    = df_s["champs"].drop_duplicates(subset=["time", "pos_x", "pos_z"])
    turrets_pos  = df_s["turrets"].drop_duplicates(subset=["time", "pos_x", "pos_z"])
    monsters_pos = df_s["monsters"].drop_duplicates(subset=["time", "pos_x", "pos_z"])
    minions_pos  = df_s["minions"].drop_duplicates(subset=["time", "pos_x", "pos_z"])

    for obj in GAME_OBJECT_LIST:
        col_list = obj if obj in COLUMNS_TO_SCALE.keys() else "other"
        cols_to_scale = COLUMNS_TO_SCALE[col_list]  # Columns to be scaled

        # Select columns to scale
        df_to_scale = df_s[obj][cols_to_scale]

        # Determine the appropriate scaler and scale the data
        if obj == "champs":
            scaled_data = champs_scaler.fit_transform(df_to_scale)
        elif obj == "missiles":
            scaled_data = missiles_scaler.fit_transform(df_to_scale)
        elif obj == "turrets":
            scaled_data = turrets_scaler.fit_transform(df_to_scale)
        elif obj == "minions":
            scaled_data = minions_scaler.fit_transform(df_to_scale)
        elif obj == "monsters":
            scaled_data = monsters_scaler.fit_transform(df_to_scale)

        # Replace the original columns in df_s[obj] with the scaled data
        df_s[obj][cols_to_scale] = scaled_data
    
    # Flatten each dataframe
    testdf_s = {}
    times = df_s["champs"]["time"].unique()
    for obj, max_objs in zip(GAME_OBJECT_LIST, MAX_OBJS):
        # print("OBJ, MAX:", obj, max_objs)
        testdf_s[obj] = flatten_obs(df_s[obj], max_objs, times)
    
    # Combine all flattened dataframes
    times_unsqueeze = np.expand_dims(times, 1)
    replay_df_vals = [df.iloc[:, 1:] for df in testdf_s.values()]
    replay_df_vals = np.hstack(replay_df_vals)
    replay_df_vals = np.hstack((times_unsqueeze, replay_df_vals))
    replay_df_cols = [list(testdf_s[k].columns[1:].values) for k in testdf_s.keys()]
    replay_df_cols = flatten_list(replay_df_cols)
    replay_df_cols = ["time"] + replay_df_cols
    replay_df = pd.DataFrame(
        data=replay_df_vals,
        columns=replay_df_cols)
    replay_df_save_time = replay_df["time"]

    return \
        replay_df_save_time, \
        player_df, \
        replay_df, \
        champ_pos, \
        turrets_pos, \
        monsters_pos, \
        minions_pos

def build_act(
        replay_df,
        player_df,
        NAMES,
        con,
        df_s,
        replay_df_save_time,
        champ_pos,
        turrets_pos,
        monsters_pos,
        minions_pos):
    
    SUMMONER_NAMES      = list(NAMES["SUMMONER_NAMES"])

    # Temporarily change time to original rather than norm to use as primary id
    replay_df["time"] = player_df["time"].values

    # Movement
    player_cur_pos  = player_df[["time", "pos_x", "pos_z"]]
    player_next_pos = player_df[["time", "pos_x", "pos_z"]].shift(+1)
    player_next_pos = player_next_pos.fillna(0)
    player_x_delta  = player_cur_pos["pos_x"] - player_next_pos["pos_x"]
    player_z_delta  = player_cur_pos["pos_z"] - player_next_pos["pos_z"]
    player_x_delta.iloc[0] = 0
    player_z_delta.iloc[0] = 0
    player_x_delta_digit = (player_x_delta / 100).round().clip(-4, +4)
    player_z_delta_digit = (player_z_delta / 100).round().clip(-4, +4)
    player_df["player_x_delta"]       = player_x_delta
    player_df["player_z_delta"]       = player_z_delta
    player_df["player_delta"]         = np.sqrt(player_x_delta ** 2 + player_z_delta ** 2)
    player_df["player_x_delta_digit"] = player_x_delta_digit
    player_df["player_z_delta_digit"] = player_z_delta_digit
    player_movement_df_final = \
        player_df[["time", "player_x_delta_digit", "player_z_delta_digit"]]
    replay_df = pd.merge(replay_df, player_movement_df_final, on="time")

    # Swap Summoner1 and Summoner2 if Summoner2 := Flash
    d_name = player_df["d_name"]
    if d_name.iloc[0] != SUMMONER_NAMES.index("SummonerFlash"):
        swap_columns(player_df, "d_name", "f_name")
        swap_columns(player_df, "d_cd", "f_cd")
    
    spell_casts = player_df[["time", "q_cd", "w_cd", "e_cd", "r_cd", "d_cd", "f_cd"]]

    # EzrealQ
    ezreal_q_df = pd.read_sql(
        "SELECT time, start_pos_x, start_pos_z, end_pos_x, end_pos_z FROM missiles WHERE spell_name = 'EzrealQ';",
        con)
    ezreal_q_df = ezreal_q_df.drop_duplicates(
        subset=['start_pos_x', 'start_pos_z'])
    ezreal_q_start_pos  = ezreal_q_df[["time", "start_pos_x", "start_pos_z"]]
    ezreal_q_end_pos = ezreal_q_df[["time", "end_pos_x", "end_pos_z"]]
    ezreal_q_end_pos = ezreal_q_end_pos.fillna(0)
    ezreal_q_x_delta  = ezreal_q_end_pos["end_pos_x"] - ezreal_q_start_pos["start_pos_x"]
    ezreal_q_z_delta  = ezreal_q_end_pos["end_pos_z"] - ezreal_q_start_pos["start_pos_z"]
    ezreal_q_x_delta_digit = (ezreal_q_x_delta / 100).round().clip(-4, +4)
    ezreal_q_z_delta_digit = (ezreal_q_z_delta / 100).round().clip(-4, +4)
    ezreal_q_df["ezreal_q_x_delta"] = ezreal_q_x_delta
    ezreal_q_df["ezreal_q_z_delta"] = ezreal_q_z_delta
    ezreal_q_df["ezreal_q_x_delta_digit"] = ezreal_q_x_delta_digit
    ezreal_q_df["ezreal_q_z_delta_digit"] = ezreal_q_z_delta_digit
    ezreal_q_df_final = \
        ezreal_q_df[["time", "ezreal_q_x_delta_digit", "ezreal_q_z_delta_digit"]]
    ezreal_q_df_final["using_q"] = 1
    replay_df = pd.merge(replay_df, ezreal_q_df_final, on="time", how="left")
    replay_df["using_q"] = replay_df["using_q"].fillna(0)
    replay_df = replay_df.fillna(0)

    # EzrealW
    ezreal_w_df = pd.read_sql(
        "SELECT time, start_pos_x, start_pos_z, end_pos_x, end_pos_z FROM missiles WHERE spell_name = 'EzrealW';",
        con)
    ezreal_w_df = ezreal_w_df.drop_duplicates(
        subset=['start_pos_x', 'start_pos_z'])
    ezreal_w_start_pos  = ezreal_w_df[["time", "start_pos_x", "start_pos_z"]]
    ezreal_w_end_pos = ezreal_w_df[["time", "end_pos_x", "end_pos_z"]]
    ezreal_w_end_pos = ezreal_w_end_pos.fillna(0)
    ezreal_w_x_delta  = ezreal_w_end_pos["end_pos_x"] - ezreal_w_start_pos["start_pos_x"]
    ezreal_w_z_delta  = ezreal_w_end_pos["end_pos_z"] - ezreal_w_start_pos["start_pos_z"]
    ezreal_w_x_delta_digit = (ezreal_w_x_delta / 100).round().clip(-4, +4)
    ezreal_w_z_delta_digit = (ezreal_w_z_delta / 100).round().clip(-4, +4)
    ezreal_w_df["ezreal_w_x_delta"] = ezreal_w_x_delta
    ezreal_w_df["ezreal_w_z_delta"] = ezreal_w_z_delta
    ezreal_w_df["ezreal_w_x_delta_digit"] = ezreal_w_x_delta_digit
    ezreal_w_df["ezreal_w_z_delta_digit"] = ezreal_w_z_delta_digit
    ezreal_w_df_final = \
        ezreal_w_df[["time", "ezreal_w_x_delta_digit", "ezreal_w_z_delta_digit"]]
    ezreal_w_df_final["using_w"] = 1
    replay_df = pd.merge(replay_df, ezreal_w_df_final, on="time", how="left")
    replay_df["using_w"] = replay_df["using_w"].fillna(0)
    replay_df = replay_df.fillna(0)

    # EzrealE
    ecd_s_cur  = spell_casts["e_cd"]
    ecd_s_prev = spell_casts["e_cd"].shift(+1).fillna(0)
    ecd_s_diff = ecd_s_cur - ecd_s_prev
    e_cast     = (ecd_s_diff > 20) & (ecd_s_cur > 0)
    e_cast_idx_s = e_cast[e_cast].index
    e_cast_vals = []
    e_cast_cols = ["time", "ezreal_e_x_delta_digit", "ezreal_e_z_delta_digit"]
    for e_cast_idx in e_cast_idx_s:
        e_row = player_df.loc[e_cast_idx-50:e_cast_idx+50][["time", "player_delta"]]
        e_row_idx = e_row.idxmax()["player_delta"]
        tm, x, y = player_df.loc[e_row_idx][["time", "player_x_delta_digit", "player_z_delta_digit"]]
        e_cast_vals.append([tm, x, y])
    ezreal_e_df_final = pd.DataFrame(data=e_cast_vals, columns=e_cast_cols)
    ezreal_e_df_final["using_e"] = 1
    ezreal_e_df_final["using_e"] = ezreal_e_df_final["using_e"].fillna(0)
    ezreal_e_df_final = ezreal_e_df_final.fillna(0)
    replay_df = pd.merge(replay_df, ezreal_e_df_final, on="time", how="left")
    replay_df["using_e"] = replay_df["using_e"].fillna(0)
    replay_df = replay_df.fillna(0)

    # Flash
    dcd_s_cur  = spell_casts["d_cd"]
    dcd_s_prev = spell_casts["d_cd"].shift(+1).fillna(0)
    dcd_s_diff = dcd_s_cur - dcd_s_prev
    d_cast     = (dcd_s_diff > 250) & (dcd_s_cur > 0)
    d_cast_idx_s = d_cast[d_cast].index
    d_cast_vals = []
    d_cast_cols = ["time", "ezreal_d_x_delta_digit", "ezreal_d_z_delta_digit"]
    for d_cast_idx in d_cast_idx_s:
        d_row = player_df.loc[d_cast_idx-50:d_cast_idx+50][["time", "player_delta"]]
        d_row_idx = d_row.idxmax()["player_delta"]
        tm, x, y = player_df.loc[d_row_idx][["time", "player_x_delta_digit", "player_z_delta_digit"]]
        d_cast_vals.append([tm, x, y])
    ezreal_d_df_final = pd.DataFrame(data=d_cast_vals, columns=d_cast_cols)
    ezreal_d_df_final["using_d"] = 1
    replay_df = pd.merge(replay_df, ezreal_d_df_final, on="time", how="left")
    replay_df["using_d"] = replay_df["using_d"].fillna(0)
    replay_df = replay_df.fillna(0)

    # Auto Attack
    AUTO_TARGET_TYPES = ["champ", "turret", "monster", "minion"]
    auto_attacks_df = pd.read_sql("SELECT * FROM missiles WHERE spell_name LIKE 'EzrealBasicAttack%';", con)
    auto_attacks_df = auto_attacks_df.drop_duplicates(subset=["start_pos_x", "start_pos_z"])

    df_s["champs"]["time"]   = replay_df_save_time
    df_s["turrets"]["time"]  = replay_df_save_time
    df_s["monsters"]["time"] = replay_df_save_time
    df_s["minions"]["time"]  = replay_df_save_time
    champ_pos_df    = champ_pos.drop_duplicates(subset=["pos_x", "pos_z"])
    turrets_pos_df  = turrets_pos.drop_duplicates(subset=["pos_x", "pos_z"])
    monsters_pos_df = monsters_pos.drop_duplicates(subset=["pos_x", "pos_z"])
    minions_pos_df  = minions_pos.drop_duplicates(subset=["pos_x", "pos_z"])

    champ_found_aa    = champ_pos_df.apply(lambda r: find_aa_target(r, auto_attacks_df), axis=1)
    turrets_found_aa  = turrets_pos_df.apply(lambda r: find_aa_target(r, auto_attacks_df), axis=1)
    monsters_found_aa = monsters_pos_df.apply(lambda r: find_aa_target(r, auto_attacks_df), axis=1)
    minions_found_aa  = minions_pos_df.apply(lambda r: find_aa_target(r, auto_attacks_df), axis=1)

    # print("champ_pos_df.iloc[:, 0:10]:", champ_pos_df.iloc[:, 0:10])
    print("champ_found_aa.sum(), turrets_found_aa.sum(), monsters_found_aa.sum(), minions_found_aa.sum():", champ_found_aa.sum(), turrets_found_aa.sum(), monsters_found_aa.sum(), minions_found_aa.sum())

    champ_autos    = champ_pos_df.loc[champ_found_aa[champ_found_aa].index][["time", "pos_x", "pos_z"]]
    turrets_autos  = turrets_pos_df.loc[turrets_found_aa[turrets_found_aa].index][["time", "pos_x", "pos_z"]]
    monsters_autos = monsters_pos_df.loc[monsters_found_aa[monsters_found_aa].index][["time", "pos_x", "pos_z"]]
    minions_autos  = minions_pos_df.loc[minions_found_aa[minions_found_aa].index][["time", "pos_x", "pos_z"]]

    all_autos_vals = pd.concat([
        champ_autos.apply(lambda row: get_target_idx(
            row, champ_pos_df, AUTO_TARGET_TYPES.index("champ")), axis=1), #[["time"]],
        turrets_autos.apply(lambda row: get_target_idx(
            row, turrets_pos_df, AUTO_TARGET_TYPES.index("turret")), axis=1), #[["time"]],
        monsters_autos.apply(lambda row: get_target_idx(
            row, monsters_pos_df, AUTO_TARGET_TYPES.index("monster")), axis=1), #[["time"]],
        minions_autos.apply(lambda row: get_target_idx(
            row, minions_pos_df, AUTO_TARGET_TYPES.index("minion")), axis=1) # [["time"]]
    ])

    all_autos_cols = ["time", "target_idx", "target_type"]
    all_autos_df   = pd.DataFrame(data=all_autos_vals, columns=all_autos_cols)
    all_autos_df["using_auto"] = 1
    replay_df = pd.merge(replay_df, all_autos_df, on="time", how="left")
    replay_df["using_auto"] = replay_df["using_auto"].fillna(0)
    replay_df = replay_df.fillna(0)

    # Set time back from original to normalised
    replay_df["time"] = replay_df_save_time
    return replay_df

def convert_db_to_np(replay_db_path, NAMES, GAME_OBJECT_LIST, MAX_OBJS, out_path):
    # Create the scaler instance
    champs_scaler   = MinMaxScaler(feature_range=(0, 1))
    missiles_scaler = MinMaxScaler(feature_range=(0, 1))
    turrets_scaler  = MinMaxScaler(feature_range=(0, 1))
    minions_scaler  = MinMaxScaler(feature_range=(0, 1))
    monsters_scaler = MinMaxScaler(feature_range=(0, 1))

    # Preprocess dataframes
    df_s, con = dataframe_preprocessing(replay_db_path, NAMES, GAME_OBJECT_LIST)

    # Get observation
    replay_df_save_time, \
    player_df, \
    replay_df, \
    champ_pos, \
    turrets_pos, \
    monsters_pos, \
    minions_pos = build_obs(
        df_s,
        NAMES,
        GAME_OBJECT_LIST,
        champs_scaler,
        missiles_scaler,
        turrets_scaler,
        minions_scaler,
        monsters_scaler,
        MAX_OBJS)
    
    # Get actions
    replay_df = build_act(
        replay_df,
        player_df,
        NAMES,
        con,
        df_s,
        replay_df_save_time,
        champ_pos,
        turrets_pos,
        monsters_pos,
        minions_pos)

    # Save pandas and float16 `replay_df`
    replay_df = replay_df.astype(np.float16)
    # replay_df.to_csv(f"{out_path}.csv")
    np.save(f"{out_path}", replay_df)

def go(replay_db_path):
    s = time.time()
    full_path = os.path.join(DB_REPLAYS_DIR, replay_db_path)
    out_path  = os.path.join(NUMPY_REPLAYS_DIR, replay_db_path.replace(".db", ""))

    convert_db_to_np(
        full_path,
        NAMES,
        GAME_OBJECT_LIST,
        MAX_OBJS,
        out_path)
    
    e = time.time() - s
    print("Converting replay tm:", e)

if __name__ == "__main__":
    REPLAY_LIST = DB_REPLAYS[MIN_IDX:MAX_IDX]
    REPLAY_LIST = [fi.replace(".db", "") for fi in REPLAY_LIST]
    EXISTING = os.listdir(NUMPY_REPLAYS_DIR)
    EXISTING = [fi.replace(".npy", "") for fi in EXISTING]

    REMAINING = list(set(REPLAY_LIST) - set(EXISTING))
    REMAINING = [f"{fi}.db" for fi in REMAINING]

    print(len(REPLAY_LIST), len(EXISTING), len(REMAINING))
    
    i = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_summoner_name = (executor.submit(
            go,
            replay_db_path
        ) for replay_db_path in REMAINING)
        for future in concurrent.futures.as_completed(future_to_summoner_name):
            try:
                data = future.result()
            except Exception as exc:
                data = str(type(exc))
                print("ERR:", data)
            finally:
                print(f"Replay: {i+1}/{len(REMAINING)}")
                i += 1
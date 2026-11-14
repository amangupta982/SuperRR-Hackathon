"""
SupeRR Selector Season 2 - Rajasthan Royals Analytical Framework
Identify the Most Reliable Players Under Pressure
Author: Quantitative Cricket Analytics System
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
os.environ['MPLCONFIGDIR'] = '/tmp/mpl_config'
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print(">>> 1. Loading datasets...")
    matches = pd.read_csv('matches.csv')
    comps = pd.read_csv('competitions.csv')
    bbb = pd.read_csv('ball_by_ball.csv')
    teams = pd.read_csv('teams.csv')

    print(f"Loaded {len(matches)} matches, {len(bbb)} ball-by-ball deliveries.")

    # Team mapping
    top_tier_teams = set([
        'Chennai Super Kings', 'Mumbai Indians', 'Rajasthan Royals', 'Punjab Kings',
        'Royal Challengers Bengaluru', 'Delhi Capitals', 'Kolkata Knight Riders',
        'Sunrisers Hyderabad', 'Gujarat Titans', 'Lucknow Super Giants',
        'India', 'Australia', 'England', 'South Africa', 'New Zealand', 'Pakistan',
        'West Indies', 'Sri Lanka', 'Afghanistan'
    ])

    team_id_to_name = teams.drop_duplicates(subset=['team_id']).set_index('team_id')['team_name'].to_dict()
    bbb['bowling_team'] = bbb['bowling_team_id'].map(team_id_to_name)
    bbb['batting_team'] = bbb['batting_team_id'].map(team_id_to_name)

    matches = matches.merge(comps[['comp_id', 'match_format', 'comp_name']], on='comp_id', how='left')

    bbb['total_ball_runs'] = bbb['runs'] + bbb['extras1_runs'] + bbb['extras_2_runs']
    bbb['is_legal'] = ~bbb['extras_type'].isin(['wide', 'Overstep'])
    bbb['is_bowler_wicket'] = bbb['is_wicket'] & bbb['dismissal_type'].isin(['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket'])
    bbb['bowler_runs_conceded'] = bbb['runs'] + np.where(bbb['extras_type'].isin(['wide', 'Overstep']), bbb['extras1_runs'] + bbb['extras_2_runs'], 0)
    bbb['is_batter_ball'] = bbb['extras_type'] != 'wide'

    # Compute 1st innings total
    inn_totals = bbb.groupby(['match_id', 'innings'])['total_ball_runs'].sum().reset_index()
    inn1 = inn_totals[inn_totals['innings'] == 1].rename(columns={'total_ball_runs': 'inn1_runs'})

    matches = matches.merge(inn1[['match_id', 'inn1_runs']], on='match_id', how='left')
    matches['target'] = matches['revised_target'].combine_first(matches['inn1_runs'] + 1)
    matches['total_balls_inn1'] = matches['first_inn_revised_overs'].fillna(120.0)
    matches['total_balls_inn2'] = matches['second_inn_revised_overs'].fillna(120.0)

    bbb = bbb.merge(matches[['match_id', 'match_format', 'comp_name', 'knock_out', 'stage', 'target', 'total_balls_inn1', 'total_balls_inn2']], on='match_id', how='left')
    bbb = bbb.sort_values(['match_id', 'innings', 'innings_ball_num']).reset_index(drop=True)

    print(">>> 2. Engineering ball-by-ball running state...")
    bbb['cum_runs'] = bbb.groupby(['match_id', 'innings'])['total_ball_runs'].cumsum()
    bbb['runs_before'] = bbb.groupby(['match_id', 'innings'])['cum_runs'].shift(1).fillna(0)

    bbb['cum_wickets'] = bbb.groupby(['match_id', 'innings'])['is_wicket'].cumsum()
    bbb['wickets_before'] = bbb.groupby(['match_id', 'innings'])['cum_wickets'].shift(1).fillna(0)

    bbb['cum_legal'] = bbb.groupby(['match_id', 'innings'])['is_legal'].cumsum()
    bbb['legal_before'] = bbb.groupby(['match_id', 'innings'])['cum_legal'].shift(1).fillna(0)

    bbb['total_scheduled_balls'] = np.where(bbb['innings'] == 1, bbb['total_balls_inn1'], bbb['total_balls_inn2'])
    bbb['balls_remaining'] = np.maximum(0, bbb['total_scheduled_balls'] - bbb['legal_before'])
    bbb['wickets_remaining'] = np.maximum(0, 10 - bbb['wickets_before'])

    bbb['runs_needed'] = np.maximum(0, bbb['target'] - bbb['runs_before'])
    bbb['rrr'] = np.where((bbb['innings'] == 2) & (bbb['balls_remaining'] > 0),
                          bbb['runs_needed'] / (bbb['balls_remaining'] / 6.0), np.nan)

    print(">>> 3. Computing continuous Pressure Index (PI)...")
    def compute_pressure(row):
        inn = row['innings']
        if inn >= 3:
            return 1.0  # Super Over
        ko_mult = 1.15 if (row['knock_out'] == 1 or row['stage'] in ['Final', 'Semi Final', 'Qualifier', 'Eliminator']) else 1.0
        
        if inn == 2:
            rrr = row['rrr']
            rem_balls = row['balls_remaining']
            rem_wkts = row['wickets_remaining']
            runs_nd = row['runs_needed']
            if runs_nd <= 0:
                return 0.10
            if rem_balls <= 0:
                return 1.0
                
            rrr_clipped = min(max(rrr, 2.0), 24.0)
            s_rrr = 1.0 / (1.0 + np.exp(-0.5 * (rrr_clipped - 9.5)))
            s_wkt = (10 - rem_wkts) / 10.0
            balls_fraction = 1.0 - (rem_balls / 120.0)
            
            alive_factor = 1.0
            if rrr > 22.0 and rem_wkts <= 2:
                alive_factor = 0.5
            elif rrr < 4.0 and rem_wkts >= 6:
                alive_factor = 0.2
                
            p = (0.50 * s_rrr + 0.25 * s_wkt + 0.25 * balls_fraction) * alive_factor
            p = min(max(p, 0.05), 1.0)
            return min(1.0, p * ko_mult)
        else:
            over = row['overid']
            legal_b = row['legal_before']
            wkts_lost = row['wickets_before']
            expected_wkts = 7.0 * (legal_b / 120.0)**0.7
            crisis = max(0.0, (wkts_lost - expected_wkts) / 4.0)
            if over >= 16:
                phase_p = 0.65 + 0.07 * (over - 16)
            elif over <= 6:
                phase_p = 0.25
            else:
                phase_p = 0.35
            p = max(phase_p, 0.30 + 0.50 * crisis)
            return min(1.0, p * ko_mult)

    bbb['PI'] = bbb.apply(compute_pressure, axis=1)
    bbb['is_high_pressure'] = bbb['PI'] >= 0.60

    bbb['opp_weight_bat'] = bbb['bowling_team'].apply(lambda t: 1.0 if t in top_tier_teams else 0.85)
    bbb['opp_weight_bowl'] = bbb['batting_team'].apply(lambda t: 1.0 if t in top_tier_teams else 0.85)

    print(f"High pressure deliveries: {(bbb['is_high_pressure']).sum()} ({(bbb['is_high_pressure']).mean()*100:.1f}%)")

    # ==================== BATTING MODEL ====================
    print(">>> 4. Modeling Batting Performance & Clutch Index (Hazard-Preserved)...")
    bat_hp = bbb[bbb['is_high_pressure'] & bbb['is_batter_ball']].groupby(['striker_id', 'striker_name']).agg(
        hp_runs=('runs', 'sum'),
        hp_balls=('runs', 'count'),
        hp_dismissals=('is_wicket', 'sum'),
        hp_fours=('is_four', 'sum'),
        hp_sixes=('is_six', 'sum'),
        hp_dots=('runs', lambda x: (x == 0).sum()),
        hp_matches=('match_id', 'nunique'),
        avg_opp_weight=('opp_weight_bat', 'mean')
    ).reset_index()

    bat_base = bbb[(~bbb['is_high_pressure']) & bbb['is_batter_ball']].groupby(['striker_id', 'striker_name']).agg(
        base_runs=('runs', 'sum'),
        base_balls=('runs', 'count'),
        base_dismissals=('is_wicket', 'sum'),
        base_fours=('is_four', 'sum'),
        base_sixes=('is_six', 'sum'),
        base_dots=('runs', lambda x: (x == 0).sum())
    ).reset_index()

    bat = bat_hp.merge(bat_base, on=['striker_id', 'striker_name'], how='inner')
    bat['hp_sr'] = bat['hp_runs'] / bat['hp_balls'] * 100
    bat['base_sr'] = bat['base_runs'] / bat['base_balls'] * 100
    bat['sr_lift'] = bat['hp_sr'] - bat['base_sr']
    bat['hp_avg'] = bat['hp_runs'] / np.maximum(1, bat['hp_dismissals'])
    bat['base_avg'] = bat['base_runs'] / np.maximum(1, bat['base_dismissals'])

    # Hazard preservation: in HP, league average drops by 32.6% (ratio = 0.674)
    # Measures how well player maintains average relative to expected drop
    bat['avg_preservation'] = bat['hp_avg'] - (bat['base_avg'] * 0.674)
    bat['avg_lift'] = bat['hp_avg'] - bat['base_avg']
    bat['hp_boundary_pct'] = (bat['hp_fours'] + bat['hp_sixes']) / bat['hp_balls'] * 100
    bat['hp_dot_pct'] = bat['hp_dots'] / bat['hp_balls'] * 100

    # Consistency
    bat_m = bbb[bbb['is_high_pressure'] & bbb['is_batter_ball']].groupby(['striker_id', 'striker_name', 'match_id']).agg(
        m_hp_runs=('runs', 'sum'),
        m_hp_balls=('runs', 'count'),
        m_hp_out=('is_wicket', 'sum')
    ).reset_index()
    bat_m['m_hp_sr'] = bat_m['m_hp_runs'] / bat_m['m_hp_balls'] * 100
    bat_m['is_success'] = (bat_m['m_hp_balls'] >= 6) & ((bat_m['m_hp_sr'] >= 150.0) | (bat_m['m_hp_runs'] >= 25))
    bat_c = bat_m[bat_m['m_hp_balls'] >= 6].groupby(['striker_id', 'striker_name']).agg(
        qual_hp_matches=('is_success', 'count'),
        successful_hp_matches=('is_success', 'sum')
    ).reset_index()
    bat_c['consistency_rate'] = bat_c['successful_hp_matches'] / bat_c['qual_hp_matches']
    bat = bat.merge(bat_c[['striker_id', 'striker_name', 'qual_hp_matches', 'consistency_rate']], on=['striker_id', 'striker_name'], how='left')
    bat['qual_hp_matches'] = bat['qual_hp_matches'].fillna(0)
    bat['consistency_rate'] = bat['consistency_rate'].fillna(0.40)

    # Eligibility: min 180 HP balls faced and >= 10 qual HP matches
    bat_el = bat[(bat['hp_balls'] >= 180) & (bat['qual_hp_matches'] >= 10)].copy()

    # Batting Z-scores
    z_hp_sr = stats.zscore(bat_el['hp_sr'])
    z_hp_avg = stats.zscore(bat_el['hp_avg'])
    z_hp_bnd = stats.zscore(bat_el['hp_boundary_pct'])
    z_hp_dot = stats.zscore(-bat_el['hp_dot_pct'])
    score_abs_bat = 0.40 * z_hp_sr + 0.30 * z_hp_avg + 0.15 * z_hp_bnd + 0.15 * z_hp_dot

    z_sr_lift = stats.zscore(bat_el['sr_lift'])
    z_avg_pres = stats.zscore(bat_el['avg_preservation'])
    score_lift_bat = 0.65 * z_sr_lift + 0.35 * z_avg_pres

    z_cons_bat = stats.zscore(bat_el['consistency_rate'])
    vol_weight_bat = np.log1p(bat_el['hp_balls']) / np.log1p(bat_el['hp_balls'].max())
    z_vol_bat = stats.zscore(vol_weight_bat * bat_el['avg_opp_weight'])

    bat_el['raw_cbi'] = 0.35 * score_abs_bat + 0.30 * score_lift_bat + 0.20 * z_cons_bat + 0.15 * z_vol_bat
    bat_el['CBI_score'] = 50 + 15 * bat_el['raw_cbi']
    bat_el = bat_el.sort_values('CBI_score', ascending=False).reset_index(drop=True)
    bat_el['rank'] = bat_el.index + 1

    top10_batters = bat_el.head(10).copy()
    top10_batters.to_csv('top_10_batters.csv', index=False)
    bat_el.to_csv('all_batters_metrics.csv', index=False)

    # ==================== BOWLING MODEL ====================
    print(">>> 5. Modeling Bowling Performance & Clutch Index (Volume-Regularized)...")
    bowl_hp = bbb[bbb['is_high_pressure']].groupby(['bowler_id', 'bowler_name']).agg(
        hp_legal=('is_legal', 'sum'),
        hp_runs_conceded=('bowler_runs_conceded', 'sum'),
        hp_wickets=('is_bowler_wicket', 'sum'),
        hp_dots=('bowler_runs_conceded', lambda x: (x == 0).sum()),
        hp_boundaries=('is_four', lambda x: (x | bbb.loc[x.index, 'is_six']).sum()),
        hp_matches=('match_id', 'nunique'),
        avg_opp_weight=('opp_weight_bowl', 'mean')
    ).reset_index()

    bowl_base = bbb[~bbb['is_high_pressure']].groupby(['bowler_id', 'bowler_name']).agg(
        base_legal=('is_legal', 'sum'),
        base_runs_conceded=('bowler_runs_conceded', 'sum'),
        base_wickets=('is_bowler_wicket', 'sum'),
        base_dots=('bowler_runs_conceded', lambda x: (x == 0).sum()),
        base_matches=('match_id', 'nunique')
    ).reset_index()

    bowl = bowl_hp.merge(bowl_base, on=['bowler_id', 'bowler_name'], how='inner')
    bowl['hp_econ'] = bowl['hp_runs_conceded'] / (bowl['hp_legal'] / 6.0)
    bowl['base_econ'] = bowl['base_runs_conceded'] / (bowl['base_legal'] / 6.0)
    bowl['econ_resistance'] = bowl['base_econ'] - bowl['hp_econ']
    bowl['hp_sr'] = bowl['hp_legal'] / np.maximum(1, bowl['hp_wickets'])
    bowl['base_sr'] = bowl['base_legal'] / np.maximum(1, bowl['base_wickets'])
    bowl['wkt_rate_hp'] = bowl['hp_wickets'] / bowl['hp_legal']
    bowl['wkt_rate_base'] = bowl['base_wickets'] / bowl['base_legal']
    bowl['wkt_rate_lift'] = bowl['wkt_rate_hp'] - bowl['wkt_rate_base']
    bowl['hp_dot_pct'] = bowl['hp_dots'] / bowl['hp_legal'] * 100
    bowl['hp_bnd_pct'] = bowl['hp_boundaries'] / bowl['hp_legal'] * 100

    bowl_m = bbb[bbb['is_high_pressure']].groupby(['bowler_id', 'bowler_name', 'match_id']).agg(
        m_hp_legal=('is_legal', 'sum'),
        m_hp_runs=('bowler_runs_conceded', 'sum'),
        m_hp_wkts=('is_bowler_wicket', 'sum')
    ).reset_index()
    bowl_m['m_hp_econ'] = bowl_m['m_hp_runs'] / (bowl_m['m_hp_legal'] / 6.0)
    bowl_m['is_success'] = (bowl_m['m_hp_legal'] >= 6) & ((bowl_m['m_hp_econ'] <= 8.5) | (bowl_m['m_hp_wkts'] >= 2))
    bowl_c = bowl_m[bowl_m['m_hp_legal'] >= 6].groupby(['bowler_id', 'bowler_name']).agg(
        qual_hp_matches=('is_success', 'count'),
        successful_hp_matches=('is_success', 'sum')
    ).reset_index()
    bowl_c['consistency_rate'] = bowl_c['successful_hp_matches'] / bowl_c['qual_hp_matches']
    bowl = bowl.merge(bowl_c[['bowler_id', 'bowler_name', 'qual_hp_matches', 'consistency_rate']], on=['bowler_id', 'bowler_name'], how='left')
    bowl['qual_hp_matches'] = bowl['qual_hp_matches'].fillna(0)
    bowl['consistency_rate'] = bowl['consistency_rate'].fillna(0.40)

    bowl_el = bowl[(bowl['hp_legal'] >= 200) & (bowl['qual_hp_matches'] >= 12)].copy()

    z_hp_econ = stats.zscore(-bowl_el['hp_econ'])
    z_hp_bsr = stats.zscore(-bowl_el['hp_sr'])
    z_hp_dot_b = stats.zscore(bowl_el['hp_dot_pct'])
    z_hp_bnd_b = stats.zscore(-bowl_el['hp_bnd_pct'])
    score_abs_bowl = 0.40 * z_hp_econ + 0.30 * z_hp_bsr + 0.15 * z_hp_dot_b + 0.15 * z_hp_bnd_b

    z_econ_res = stats.zscore(bowl_el['econ_resistance'])
    z_wkt_lift = stats.zscore(bowl_el['wkt_rate_lift'])
    score_lift_bowl = 0.60 * z_econ_res + 0.40 * z_wkt_lift

    z_cons_bowl = stats.zscore(bowl_el['consistency_rate'])
    vol_weight_bowl = np.log1p(bowl_el['hp_legal']) / np.log1p(bowl_el['hp_legal'].max())
    z_vol_bowl = stats.zscore(vol_weight_bowl * bowl_el['avg_opp_weight'])

    bowl_el['raw_cbi'] = 0.35 * score_abs_bowl + 0.30 * score_lift_bowl + 0.20 * z_cons_bowl + 0.15 * z_vol_bowl
    bowl_el['CBI_score'] = 50 + 15 * bowl_el['raw_cbi']
    bowl_el = bowl_el.sort_values('CBI_score', ascending=False).reset_index(drop=True)
    bowl_el['rank'] = bowl_el.index + 1

    top10_bowlers = bowl_el.head(10).copy()
    top10_bowlers.to_csv('top_10_bowlers.csv', index=False)
    bowl_el.to_csv('all_bowlers_metrics.csv', index=False)

    print(">>> 6. Generating Visualizations...")
    sns.set_theme(style="whitegrid")

    # FIG 1: Pressure Index Distribution
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.histplot(bbb['PI'], bins=35, kde=True, color='#002D62', ax=ax, alpha=0.6)
    ax.axvline(0.60, color='#E6007E', linestyle='--', linewidth=2.5, label='High-Pressure Cutoff (PI >= 0.60)')
    ax.set_title('Distribution of Situational Pressure Index (PI) Across All Deliveries', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Pressure Index (0 = Cruise / Low Stakes, 1.0 = Max Criticality)', fontsize=12)
    ax.set_ylabel('Delivery Count', fontsize=12)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    plt.tight_layout()
    plt.savefig('fig1_pressure_distribution.png')
    plt.close()

    # FIG 2: Batting Clutch Matrix (HP Strike Rate vs SR Lift)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    scatter = ax.scatter(bat_el['sr_lift'], bat_el['hp_sr'], s=bat_el['hp_balls']/2, c=bat_el['CBI_score'], cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Clutch Batting Index (CBI Score)', fontsize=11)
    
    # Label Top 10 Batters
    for _, row in top10_batters.iterrows():
        ax.annotate(row['striker_name'].replace('Namboori ', ''), (row['sr_lift'], row['hp_sr']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9.5, fontweight='bold',
                    color='#002D62', bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6, ec="none"))

    ax.axvline(0, color='red', linestyle=':', alpha=0.7)
    ax.axhline(155.88, color='gray', linestyle='--', alpha=0.7, label='League HP Average SR (155.9)')
    ax.set_title('Batting Clutch Performance: High-Pressure Strike Rate vs SR Lift (vs Baseline)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Strike Rate Lift under Pressure (HP SR - Baseline SR)', fontsize=12)
    ax.set_ylabel('High-Pressure Strike Rate (SR)', fontsize=12)
    ax.legend(loc='lower left', frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig('fig2_batting_clutch_matrix.png')
    plt.close()

    # FIG 3: Bowling Clutch Matrix (HP Economy vs Economy Resistance)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    scatter = ax.scatter(bowl_el['econ_resistance'], bowl_el['hp_econ'], s=bowl_el['hp_legal']/2, c=bowl_el['CBI_score'], cmap='magma', alpha=0.8, edgecolors='black', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Clutch Bowling Index (CBI Score)', fontsize=11)
    
    # Label Top 10 Bowlers
    for _, row in top10_bowlers.iterrows():
        ax.annotate(row['bowler_name'], (row['econ_resistance'], row['hp_econ']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9.5, fontweight='bold',
                    color='#002D62', bbox=dict(boxstyle="round,pad=0.2", fc="cyan", alpha=0.6, ec="none"))

    ax.axvline(0, color='red', linestyle=':', alpha=0.7, label='Zero Inflation (Resistant)')
    ax.axhline(9.35, color='gray', linestyle='--', alpha=0.7, label='League HP Average Economy (9.35)')
    ax.set_title('Bowling Clutch Performance: High-Pressure Economy vs Economy Resistance (vs Baseline)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Economy Resistance (Baseline Economy - HP Economy, Higher = Better)', fontsize=12)
    ax.set_ylabel('High-Pressure Economy Rate (Lower = Better)', fontsize=12)
    ax.invert_yaxis()
    ax.legend(loc='lower left', frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig('fig3_bowling_clutch_matrix.png')
    plt.close()

    # FIG 4: Top 10 Summary Infographic
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    
    bat_sorted = top10_batters.sort_values('CBI_score', ascending=True)
    bars1 = ax1.barh(bat_sorted['striker_name'].str.replace('Namboori ', ''), bat_sorted['CBI_score'], color='#E6007E', alpha=0.85, edgecolor='#002D62')
    ax1.set_xlim(55, 75)
    ax1.set_title('Top 10 Batters Under Pressure (CBI Score)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Clutch Batting Index Score')
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:.1f}', va='center', ha='left', fontsize=9, fontweight='bold')

    bowl_sorted = top10_bowlers.sort_values('CBI_score', ascending=True)
    bars2 = ax2.barh(bowl_sorted['bowler_name'], bowl_sorted['CBI_score'], color='#002D62', alpha=0.85, edgecolor='#E6007E')
    ax2.set_xlim(55, 85)
    ax2.set_title('Top 10 Bowlers Under Pressure (CBI Score)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Clutch Bowling Index Score')
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:.1f}', va='center', ha='left', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('fig4_top10_rankings_summary.png')
    plt.close()

    print(">>> Finished! Generated top_10_batters.csv, top_10_bowlers.csv, and figures.")

if __name__ == '__main__':
    main()

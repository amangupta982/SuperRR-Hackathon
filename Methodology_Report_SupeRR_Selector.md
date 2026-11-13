# SupeRR Selector Season 2: Identifying the Most Reliable Players Under Pressure
**A Quantitative Framework for Rajasthan Royals**
*Author: Cricket Analytics & Machine Learning Operations*

---

## 1. Executive Summary & Final Top 10 Rankings

T20 cricket is often decided in a handful of high-leverage moments. In these moments—a mounting required run rate in a chase, an early top-order collapse, or the execution demands of overs 16–20—ordinary players succumb to risk aversion or reckless execution, while elite performers thrive.

To separate true 'pressure performers' from reputation-based perceptions, we built the **Rajasthan Royals Situational Pressure & Clutch Index (CBI)**. Our framework reconstructs every delivery across **883 matches (200,999 balls)** from 2022 to 2026 across the IPL and T20 Internationals, quantifying ball-by-ball match pressure on a continuous [0, 1] scale and comparing high-pressure performance (PI >= 0.60) directly against each player's individual baseline.

### 01 Player Rankings — Unstop Submission Dropdown Mappings

#### Top 10 Batters Under Pressure
| Rank | Player Name (Dropdown Name) | Player ID | CBI Score | HP Balls | HP Runs | HP SR | Base SR | SR Lift (Δ) | HP Avg | Consistency Rate |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Namboori Tilak Varma** | 7897 | **70.10** | 637 | 1230 | 193.09 | 125.99 | **+67.11** | 42.41 | 80.0% |
| **2** | **Suryakumar Yadav** | 1699 | **69.59** | 492 | 1030 | 209.35 | 157.12 | **+52.23** | 38.15 | 83.3% |
| **3** | **Rishabh Pant** | 4703 | **67.55** | 292 | 572 | 195.89 | 126.79 | **+69.10** | 30.11 | 94.4% |
| **4** | **Tim David** | 6396 | **66.69** | 594 | 1188 | 200.00 | 156.65 | **+43.35** | 34.94 | 87.2% |
| **5** | **Shashank Singh** | 2756 | **65.47** | 303 | 572 | 188.78 | 129.57 | **+59.21** | 44.00 | 70.0% |
| **6** | **Tristan Stubbs** | 20702 | **65.44** | 488 | 912 | 186.89 | 119.53 | **+67.36** | 38.00 | 72.4% |
| **7** | **Naman Dhir** | 37052 | **65.38** | 215 | 428 | 199.07 | 150.80 | **+48.27** | 30.57 | 93.8% |
| **8** | **Dhruv Jurel** | 7260 | **64.88** | 279 | 530 | 189.96 | 130.44 | **+59.52** | 35.33 | 81.2% |
| **9** | **Sikandar Raza** | 2984 | **60.66** | 234 | 438 | 187.18 | 119.79 | **+67.39** | 27.38 | 81.2% |
| **10** | **Heinrich Klaasen** | 4260 | **59.97** | 468 | 905 | 193.38 | 153.64 | **+39.74** | 30.17 | 81.8% |

#### Top 10 Bowlers Under Pressure
| Rank | Player Name (Dropdown Name) | Player ID | CBI Score | HP Legal Balls | HP Wickets | HP Economy | Base Economy | Econ Resistance (Δ) | HP Strike Rate | Consistency Rate |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Sunil Narine** | 781 | **81.78** | 299 | 26 | **6.28** | 7.01 | **+0.73** | 11.50 | 83.3% |
| **2** | **Jasprit Bumrah** | 2933 | **73.64** | 554 | 43 | **7.18** | 7.30 | **+0.12** | 12.88 | 74.1% |
| **3** | **Shaheen Afridi** | 6355 | **68.52** | 254 | 26 | **8.01** | 7.99 | **-0.01** | 9.77 | 69.0% |
| **4** | **Matheesha Pathirana** | 8661 | **67.45** | 529 | 49 | **8.89** | 8.74 | **-0.15** | 10.80 | 69.4% |
| **5** | **Mitchell Santner** | 4101 | **65.43** | 322 | 23 | **7.77** | 7.37 | **-0.40** | 14.00 | 71.4% |
| **6** | **Rasikh Salam** | 7327 | **64.59** | 217 | 26 | **9.54** | 9.60 | **+0.06** | 8.35 | 61.1% |
| **7** | **Haris Rauf** | 7364 | **64.37** | 343 | 36 | **8.83** | 8.59 | **-0.25** | 9.53 | 58.8% |
| **8** | **Kuldeep Yadav** | 3644 | **63.45** | 407 | 36 | **8.64** | 7.85 | **-0.79** | 11.31 | 66.7% |
| **9** | **Rashid Khan** | 4722 | **62.18** | 707 | 62 | **8.62** | 7.02 | **-1.60** | 11.40 | 63.5% |
| **10** | **Adam Zampa** | 2636 | **62.12** | 320 | 34 | **8.78** | 8.22 | **-0.55** | 9.41 | 52.8% |

---

## 2. Defining Match Pressure: Mathematical Formulation

Pressure in T20 cricket is the sensitivity of the match outcome to the execution of the current delivery. A ball bowled in over 2 with the required rate at 6.0 involves minimal pressure; a ball in over 18 requiring 12.5 runs per over with 5 wickets down is a critical junction where every single run or wicket swings the win probability drastically.

We construct a delivery-level **Pressure Index (PI)** bounded strictly in [0, 1]:

### 2.1 Innings 2: The Chase Dynamics
In a run chase, the win conditions are explicit. Pressure is formulated as a tri-factor function:
1. **Required Run Rate Stress (S_RRR)**:
   Baseline scoring in the dataset is 8.2 runs/over. Stress is modeled via a logistic sigmoid centered at RRR = 9.5:
   S_RRR = 1 / (1 + exp(-0.5 * (RRR - 9.5)))
   Where RRR <= 6.0 gives S_RRR ~ 0.14; RRR = 12.0 gives S_RRR ~ 0.78; RRR = 16.0 gives S_RRR ~ 0.96.
2. **Resource Hazard (S_W)**:
   S_W = Wickets Fallen / 10 = 1 - (Wickets in Hand / 10)
   A team 6 down has S_W = 0.60; 8 down has S_W = 0.80.
3. **Late-Phase Leverage (S_Late)**:
   As legal balls remaining decrease, margin for error evaporates:
   S_Late = 1 - (Balls Remaining / 120)
4. **Contest Viability Filter (A_t)**:
   If a match is already mathematically lost (RRR > 22.0 with <= 2 wickets in hand), dead-rubber deliveries are downweighted (A_t = 0.50). If a match is effectively won (RRR < 4.0 with >= 6 wickets in hand), pressure is neutralized (A_t = 0.20).

PI_inn2 = (0.50 * S_RRR + 0.25 * S_W + 0.25 * S_Late) * A_t * M_stakes

### 2.2 Innings 1: Setting the Target
In the 1st innings, pressure stems from two distinct match dynamics:
1. **Collapse / Crisis Management (Crisis)**:
   When wickets fall ahead of par, batters must rebuild without stagnating. Expected wickets lost at legal ball b:
   E[W_b] = 7.0 * (b / 120)^0.7
   Crisis = max(0, (Wickets Lost - E[W_b]) / 4.0)
2. **Death Phase Leverage (P_phase)**:
   Overs 16-20 dictate the defensibility of the match total. In overs 16-20:
   P_phase = 0.65 + 0.07 * (Over - 16)
   (Over 16 = 0.65, Over 17 = 0.72, Over 18 = 0.79, Over 19 = 0.86, Over 20 = 0.93).
3. Combining for Innings 1:
   PI_inn1 = max(P_phase, 0.30 + 0.50 * Crisis) * M_stakes

### 2.3 Super Overs & Tournament Stakes Multiplier
- **Super Overs (Innings 3 & 4)**: Instant maximum pressure (PI = 1.0).
- **Tournament Stakes (M_stakes)**: Knockout fixtures (knock_out == 1 or stages: Final, Semi Final, Qualifier, Eliminator) carry an inherent 1.15x stakes multiplier.

### 2.4 The Operational High-Pressure Cutoff (PI >= 0.60)
Deliveries with PI >= 0.60 constitute **24.2%** of all deliveries in the dataset. This precisely isolates the decisive quadrant of T20 cricket:
- **Baseline Deliveries (PI < 0.60)**: 152,330 balls. League Batting SR: 134.3, Batting Avg: 28.1, Bowling Econ: 8.06, Bowling SR: 21.6 balls/wkt.
- **High-Pressure Deliveries (PI >= 0.60)**: 48,669 balls. League Batting SR: 155.9, Batting Avg: 18.9, Bowling Econ: 9.35, Bowling SR: 13.3 balls/wkt.

---

## 3. Performance Measurement: The Clutch Index Architecture

The problem statement mandates measuring how players perform in high-pressure match situations **compared with their usual performance levels**, and ranking those who deliver **most consistently**.

We define the **Clutch Index (CBI)** as a four-component standardized composite score (scaled to Mean = 50, SD = 15):

CBI = 50 + 15 * (0.40 * S_abs + 0.30 * S_lift + 0.20 * S_cons + 0.10 * S_vol)

### 3.1 Batting Architecture
1. **Absolute High-Pressure Execution (S_abs, 40%)**:
   z(HP SR) * 0.40 + z(HP Avg) * 0.30 + z(HP Boundary %) * 0.15 + z(-HP Dot %) * 0.15
2. **Relative Clutch Lift (S_lift, 30%)**:
   - Strike Rate Lift: Delta_SR = HP_SR - Base_SR
   - Average Lift: Delta_Avg = HP_Avg - Base_Avg
   S_lift = 0.60 * z(Delta_SR) + 0.40 * z(Delta_Avg)
   *Note: In high pressure, the league average batter experiences a -9.1 drop in average. Players who maintain or elevate their average alongside a massive SR boost demonstrate generational composure.*
3. **Match-Level Consistency (S_cons, 20%)**:
   Percentage of matches (where batter faced >= 6 HP balls) with a successful clutch contribution (SR >= 150.0 or >= 25 runs).
4. **Volume & Opposition Confidence (S_vol, 10%)**:
   Log-scaled HP balls faced multiplied by the opponent tier weight (1.0 for elite franchise/top-tier T20I; 0.85 for emerging).

### 3.2 Bowling Architecture
1. **Absolute High-Pressure Execution (S_abs, 40%)**:
   z(-HP Econ) * 0.40 + z(-HP BSR) * 0.30 + z(HP Dot %) * 0.15 + z(-HP Boundary %) * 0.15
2. **Relative Clutch Resistance (S_lift, 30%)**:
   - Economy Resistance: Delta_ER = Base_Econ - HP_Econ
   *(For the league, economy balloons by +1.29 in HP. A bowler who keeps economy flat or suppresses it under pressure has positive resistance).*
   - Wicket-Taking Surge: Delta_WktRate = (Wkts/Ball)_HP - (Wkts/Ball)_Base
   S_lift = 0.60 * z(Delta_ER) + 0.40 * z(Delta_WktRate)
3. **Match-Level Consistency (S_cons, 20%)**:
   Percentage of matches (where bowler bowled >= 6 HP balls) where they conceded <= 8.50 RPO or took >= 2 wickets.
4. **Volume & Opposition Confidence (S_vol, 10%)**:
   Log-scaled HP legal balls bowled multiplied by opponent tier weight.

---

## 4. Fair Comparisons & Eligibility Protocols

To ensure statistical validity and prevent small-sample anomalies:
1. **Eligibility Thresholds**:
   - **Batters**: Minimum **180 balls faced under high pressure** (PI >= 0.60) across >= 10 qualifying high-pressure matches.
   - **Bowlers**: Minimum **200 legal balls bowled under high pressure** across >= 12 qualifying high-pressure matches.
2. **Opposition Quality Harmonization**:
   82.5% of balls in the dataset occur against Top-Tier teams (10 IPL franchises + Top 9 Full Members). To prevent stat-padding against associate nations, balls bowled/faced against Emerging nations are weighted at 0.85.
3. **Cross-Competition Balancing (IPL vs T20I)**:
   IPL represents the highest density of elite death bowling and aggressive batting depths. Our model evaluates players on their delivery-level response function rather than raw counting stats, normalizing metrics against the prevailing competition context.

---

## 5. Evidence & Analytical Profiles: Top 10 Batters

### 1. Namboori Tilak Varma (CBI: 70.10) | Rank 1
- **The Empirical Revelation**: Facing 637 high-pressure balls (highest in dataset), Tilak scored an astounding 1230 runs with an HP Strike Rate of **193.09** and an HP Average of **42.41**.
- **The Lift**: While his baseline strike rate is 125.98 (anchor mode), under pressure he accelerates by **+67.11 points** without losing his wicket (Average Lift: +2.98).
- **Match Consistency**: Delivered match-defining contributions in **80.0%** of high-pressure appearances.

### 2. Suryakumar Yadav (CBI: 69.59) | Rank 2
- **The 360-Degree Weapon**: SKY boasts an HP Strike Rate of **209.35** across 492 balls, accompanied by an elite HP Average of **38.15**.
- **Execution Under Fire**: An unprecedented 83.3% match consistency rate. His ability to manipulate death fields makes him virtually undefendable in chases.

### 3. Rishabh Pant (CBI: 67.55) | Rank 3
- **Highest Match Reliability**: Pant produced a jaw-dropping **94.4% consistency rate** across qualifying high-pressure matches.
- **Clutch Differential**: Elevates from a base SR of 126.79 to an HP SR of **195.89** (+69.10 lift), exemplifying fearlessness in crunch chases.

### 4. Tim David (CBI: 66.69) | Rank 4
- **The Pure Death Finisher**: 594 HP balls faced, 1188 runs at exactly **200.00 SR** and **34.94 Average**. Delivered an 87.2% consistency rate across 39 high-pressure games.

### 5. Shashank Singh (CBI: 65.47) | Rank 5
- **The Miracle Chase Anchor**: Boasts the highest HP Average among all qualified batters at **44.00**, combined with a **188.78 SR** (+59.21 lift). Proved his clutch status across IPL 2024-2026 miracle chases.

### 6. Tristan Stubbs (CBI: 65.44) | Rank 6
- **Power & Precision**: 912 runs at **186.89 SR** with an average of **38.00** (+67.36 SR lift). Excellent boundary percentage (27.2%) and minimal dot ball rate.

### 7. Naman Dhir (CBI: 65.38) | Rank 7
- **The Emerging Finisher**: Exceptional **93.8% consistency rate** (15 successful contributions in 16 matches). HP Strike Rate of **199.07** with an average of 30.57.

### 8. Dhruv Jurel (CBI: 64.88) | Rank 8
- **The Royals' Homegrown Clutch Star**: HP Average of **35.33** with a **189.96 SR** (+59.52 lift) and **81.3% consistency**. Ice in his veins during lower-middle order crises.

### 9. Sikandar Raza (CBI: 60.66) | Rank 9
- **The Lone Crusader**: Consistently elevates under-pressure teams with an HP SR of **187.18** (+67.39 lift) and 81.3% consistency.

### 10. Heinrich Klaasen (CBI: 59.97) | Rank 10
- **The Powerhouse Destructor**: 905 runs at an HP SR of **193.38** and **81.8% consistency**. His sheer boundary output creates instant match leverage.

---

## 6. Evidence & Analytical Profiles: Top 10 Bowlers

### 1. Sunil Narine (CBI: 81.78) | Rank 1
- **The Unreachable Standard**: An astronomical High-Pressure Economy of **6.28** across 299 legal balls—more than 3 runs per over below the 9.35 league pressure average!
- **Positive Resistance**: Narine actually concedes *fewer* runs per over under pressure than in baseline situations (Delta_ER = +0.73).
- **Wicket Threat & Consistency**: 11.5 balls per wicket and an **83.3% match success rate**.

### 2. Jasprit Bumrah (CBI: 73.64) | Rank 2
- **The Ultimate Match Winner**: Faced with 554 high-pressure legal balls (nearly all death overs 17-20 and crunch powerplay), Bumrah maintains a staggering **7.18 Economy** and takes 43 wickets (12.88 balls/wkt).
- **Relentless Reliability**: Achieved positive clutch match output in **74.1% of 54 high-pressure matches**.

### 3. Shaheen Afridi (CBI: 68.52) | Rank 3
- **Lethal Wicket Taker**: 26 wickets in 254 HP balls (Strike Rate: **9.77 balls/wkt**), keeping economy at **8.01** with a 69.0% consistency rate.

### 4. Matheesha Pathirana (CBI: 67.45) | Rank 4
- **The Death Yorker Specialist**: 49 wickets in 529 HP balls at an extraordinary strike rate of **10.80 balls/wkt** and **8.89 Economy**.

### 5. Mitchell Santner (CBI: 65.43) | Rank 5
- **Chokehold Spin**: HP Economy of **7.77** with **71.4% consistency** across 35 high-pressure matches.

### 6. Rasikh Salam (CBI: 64.59) | Rank 6
- **Elite Strike Bowler**: Best HP strike rate in the top tier at **8.35 balls per wicket** (26 wickets in 217 balls) while maintaining zero economy inflation.

### 7. Haris Rauf (CBI: 64.37) | Rank 7
- **Raw Pace Execution**: 36 wickets in 343 HP balls (Strike Rate: **9.53 balls/wkt**, 8.83 Economy).

### 8. Kuldeep Yadav (CBI: 63.45) | Rank 8
- **Middle-Overs Pressure Enforcer**: HP Economy of **8.64**, 36 wickets, 66.7% consistency rate.

### 9. Rashid Khan (CBI: 62.18) | Rank 9
- **The Volume Workhorse**: 707 HP balls (highest in cricket), taking 62 wickets (11.40 balls/wkt) at an **8.62 Economy**.

### 10. Adam Zampa (CBI: 62.12) | Rank 10
- **Tournament Match Winner**: 34 wickets at **9.41 balls per wicket** with an 8.78 Economy under pressure.

---

## 7. Reliability, Validation & Limitations

### 7.1 Cross-Season Consistency
We evaluated model rank correlations across split seasons (2022-2023 vs 2024-2026). The top-tier performers displayed a Spearman rank correlation of rho = 0.79 for batters and rho = 0.83 for bowlers, demonstrating that pressure reliability is an intrinsic player skill rather than variance.

### 7.2 Sensitivity Analysis on Pressure Threshold (PI >= 0.55 vs 0.60 vs 0.65)
Varying the pressure threshold across +/- 0.05 resulted in 9 out of 10 batters and 9 out of 10 bowlers remaining unchanged in the top 10, validating stability.

### 7.3 Model Limitations
1. **Unrecorded Field Placements**: Ball-by-ball records do not capture whether a bowler had favorable boundary dimensions or field restrictions beyond standard phase rules.
2. **Dropped Catches / Misfields**: A dropped catch credited as runs hurts bowler metrics despite perfect execution.
3. **Emerging Sample Horizons**: Promising clutch batters with fewer than 180 HP balls were excluded by eligibility filters to maintain rigorous confidence.

---

## 8. Conclusion & Rajasthan Royals Recommendations
1. **Batting Finishers**: Prioritize batters who demonstrate a positive Delta_SR > +50 while maintaining an HP average above 32.0 (e.g. Tilak Varma, Dhruv Jurel, Tristan Stubbs).
2. **Bowling Enforcers**: In death overs, raw economy resistance (Delta_ER >= 0) combined with a sub-12.0 bowling strike rate (Narine, Bumrah, Pathirana) is the single most predictive metric for winning tight matches.
3. **Tactical Utilization**: Deploy clutch spinners (Narine, Kuldeep) in the 15th-17th overs of run chases to induce fatal dot-ball pressure prior to the pace death overs.

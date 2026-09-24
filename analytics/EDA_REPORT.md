# Titanic EDA Report

## Initial profile

Shape: `(891, 15)`

### df.info()
```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    object  
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    object  
 8   class        891 non-null    category
 9   who          891 non-null    object  
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object  
 13  alive        891 non-null    object  
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.7+ KB

```
### df.describe()
```text
         survived      pclass         age       sibsp       parch        fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
```
## Missing-value percentages

|             |   missing_percent |
|:------------|------------------:|
| deck        |             77.22 |
| age         |             19.87 |
| embarked    |              0.22 |
| embark_town |              0.22 |
## Cleaning decisions

| column      |   missing_percent | strategy                                  |
|:------------|------------------:|:------------------------------------------|
| deck        |             77.22 | drop column (>30%; imputation unreliable) |
| age         |             19.87 | median imputation (5–30%)                 |
| embarked    |              0.22 | drop affected rows (<5%)                  |
| embark_town |              0.22 | drop affected rows (<5%)                  |

Cleaned shape: `(889, 14)`

## Univariate analysis

- Age IQR outliers: **65**
- Fare IQR outliers: **114**
- Fare mean: **32.0967**
- Fare median: **14.4542**
- Fare mode: **8.0500**
- Fare distribution: **right-skewed** because mean/median/mode are ordered 32.10 / 14.45 / 8.05.

## Survival-rate breakdowns

Sex — female: **0.740**, male: **0.189**.

By pclass:

|   pclass |   survival_rate |
|---------:|----------------:|
|        1 |           0.626 |
|        2 |           0.473 |
|        3 |           0.242 |

By sex and pclass:

|               |   survival_rate |
|:--------------|----------------:|
| ('female', 1) |           0.967 |
| ('female', 2) |           0.921 |
| ('female', 3) |           0.500 |
| ('male', 1)   |           0.369 |
| ('male', 2)   |           0.157 |
| ('male', 3)   |           0.135 |

### Two strongest absolute off-diagonal correlations

- `pclass` vs `fare`: correlation = **-0.5482**.
- `sibsp` vs `parch`: correlation = **0.4145**.

The correlation matrix contains exactly the six required columns; `adult_male` and `alone` are excluded.

### Chart 1 — Survival by sex
The survival rates differ substantially between the two sex categories. This indicates sex is strongly associated with the observed survival outcome in this dataset, although the chart is descriptive rather than causal.

### Chart 2 — Survival by class and sex
Survival varies across passenger classes within the sex groups. The combined view shows why looking at only one grouping can hide important differences between subgroups.

### Chart 3 — Age and survival
The age distributions overlap considerably, but their medians and spreads can still differ. This suggests age may contribute information while not being sufficient by itself to separate survivors from non-survivors.

### Chart 4 — Fare and survival
Fare distributions differ between the observed survival groups, with the non-survivor group and survivor group showing different central tendencies. Because fare is related to passenger class, this relationship should be interpreted together with pclass rather than in isolation.

### Chart 5 — Age, fare and survival
The scatter plot combines two numeric variables with the survival label. It illustrates that the groups overlap, reinforcing the need for a multivariate model rather than a single-threshold rule.


## Standardization sanity check — age

Before: mean=29.315152, std=12.984932. After: mean=0.000000, std=1.000000.

## Standardization sanity check — fare

Before: mean=32.096681, std=49.697504. After: mean=0.000000, std=1.000000.

This standardization is EDA-only and is not passed into the modeling pipeline. The modeling pipeline performs its own training-only scaling.

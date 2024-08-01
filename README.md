# TJ_HYD_NAM

Python implementation of NedborAfstromnings Model (NAM) lumped rainfall–runoff model, based on the original code
from [NAM_Model](https://github.com/hckaraman/NAM_Model) by [hckaraman](https://github.com/hckaraman)

### Installation

```
pip install tj_hyd_nam
```

### Getting Started

#### Prepare the Dataset

The dataset should include columns: Date, Temperature, Precipitation, Evapotranspiration, and Discharge, with column
names customizable.

| Date       | Temp | Q       | P   | E    |
|------------|------|---------|-----|------|
| 10/9/2016  | 15.4 | 0.25694 | 0   | 2.79 |
| 10/10/2016 | 14.4 | 0.25812 | 0   | 3.46 |
| 10/11/2016 | 14.9 | 0.30983 | 0   | 3.65 |
| 10/12/2016 | 16.1 | 0.31422 | 0   | 3.46 |
| 10/13/2016 | 20.1 | 0.30866 | 0   | 5.64 |
| 10/14/2016 | 13.9 | 0.30868 | 0   | 3.24 |
| 10/15/2016 | 11.1 | 0.31299 | 0   | 3.41 |
| ...        | ...  | ...     | ... | ...  |

The time intervals between dates must be equal (e.g., 24 hours) for the model to function accurately.

#### Initialize the NAM Model

```python
import pandas as pd
from tj_hyd_nam import TJHydNAM, NAMColNames, NAMConfig

# Load the dataset
df = pd.read_csv('data_example.csv')
# Specify the column names as required
nam_col_names = NAMColNames(
    date='Date',
    temperature='Temp',
    precipitation='P',
    evapotranspiration='E',
    discharge='Q'
)
# Configure the model parameters
nam_config = NAMConfig(
    area=58.8,
    start_date=None,
    end_date=None,
    interval=24.0,
    spin_off=0.0,
    umax=0.97,
    lmax=721.56,
    cqof=0.18,
    ckif=495.91,
    ck12=25.16,
    tof=0.97,
    tif=0.11,
    tg=0.19,
    ckbf=1121.74,
    csnow=2.31,
    snowtemp=3.51,
)
NAM = TJHydNAM(
    dataset=df,
    nam_col_names=nam_col_names,
    nam_config=nam_config
)
print(NAM)
```

The output will detail the NAM model based on the loaded dataset:

```
TJ_HYD_NAM 🍃 🌧 ☔ 💦
FROM: 2016-10-09 00:00:00+00:00
TO: 2018-09-30 00:00:00+00:00
NAMConfig(area=58.8, start_date=None, end_date=None, flow_rate=0.6805555555555555, interval=24.0, spin_off=0.0, umax=0.97, lmax=721.56, cqof=0.18, ckif=495.91, ck12=25.16, tof=0.97, tif=0.11, tg=0.19, ckbf=1121.74, csnow=2.31, snowtemp=3.51)
NAMStatistic(nse=-0.17835445482281131, rmse=1.7864602332317054, fbias=75.23828249740461)
```

#### Display and Save Graphs

```python
# Plot and save the discharge comparison graph
NAM.show_discharge(save=True, filename='discharge.png')
# Plot and save all calculated model information
NAM.show(save=True, filename='result.png')
```

#### Optimize the Model

```python
NAM.optimize()
print(NAM)
```

```shell
Optimization terminated successfully    (Exit mode 0)
            Current function value: 2.036882878807083
            Iterations: 7
            Function evaluations: 70
            Gradient evaluations: 7
TJ_HYD_NAM 🍃 🌧 ☔ 💦
FROM: 2016-10-09 00:00:00+00:00
TO: 2018-09-30 00:00:00+00:00
NAMConfig(area=58.8, start_date=None, end_date=None, flow_rate=0.6805555555555555, interval=24.0, spin_off=0.0, umax=0.97, lmax=721.56, cqof=0.18, ckif=495.91, ck12=25.16, tof=0.97, tif=0.11, tg=0.19, ckbf=1121.74, csnow=2.31, snowtemp=3.51)
NAMStatistic(nse=-0.5318680456177058, rmse=2.036882878807083, fbias=91.77841692580132)
```

#### Reconfigure the Model Based on Properties

The model parameters will change and be recalculated based on new properties.

```python
NAM.re_config_by_props(
    start_date=pd.to_datetime('09/10/2016', dayfirst=True, utc=True),
    end_date=pd.to_datetime('20/10/2016', dayfirst=True, utc=True)
)
```

#### Reconfigure All Parameters

```python
NAM.re_config(
    NAMConfig(
        area=60,
        start_date=None,
        end_date=None,
        interval=24.0,
        spin_off=0.0,
        umax=0.8,
        lmax=719.56,
        cqof=0.14,
        ckif=493.86,
        ck12=45.16,
        tof=0.97,
        tif=0.45,
        tg=0.19,
        ckbf=1121.74,
        csnow=2.31,
        snowtemp=3.51,
    )
)
```

#### Save Calculated Model Data

```python
NAM.save_to_csv('result.csv')
```

#### Convert Calculated Data to DataFrame

```python
nam_df = NAM.to_dataframe()
```

#### Save the Model

```python
NAM.save('nam_model')
```

#### Load a Saved Model

```python
SAVED_NAM = NAM.load('nam_model.tjnam')
```

#### Use the Previous Model's Configuration for Prediction

```python
PRED_NAM = TJHydNAM(
    pd.read_csv('future_data.csv'),
    NAMColNames(
        date='Date',
        temperature='Temp',
        precipitation='P',
        evapotranspiration='E',
        discharge='Q'
    ),
    SAVED_NAM.config
)
```
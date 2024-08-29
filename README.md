# TJ_HYD_NAM

**TJ_HYD_NAM** is a Python implementation of the NedborAfstromnings Model (NAM), a lumped rainfall–runoff model. This implementation is based on the original code from [NAM_Model](https://github.com/hckaraman/NAM_Model) by [hckaraman](https://github.com/hckaraman).

## Installation

You can install the package via pip:

```bash
pip install tj_hyd_nam
```

## Getting Started

### 1. Prepare the Dataset

The dataset should contain the following columns: Date, Temperature, Precipitation, Evapotranspiration, and Discharge. Column names can be customized.

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

Ensure that the time intervals between dates are consistent (e.g., 24 hours) for accurate model performance.

### 2. Initialize the NAM Model

```python
import pandas as pd
from tj_hyd_nam import TJHydNAM, NAMColNames, NAMConfig

# Load the dataset
df = pd.read_csv('data_example.csv')

# Specify the column names
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

# Initialize the NAM model
NAM = TJHydNAM(
    dataset=df,
    nam_col_names=nam_col_names,
    nam_config=nam_config
)

print(NAM)
```

The output will display details of the NAM model based on the loaded dataset.

### 3. Display and Save Graphs

```python
# Plot and save the discharge comparison graph
NAM.show_discharge(save=True, filename='discharge.png')

# Plot and save all calculated model information
NAM.show(save=True, filename='result.png')
```

### 4. Optimize the Model

```python
NAM.optimize()
print(NAM)
```

The optimization process will refine the model parameters and display the updated configuration.

### 5. Reconfigure the Model Based on Properties

You can reconfigure the model parameters and recalculate the results based on new date ranges or other properties.

```python
NAM.re_config_by_props(
    start_date=pd.to_datetime('09/10/2016', dayfirst=True, utc=True),
    end_date=pd.to_datetime('20/10/2016', dayfirst=True, utc=True)
)
```

### 6. Reconfigure All Parameters

To fully reconfigure the model, use:

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

### 7. Save Calculated Model Data

```python
NAM.save_to_csv('result.csv')
```

### 8. Convert Calculated Data to DataFrame

```python
nam_df = NAM.to_dataframe()
```

### 9. Save and Load the Model

Save the current model configuration:

```python
NAM.save('nam_model')
```

Load a saved model:

```python
SAVED_NAM = NAM.load('nam_model.tjnam')
```

### 10. Use the Previous Model's Configuration for Prediction

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

### 11. Accessing Calculated Variables (>=1.1.0)

You can access various calculated variables directly:

```python
NAM.size       # Access the value of _size
NAM.date       # Access the value of _date
NAM.T          # Access the value of _T (temperature series)
NAM.P          # Access the value of _P (precipitation series)
NAM.E          # Access the value of _E (evaporation series)
NAM.Q_obs      # Access the value of _Q_obs (observed discharge)
NAM.U_soil     # Access the value of _U_soil (upper soil layer moisture)
NAM.S_snow     # Access the value of _S_snow (snow storage)
NAM.Q_snow     # Access the value of _Q_snow (snowmelt discharge)
NAM.Q_inter    # Access the value of _Q_inter (interflow discharge)
NAM.E_eal      # Access the value of _E_eal (actual evapotranspiration)
NAM.Q_of       # Access the value of _Q_of (overland flow)
NAM.Q_g        # Access the value of _Q_g (groundwater discharge)
NAM.Q_bf       # Access the value of _Q_bf (baseflow)
NAM.Q_sim      # Access the value of _Q_sim (simulated discharge)
NAM.L_soil     # Access the value of _L_soil (soil moisture)
```

## Exception Classes

### `MissingColumnsException`

Raised when one or more required columns are missing from the dataset. The exception message specifies which column is missing, helping users to correct their data.

### `ColumnContainsEmptyDataException`

Raised when a specified column contains empty data. This ensures that critical data is complete before proceeding with the model.

### `InvalidDatetimeException`

Raised when an invalid datetime value is encountered. This could be due to incorrect formatting or values that do not conform to the expected standards.

### `InvalidDatetimeIntervalException`

Raised when an invalid datetime interval is provided. This ensures that the interval between dates or times is logically consistent.

### `InvalidStartDateException`

Raised when the provided start date is invalid. This helps catch errors where the beginning of a time-based event or range is not properly defined.

### `InvalidEndDateException`

Raised when the provided end date is invalid. This ensures the end of a time-based event or range is correctly defined.

### `InvalidDateRangeException`

Raised when the date range provided is invalid, such as when the start date is after the end date.

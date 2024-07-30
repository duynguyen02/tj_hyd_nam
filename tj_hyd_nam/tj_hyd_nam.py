import dataclasses
import math
from typing import Any

import numpy as np
import pandas
import pandas as pd
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit
from pandas import Series, DataFrame
from scipy.optimize import minimize

from .nam_func import nam_cal


class MissingColumnsException(Exception):
    def __init__(self, col: str):
        super(f"Missing columns, please provide the following column: '{col}'.")


class ColumnContainsEmptyDataException(Exception):
    def __init__(self, col: str):
        super(f"Column '{col}' contains empty data.")


class InvalidDatetimeException(Exception):
    def __init__(self):
        super(f"Invalid datetime.")


class InvalidDatetimeIntervalException(Exception):
    def __init__(self):
        super(f"Invalid datetime interval.")


@dataclasses.dataclass
class NAMConfig:
    area: float
    interval: float = 24.0
    umax: float = 10.0
    lmax: float = 100.0
    cqof: float = 0.1
    ckif: float = 200.0
    ck12: float = 10.0
    tof: float = 0.0
    tif: float = 0.0
    tg: float = 0.0
    ckbf: float = 1000.0
    csnow: float = 0.0
    snowtemp: float = 0.0

    def to_initial_params(self):
        return np.array(
            [
                self.umax,
                self.lmax,
                self.cqof,
                self.ckif,
                self.ck12,
                self.tof,
                self.tif,
                self.tg,
                self.ckbf,
                self.csnow,
                self.snowtemp,
            ]
        )


@dataclasses.dataclass
class NAMDatasetColumnNames:
    time: str = 'Time'
    temperature: str = 'Temperature'
    precipitation: str = 'Precipitation'
    evapotranspiration: str = 'Evapotranspiration'
    discharge: str = 'Discharge'


@dataclasses.dataclass
class NAM:
    time: Series
    temperature: Series  # T
    precipitation: Series  # P
    evapotranspiration: Series  # E
    observed_discharge: Series  # Qobs
    simulator_discharge: ndarray[Any, dtype[floating[_64Bit]]]  # Qsim
    soil_moisture: ndarray[Any, dtype[floating[_64Bit]]]  # Lsoil
    upper_soil_layer_moisture = None  # Usoil
    snow_storage = None  # Ssnow
    snowmelt_discharge = None  # Qsnow
    interflow_discharge = None  # Qinter
    actual_evapotranspiration = None  # Eeal
    overland_flow = None  # Qof
    groundwater_discharge = None  # Qg
    baseflow = None  # Qbf
    size: int


class TJHydNAM:
    def __init__(
            self,
            nam_config: NAMConfig,
            nam_dataset_column_names: NAMDatasetColumnNames,
            dataset: DataFrame
    ):
        self._nam_configs = nam_config
        self._nam_dataset_column_names = nam_dataset_column_names
        self._parameters = None
        # Min - Max
        self._bounds = ((10, 20), (100, 300), (0.1, 1), (200, 1000), (10, 50),
                        (0, 0.99), (0, 0.99), (0, 0.99), (1000, 4000), (0, 0), (0, 0))

        dataset_ = dataset.copy()
        self._validate_dataset(dataset_)
        dataset_size = dataset.size
        self._nam = NAM(
            time=dataset_[nam_dataset_column_names.time],
            temperature=dataset_[nam_dataset_column_names.temperature],
            precipitation=dataset_[nam_dataset_column_names.precipitation],
            evapotranspiration=dataset_[nam_dataset_column_names.evapotranspiration],
            observed_discharge=dataset_[nam_dataset_column_names.discharge],
            simulator_discharge=np.zeros(dataset_size),
            soil_moisture=np.zeros(dataset_size),
            size=dataset_size
        )
        self._flow_rate = nam_config.area / (3.6 * nam_config.interval)
        self._spin_off = 0

    def _validate_dataset(
            self,
            dataset: DataFrame
    ):
        required_columns = [
            self._nam_dataset_column_names.time,
            self._nam_dataset_column_names.temperature,
            self._nam_dataset_column_names.precipitation,
            self._nam_dataset_column_names.evapotranspiration,
            self._nam_dataset_column_names.discharge
        ]

        for col in required_columns:
            if col not in dataset.columns:
                raise MissingColumnsException(col)

        dataset_size = dataset.size
        for column in dataset.columns:
            if len(dataset[column]) == dataset_size:
                raise ColumnContainsEmptyDataException(
                    column
                )

        try:
            dataset[self._nam_dataset_column_names.time] = pandas.to_datetime(
                dataset[self._nam_dataset_column_names.time],
                utc=True
            )
        except Exception as _:
            str(_)
            raise InvalidDatetimeException()

        dataset['Interval'] = dataset[self._nam_dataset_column_names.time].diff()
        interval_hours = pd.Timedelta(hours=self._nam_configs.interval)
        is_valid_interval_hours = dataset['Interval'].dropna().eq(interval_hours).all()
        if not is_valid_interval_hours:
            raise InvalidDatetimeIntervalException()

    def _objective(self, x):
        (self._nam.simulator_discharge,
         self._nam.soil_moisture,
         self._nam.upper_soil_layer_moisture,
         self._nam.snow_storage,
         self._nam.snowmelt_discharge,
         self._nam.interflow_discharge,
         self._nam.actual_evapotranspiration,
         self._nam.overland_flow,
         self._nam.groundwater_discharge,
         self._nam.baseflow) = nam_cal(
            x,
            self._nam.precipitation,
            self._nam.temperature,
            self._nam.evapotranspiration,
            self._flow_rate,
            self._nam_configs.interval,
            self._spin_off
        )

        n = math.sqrt((sum((self._nam.simulator_discharge - self._nam.observed_discharge) ** 2)) / len(
            self._nam.observed_discharge))

        return n

    def run(self, cal: bool = False):
        params = None
        if cal:
            self._parameters = minimize(
                self._objective, self._nam_configs.to_initial_params(), method='SLSQP', bounds=self._bounds,
                options={'maxiter': 1e8, 'disp': True}
            )
            params = self._parameters.x
        else:
            params = self._nam_configs.to_initial_params()

        (self._nam.simulator_discharge,
         self._nam.soil_moisture,
         self._nam.upper_soil_layer_moisture,
         self._nam.snow_storage,
         self._nam.snowmelt_discharge,
         self._nam.interflow_discharge,
         self._nam.actual_evapotranspiration,
         self._nam.overland_flow,
         self._nam.groundwater_discharge,
         self._nam.baseflow) = nam_cal(
            params,
            self._nam.precipitation,
            self._nam.temperature,
            self._nam.evapotranspiration,
            self._flow_rate,
            self._nam_configs.interval,
            self._spin_off
        )

    def stats(self):
        mean = np.mean(self._nam.observed_discharge)

        self.NSE = 1 - (sum((self.Qsim - self.Qobs) ** 2) /
                        sum((self.Qobs - mean) ** 2))
        self.RMSE = np.sqrt(sum((self.Qsim - self.Qobs) ** 2) / len(self.Qsim))
        self.PBIAS = (sum(self.Qobs - self.Qsim) / sum(self.Qobs)) * 100
        self.statistics = obj.calculate_all_functions(self.Qobs, self.Qsim)

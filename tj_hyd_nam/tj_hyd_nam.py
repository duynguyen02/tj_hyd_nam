import dataclasses
import datetime
import math
from typing import Any

import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit
from pandas import Series, DataFrame
from scipy.optimize import minimize

from .nam_exception import MissingColumnsException, ColumnContainsEmptyDataException, InvalidDatetimeException, \
    InvalidDatetimeIntervalException


@dataclasses.dataclass
class NAMColNames:
    date: str = 'Date'
    temperature: str = 'Temperature'
    precipitation: str = 'Precipitation'
    evapotranspiration: str = 'Evapotranspiration'
    discharge: str = 'Discharge'


@dataclasses.dataclass
class NAMConfig:
    area: float
    first_date: datetime.datetime
    last_date: datetime.datetime
    flow_rate: float = 0.0
    interval: float = 24.0
    spin_off: float = 0.0
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
    bounds = ((10, 20), (100, 300), (0.1, 1), (200, 1000), (10, 50),
              (0, 0.99), (0, 0.99), (0, 0.99), (1000, 4000), (0, 0), (0, 0))

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

    def __post_init__(self):
        self.flow_rate = self.area / (3.6 * self.interval)


@dataclasses.dataclass
class NAMStatistic:
    nse: float = None
    rmse: float = None
    fbias: float = None


class TJHydNAM:
    def __init__(
            self,
            dataset: DataFrame,
            nam_col_names: NAMColNames,
            nam_config: NAMConfig
    ):
        self._Q_sim = None
        proc_dataset = dataset.copy()
        self._validate_dataset(proc_dataset, nam_config, nam_col_names)

        self._dataset = dataset.copy()
        self._nam_col_names = nam_col_names
        self._nam_configs: NAMConfig = nam_config
        self._nam_statistic: NAMStatistic = NAMStatistic()

        self._size: int = proc_dataset.size
        self._date: Series = proc_dataset[nam_col_names.date]
        self._T: Series = proc_dataset[nam_col_names.temperature]  # temperature
        self._P: Series = proc_dataset[nam_col_names.precipitation]  # precipitation
        self._E: Series = proc_dataset[nam_col_names.evapotranspiration]  # evapotranspiration
        self._Q_obs: Series = proc_dataset[nam_col_names.discharge]  # observed_discharge
        self._U_soil = None  # upper_soil_layer_moisture
        self._S_snow = None  # snow_storage
        self._Q_snow = None  # snowmelt_discharge
        self._Q_inter = None  # interflow_discharge
        self._E_eal = None  # actual_evapotranspiration
        self._Q_of = None  # overland_flow
        self._Q_g = None  # groundwater_discharge
        self._Q_bf = None  # baseflow
        self._Q_sim: ndarray[Any, dtype[floating[_64Bit]]] = np.zeros(self._size)  # simulator_discharge
        self._L_soil: ndarray[Any, dtype[floating[_64Bit]]] = np.zeros(self._size)  # soil_moisture

        self._calculate()



    @staticmethod
    def _validate_dataset(
            dataset: DataFrame,
            nam_config: NAMConfig,
            nam_col_names: NAMColNames
    ):
        required_columns = [
            nam_col_names.date,
            nam_col_names.temperature,
            nam_col_names.precipitation,
            nam_col_names.evapotranspiration,
            nam_col_names.discharge
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
            dataset[nam_col_names.date] = pandas.to_datetime(
                dataset[nam_col_names.date],
                utc=True
            )
        except Exception as _:
            str(_)
            raise InvalidDatetimeException()

        dataset['Interval'] = dataset[nam_col_names.date].diff()
        interval_hours = pd.Timedelta(hours=nam_config.interval)
        is_valid_interval_hours = dataset['Interval'].dropna().eq(interval_hours).all()
        if not is_valid_interval_hours:
            raise InvalidDatetimeIntervalException()

    def _nam_cal(self, x):
        qofmin, beta, pmm, carea = 0.4, 0.1, 10, 1.0
        interval = self._nam_configs.interval

        states = np.array([0, 0, 0.9 * x[1], 0, 0, 0, 0, 0.1])
        snow, u, _l, if1, if2, of1, of2, bf = states

        umax, lmax, cqof, ckif, ck12, tof, tif, tg, ckbf, csnow, snowtemp = x[:11]
        ckif /= interval
        ck12 /= interval
        ckbf /= interval

        lfrac = _l / lmax
        fact = self._nam_configs.flow_rate

        q_sim = np.zeros(len(self._P))
        l_soil, u_soil, s_snow, q_snow = (np.zeros(len(self._P)) for _ in range(4))
        q_inter, e_eal, q_of, q_g, q_bf = (np.zeros(len(self._P)) for _ in range(5))

        for t, (prec, evap, temp) in enumerate(zip(self._P, self._E, self._T)):
            if temp < snowtemp:
                snow += prec
                qs = 0
            else:
                qs = min(csnow * temp, snow)
                snow -= qs

            u1 = u + (prec + qs) if temp >= 0 else u
            eau = min(u1, evap)
            eal = (evap - eau) * lfrac if u1 > evap else 0

            u2 = min(u1 - eau, umax)
            qif = (lfrac - tif) / (1 - tif) * u2 / ckif if lfrac > tif else 0
            u3 = u1 - eau - qif
            pn = max(0, u3 - umax)
            u = min(u3, umax)

            n = int(pn / pmm) + 1
            pnlst = pn - (n - 1) * pmm
            eal /= n

            qofsum, gsum = 0, 0
            for i in range(n):
                pn = pmm if i < n - 1 else pnlst
                qof = cqof * (lfrac - tof) / (1 - tof) * pn if lfrac > tof else 0
                qofsum += qof
                g = (lfrac - tg) / (1 - tg) * (pn - qof) if lfrac > tg else 0
                gsum += g

            c = np.exp(-1. / ckbf)
            bf = bf * c + gsum * carea * (1 - c)

            c = np.exp(-1. / ck12)
            if1 = if1 * c + qif * (1 - c)
            if2 = if2 * c + if1 * (1 - c)

            of = 0.5 * (of1 + of2) / interval
            ckqof = ck12 * (of / qofmin) ** (-beta) if of > qofmin else ck12
            c = np.exp(-1. / ckqof)
            of1 = of1 * c + qofsum * (1 - c)
            of2 = of2 * c + of1 * (1 - c)

            if t >= self._nam_configs.spin_off:
                q_sim[t] = fact * (if2 + of2 + bf)
                l_soil[t], u_soil[t] = lfrac, u
                s_snow[t], q_snow[t] = snow, qs
                q_inter[t], e_eal[t] = qif, eal
                q_of[t], q_g[t], q_bf[t] = qofsum, gsum, bf
                dl = pn - qofsum - gsum
                _l = min(_l + dl - eal, lmax)
                lfrac = _l / lmax

        self._Q_sim = q_sim
        self._L_soil = l_soil
        self._U_soil = u_soil
        self._S_snow = s_snow
        self._Q_snow = q_snow
        self._Q_inter = q_inter
        self._E_eal = e_eal
        self._Q_of = q_of
        self._Q_g = q_g
        self._Q_bf = q_bf

        # return q_sim, l_soil, u_soil, s_snow, q_snow, q_inter, e_eal, q_of, q_g, q_bf

    def _objective(self, x):
        self._nam_cal(x)
        n = math.sqrt((sum((self._Q_sim - self._Q_obs) ** 2)) / len(self._Q_obs))
        return n

    def _stats(self):
        mean = np.mean(self._Q_obs)
        self._nam_statistic.nse = 1 - (sum((self._Q_sim - self._Q_obs) ** 2) /
                                       sum((self._Q_obs - mean) ** 2))
        self._nam_statistic.rmse = float(np.sqrt(sum((self._Q_sim - self._Q_obs) ** 2) / len(self._Q_sim)))
        self._nam_statistic.fbias = (sum(self._Q_obs - self._Q_sim) / sum(self._Q_obs)) * 100

    def _run(self, cal: bool = False):
        if cal:
            params = minimize(
                self._objective,
                self._nam_configs.to_initial_params(),
                method='SLSQP',
                bounds=self._nam_configs.bounds,
                options={'maxiter': 1e8, 'disp': True}
            ).x
        else:
            params = self._nam_configs.to_initial_params()
        self._nam_cal(params)
        self._stats()

    def _init_data(self):
        pass

    def _calculate(self):
        # self._validate_dataset()
        self._init_data()
        self._run(False)

    def optimize(self):
        self._run(True)

    def to_dataframe(self):
        data = {
            'date': self._date,
            'temperature': self._T,
            'precipitation': self._P,
            'evapotranspiration': self._E,
            'observed_discharge': self._Q_obs,
            'upper_soil_layer_moisture': self._U_soil,
            'snow_storage': self._S_snow,
            'snowmelt_discharge': self._Q_snow,
            'interflow_discharge': self._Q_inter,
            'actual_evapotranspiration': self._E_eal,
            'overland_flow': self._Q_of,
            'groundwater_discharge': self._Q_g,
            'baseflow': self._Q_bf,
            'simulator_discharge': self._Q_sim,
            'soil_moisture': self._L_soil
        }
        return pd.DataFrame(data)

    def save_to_csv(self, filename):
        df = self.to_dataframe()
        df.to_csv(filename, index=False)

    @property
    def stats(self):
        return self._nam_statistic

    @property
    def config(self):
        return self._nam_configs

    def re_config(self, nam_config: NAMConfig):
        self._nam_configs = nam_config
        self._validate_dataset(self._dataset, self._nam_configs, self._nam_col_names)
        self._calculate()

    def re_config_by_props(self, **kwargs):
        attributes = [
            'area', 'flow_rate', 'interval', 'spin_off', 'umax', 'lmax', 'cqof',
            'ckif', 'ck12', 'tof', 'tif', 'tg', 'ckbf', 'csnow', 'snowtemp'
        ]

        for attr in attributes:
            setattr(self._nam_configs, attr, kwargs.get(attr, getattr(self._nam_configs, attr)))
        self._validate_dataset(self._dataset, self._nam_configs, self._nam_col_names)
        self._calculate()

    def show(self, save: bool = False, filename: str = 'result.png'):
        df = self.to_dataframe()
        date = df['date']
        df = df.drop(columns=['date'])
        columns = df.columns
        fig, axes = plt.subplots(nrows=len(columns) + 1, ncols=1, figsize=(10, 2 * (len(columns) + 1)))
        for i, column in enumerate(columns):
            axes[i].plot(date, df[column], label=column)
            axes[i].set_title(column)
            axes[i].legend()
        axes[-1].plot(date, df['simulator_discharge'], label='simulator_discharge', color='blue')
        axes[-1].plot(date, df['observed_discharge'], label='observed_discharge', color='orange')
        axes[-1].set_title('Simulator vs Observed Discharge')
        axes[-1].legend()
        plt.tight_layout()
        plt.show()
        if save:
            fig.savefig(filename, dpi=300)

    def __str__(self):
        return f"""TJ_HYD_NAM 🍃 🌧 ☔ 💦
FROM: {self._date.iloc[0]}
TO: {self._date.iloc[-1]}
{self._nam_configs}
{self._nam_statistic}
        """

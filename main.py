import datetime

import pandas as pd
from matplotlib import pyplot as plt

from tj_hyd_nam.tj_hyd_nam import TJHydNAM, NAMColNames, NAMConfig

import pandas as pd
from tj_hyd_nam import TJHydNAM, NAMColNames, NAMConfig

NAM = TJHydNAM(
    pd.read_csv('Test.csv'),
    NAMColNames(
        date='Date',
        temperature='Temp',
        precipitation='P',
        evapotranspiration='E',
        discharge='Q'
    ),
    NAMConfig(
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
)
NAM.optimize()
print(NAM)
NAM.show_discharge()
NAM.show()
NAM.re_config_by_props(
    start_date=pd.to_datetime('09/10/2016', dayfirst=True, utc=True),
    end_date=pd.to_datetime('20/10/2016', dayfirst=True, utc=True)
)
NAM.show()
print(NAM)

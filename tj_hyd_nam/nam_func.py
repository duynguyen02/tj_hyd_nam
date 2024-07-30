import numpy as np
import math


# Define NAM
def nam_cal(x, p, t, e, area, delta_t, spin_off):
    # Parameters
    q_of_min = 0.4
    beta = 0.1
    p_mm = 10
    c_area = 1.0

    # Set initial states
    states = np.array([0, 0, 0.9 * x[1], 0, 0, 0, 0, 0.1])
    snow, u, l, if1, if2, of1, of2, bf = states

    # Set parameters
    umax = x[0]
    lmax = x[1]
    cqof = x[2]
    ck_if = x[3] / delta_t
    ck_12 = x[4] / delta_t
    tof = x[5]
    tif = x[6]
    tg = x[7]
    ck_bf = x[8] / delta_t
    qs = 0
    csnow = x[9]
    snow_temp = x[10]
    l_frac = l / lmax
    fact = area

    # Arrays to store results
    q_sim = np.zeros(len(p))  # Simulated Discharge
    l_soil = np.zeros(len(p))  # Water content in root zone (l)
    u_soil = np.zeros(len(p))  # Water content in surface (u)
    s_snow = np.zeros(len(p))  # Snow Storage (snow)
    q_snow = np.zeros(len(p))  # Snow melt (qs)
    q_inter = np.zeros(len(p))  # Interflow (q_if)
    e_real = np.zeros(len(p))  # Actual evaporation (e_real)
    q_of = np.zeros(len(p))  # Overland flow (q_of)
    q_g = np.zeros(len(p))  # Recharge (g)
    q_bf = np.zeros(len(p))  # Baseflow (bf)

    for t in range(len(p)):
        # Set boundary conditions
        prec = p[t]
        evap = e[t]
        temp = t[t]

        # Snow storage and snow melt
        if temp < snow_temp:
            snow += prec
        else:
            qs = csnow * temp
            if snow < qs:
                qs = snow
                snow = 0
            else:
                snow -= qs

        # Evapotranspiration module
        if temp < 0:
            u1 = u
        else:
            u1 = u + prec + qs
        if u1 > evap:
            eau = evap
            eal = 0
        else:
            eau = u1
            eal = (evap - eau) * l_frac

        u2 = min(u1 - eau, umax)

        if l_frac > tif:
            q_if = (l_frac - tif) / (1 - tif) * u2 / ck_if
        else:
            q_if = 0

        u3 = u1 - eau - q_if

        if u3 > umax:
            pn = u3 - umax
            u = umax
        else:
            pn = 0
            u = u3

        # Net precipitation
        n = int(pn / p_mm) + 1
        pnlst = pn - (n - 1) * p_mm
        eal /= n

        q_of_sum = 0
        g_sum = 0
        for i in range(1, n + 1):
            pn = p_mm
            if i == n:
                pn = pnlst

            # Overland flow
            if l_frac > tof:
                q_of = cqof * (l_frac - tof) / (1 - tof) * pn
            else:
                q_of = 0

            q_of_sum += q_of

            # Recharge
            if l_frac > tg:
                g = (l_frac - tg) / (1 - tg) * (pn - q_of)
            else:
                g = 0

            g_sum += g

            # Lower zone storage
            dl = pn - q_of - g
            l = l + dl - eal

            if l > lmax:
                g_sum += l - lmax
                l = lmax

            l_frac = l / lmax

        q_of = q_of_sum
        g = g_sum
        eal *= n

        # Baseflow
        c = math.exp(-1. / ck_bf)
        bf = bf * c + g * c_area * (1 - c)

        # Interflow
        c = math.exp(-1. / ck_12)
        if1 = if1 * c + q_if * (1 - c)
        if2 = if2 * c + if1 * (1 - c)

        # Overland flow routing and component
        of = 0.5 * (of1 + of2) / delta_t

        if of > q_of_min:
            ck_qof = ck_12 * (of / q_of_min) ** (-beta)
        else:
            ck_qof = ck_12

        c = math.exp(-1. / ck_qof)

        of1 = of1 * c + q_of * (1 - c)
        of2 = of2 * c + of1 * (1 - c)

        # Update state variables
        states = np.array([snow, u, l, if1, if2, of1, of2, bf])

        # Update simulated values
        if t >= spin_off:
            q_sim[t] = fact * (if2 + of2 + bf)
            l_soil[t] = l_frac
            u_soil[t] = u
            s_snow[t] = snow
            q_snow[t] = qs
            q_inter[t] = q_if
            e_real[t] = eal
            q_of[t] = q_of
            q_g[t] = g
            q_bf[t] = bf

    return q_sim, l_soil, u_soil, s_snow, q_snow, q_inter, e_real, q_of, q_g, q_bf

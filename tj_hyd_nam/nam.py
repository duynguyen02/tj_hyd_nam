import numpy as np
import pandas as pd

from .tj_hyd_nam import NAMConfig


class NAM:
    def __init__(
            self,
            filename: str,
            nam_config: NAMConfig,
            cal: bool,
    ):
        self.Data_file = None
        self.df = None
        self.P = None
        self.T = None
        self.E = None
        self.Qobs = None

        self.flow_rate = nam_config.area / (3.6 * nam_config.interval)
        self.interval = nam_config.interval
        self.initial = np.array(
            [
                nam_config.umax,
                nam_config.lmax,
                nam_config.cqof,
                nam_config.ckif,
                nam_config.ck12,
                nam_config.tof,
                nam_config.tif,
                nam_config.tg,
                nam_config.ckbf,
                nam_config.csnow,
                nam_config.snowtemp,
            ]
        )

        self.Spinoff = 0
        self.parameters = None

        self.States = None
        self.Qsim = None
        self.Lsoil = None
        self.n = None
        self.Date = None
        # Min - Max
        self.bounds = ((10, 20), (100, 300), (0.1, 1), (200, 1000), (10, 50),
                       (0, 0.99), (0, 0.99), (0, 0.99), (1000, 4000), (0, 0), (0, 0))
        self.NSE = None
        self.RMSE = None
        self.PBIAS = None
        self.Cal = cal
        self.statistics = None
        self.export = f'{filename}.nam.csv'
        self.Sm = None
        self.Ssnow = None
        self.Qsnow = None
        self.Qinter = None
        self.Eeal = None
        self.Qof = None
        self.Qg = None
        self.Qbf = None
        self.usoil = None
        self.flowduration = None

    def InitData(self):
        self.P = self.df.P
        self.T = self.df.Temp
        self.E = self.df.E
        self.Qobs = self.df.Q
        self.n = self.df.__len__()
        self.Qsim = np.zeros(self.n)
        self.Lsoil = np.zeros(self.n)
        self.Date = self.df.index

    def nash(self, qobserved, qsimulated):
        if len(qobserved) == len(qsimulated):
            s, e = np.array(qobserved), np.array(qsimulated)
            # s,e=simulation,evaluation
            mean_observed = np.nanmean(e)
            # compute numerator and denominator
            numerator = np.nansum((e - s) ** 2)
            denominator = np.nansum((e - mean_observed) ** 2)
            # compute coefficient
            return 1 - (numerator / denominator)

        else:
            logging.warning(
                "evaluation and simulation lists does not have the same length.")
            return np.nan

    def Objective(self, x):
        self.Qsim, self.Lsoil, self.usoil, self.Ssnow, self.Qsnow, self.Qinter, self.Eeal, self.Qof, self.Qg, self.Qbf = nm.NAM(
            x, self.P, self.T, self.E, self.flow_rate, self.interval, self.Spinoff)
        n = math.sqrt((sum((self.Qsim - self.Qobs) ** 2)) / len(self.Qobs))
        # n = obj.nashsutcliffe(self.Qobs, self.Qsim)
        return n

    def run(self):
        self.InitData()
        if self.Cal == True:
            self.parameters = minimize(self.Objective, self.initial, method='SLSQP', bounds=self.bounds,
                                       options={'maxiter': 1e8, 'disp': True})
            self.Qsim, self.Lsoil, self.usoil, self.Ssnow, self.Qsnow, self.Qinter, self.Eeal, self.Qof, self.Qg, self.Qbf = nm.NAM(
                self.parameters.x, self.P, self.T, self.E, self.flow_rate, self.interval, self.Spinoff)
            print(self.parameters.x)
        else:
            self.Qsim, self.Lsoil, self.usoil, self.Ssnow, self.Qsnow, self.Qinter, self.Eeal, self.Qof, self.Qg, self.Qbf = nm.NAM(
                self.initial, self.P, self.T, self.E, self.flow_rate, self.interval, self.Spinoff)

    def update(self):
        self.df['Qsim'] = self.Qsim
        self.df['Lsoil'] = self.Lsoil
        # TODO: lưu vào TJ_HYD_NAM
        # self.df.to_csv(os.path.join(self.process_path,
        #                             self.export), index=True, header=True)

    def stats(self):
        mean = np.mean(self.Qobs)
        mean2 = np.mean(self.Qsim)
        self.NSE = 1 - (sum((self.Qsim - self.Qobs) ** 2) /
                        sum((self.Qobs - mean) ** 2))
        self.RMSE = np.sqrt(sum((self.Qsim - self.Qobs) ** 2) / len(self.Qsim))
        self.PBIAS = (sum(self.Qobs - self.Qsim) / sum(self.Qobs)) * 100
        self.statistics = obj.calculate_all_functions(self.Qobs, self.Qsim)

    def interpolation(self):
        fit = np.polyfit(self.Qobs, self.Qsim, 1)
        fit_fn = np.poly1d(fit)
        return fit_fn

    def draw(self):
        self.stats()
        fit = self.interpolation()
        Qfit = fit(self.Qobs)
        width = 15  # Figure widthmean2 = np.mean(self.Qsim)
        height = 10  # Figure height
        f = plt.figure(figsize=(width, height))
        widths = [2, 2, 2]
        heights = [2, 3, 1]
        gs = GridSpec(3, 3, figure=f, width_ratios=widths,
                      height_ratios=heights)
        ax1 = f.add_subplot(gs[1, :])
        ax2 = f.add_subplot(gs[0, :], sharex=ax1)
        ax3 = f.add_subplot(gs[-1, 0])
        ax4 = f.add_subplot(gs[-1, -1])
        ax5 = f.add_subplot(gs[-1, -2])
        color = 'tab:blue'
        ax2.set_ylabel('Precipitation ,mm ', color=color,
                       style='italic', fontweight='bold', labelpad=20, fontsize=13)
        ax2.bar(self.Date, self.df.P, color=color,
                align='center', alpha=0.6, width=1)
        ax2.tick_params(axis='y', labelcolor=color)
        # ax2.set_ylim(0, max(self.df.P) * 1.1, )
        ax2.set_ylim(max(self.df.P) * 1.1, 0)
        ax2.legend(['Precipitation'])
        color = 'tab:red'
        ax2.set_title('NAM Simulation', style='italic',
                      fontweight='bold', fontsize=16)
        ax1.set_ylabel(r'Discharge m$^3$/s', color=color,
                       style='italic', fontweight='bold', labelpad=20, fontsize=13)
        ax1.plot(self.Date, self.Qobs, 'b-', self.Date,
                 self.Qsim, 'r--', linewidth=2.0)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis='x', labelrotation=45)
        ax1.set_xlabel('Date', style='italic',
                       fontweight='bold', labelpad=20, fontsize=13)
        ax1.legend(('Observed Run-off', 'Simulated Run-off'), loc=2)
        plt.setp(ax2.get_xticklabels(), visible=False)
        anchored_text = AnchoredText("NSE = %.2f\nRMSE = %0.2f\nPBIAS = %0.2f" % (self.NSE, self.RMSE, self.PBIAS),
                                     loc=1, prop=dict(size=11))
        ax1.add_artist(anchored_text)
        # plt.subplots_adjust(hspace=0.05)
        ax3.set_title('Flow Duration Curve', fontsize=11, style='italic')
        ax3.set_yscale("log")
        ax3.set_ylabel(r'Discharge m$^3$/s', style='italic',
                       fontweight='bold', labelpad=20, fontsize=9)
        ax3.set_xlabel('Percentage Exceedence (%)', style='italic',
                       fontweight='bold', labelpad=20, fontsize=9)
        exceedence, sort, low_percentile, high_percentile = self.flowdur(
            self.Qsim)
        ax3.legend(['Precipitation'])
        ax3.plot(self.flowdur(self.Qsim)[0], self.flowdur(self.Qsim)[
            1], 'b-', self.flowdur(self.Qobs)[0], self.flowdur(self.Qobs)[1], 'r--')
        # ax3.plot(self.flowdur(self.Qobs)[0], self.flowdur(self.Qobs)[1])
        ax3.legend(('Observed', 'Simulated'),
                   loc="upper right", prop=dict(size=7))

        plt.grid(True, which="minor", ls="-")

        st = stats.linregress(self.Qobs, self.Qsim)
        # ax4.set_yscale("log")
        # ax4.set_xscale("log")
        ax4.set_title('Regression Analysis', fontsize=11, style='italic')
        ax4.set_ylabel(r'Simulated', style='italic',
                       fontweight='bold', labelpad=20, fontsize=9)
        ax4.set_xlabel('Observed', style='italic',
                       fontweight='bold', labelpad=20, fontsize=9)
        anchored_text = AnchoredText("y = %.2f\n$R^2$ = %0.2f" % (
            st[0], (st[2]) ** 2), loc=4, prop=dict(size=7))
        # ax4.plot(self.Qobs, fit(self.Qsim), '--k')
        # ax4.scatter(self.Qsim, self.Qobs)
        ax4.plot(self.Qobs, self.Qsim, 'bo', self.Qobs, Qfit, '--k')
        ax4.add_artist(anchored_text)

        self.update()
        dfh = self.df.resample('M').mean()
        Date = dfh.index.to_pydatetime()
        ax5.set_title('Monthly Mean', fontsize=11, style='italic')
        ax5.set_ylabel(r'Discharge m$^3$/s', color=color,
                       style='italic', fontweight='bold', labelpad=20, fontsize=9)
        # ax5.set_xlabel('Date', style='italic', fontweight='bold', labelpad=20, fontsize=9)
        ax5.tick_params(axis='y', labelcolor=color)
        ax5.tick_params(axis='x', labelrotation=45)
        # ax5.set_xlabel('Date', style='italic', fontweight='bold', labelpad=20, fontsize=9)
        ax5.legend(('Observed', 'Simulated'), loc="upper right")
        exceedence, sort, low_percentile, high_percentile = self.flowdur(
            self.Qsim)
        ax5.tick_params(axis='x', labelsize=9)
        ax5.plot(Date, dfh.Q, 'b-', Date, dfh.Qsim, 'r--', linewidth=2.0)
        ax5.legend(('Observed', 'Simulated'), prop={'size': 7}, loc=1)
        # ax5.plot(dfh.Q)
        # ax5.plot(dfh.Qsim)
        # ax5.legend()
        plt.grid(True, which="minor", ls="-")

        plt.subplots_adjust(hspace=0.03)
        f.tight_layout()
        plt.show()

    def flowdur(self, x):
        exceedence = np.arange(1., len(np.array(x)) + 1) / len(np.array(x))
        exceedence *= 100
        sort = np.sort(x, axis=0)[::-1]
        low_percentile = np.percentile(sort, 5, axis=0)
        high_percentile = np.percentile(sort, 95, axis=0)
        return exceedence, sort, low_percentile, high_percentile

    def drawflow(self):
        f = plt.figure(figsize=(15, 10))
        ax = f.add_subplot(111)
        # fig, ax = plt.subplots(1, 1)
        ax.set_yscale("log")
        ax.set_ylabel(r'Discharge m$^3$/s', style='italic',
                      fontweight='bold', labelpad=20, fontsize=13)
        ax.set_xlabel('Percentage Exceedence (%)', style='italic',
                      fontweight='bold', labelpad=20, fontsize=13)
        exceedence, sort, low_percentile, high_percentile = self.flowdur(
            self.Qsim)
        ax.plot(self.flowdur(self.Qsim)[0], self.flowdur(self.Qsim)[1])
        ax.plot(self.flowdur(self.Qobs)[0], self.flowdur(self.Qobs)[1])
        plt.grid(True, which="minor", ls="-")
        # ax.fill_between(exceedence, low_percentile, high_percentile)
        # plt.show()
        return ax

    def drawscatter(self):
        f = plt.figure(figsize=(15, 10))
        ax = f.add_subplot(111)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylabel(r'Discharge m$^3$/s', style='italic',
                      fontweight='bold', labelpad=20, fontsize=13)
        ax.set_xlabel('Percentage Exceedence (%)', style='italic',
                      fontweight='bold', labelpad=20, fontsize=13)
        ax.scatter(self.Qsim, self.Qobs)
        plt.show()

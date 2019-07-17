import pandas as pd
import numpy as np
import datetime
import copy
import random
from lazy import lazy





class CIO_portfolio_v2 (object):
    # v2 is much the same as v1. The difference is
    # 1) v2 takes the manager selection function as input (allowing more flexible selection strategy)
    # 2) improve the dimension exposure tracking and calculation

    def __init__(self,
                 ID,  # string,
                 start_date,  # str, eg '2010-01-01'
                 end_date,  # str, eg '2010-01-01'
                 invest_horizon,  # float, investment horizon in years
                 capital_allocation,
                 # dict {manager_code: initial invested capital (in million
                 # dollars, float)}
                 managerFee_calculator,
                 # obj, calculate management fee based on EID and invested capital
                 # managerFee_calculator.mg_fee(EID, invested_capital ) will be called
                 # it will return a monthly management fee cost as portion of asset market value
                 PM_returns,  # pd df, manager gross of fee returns table
                 PM_Info,       # dict {EID: {'universe': 0 (Core), -1 (Value), 1(Growth),
                                # 'approach': 1(Q), -1(F),
                                # 'te_level': 1(H), -1(L) }}
                 PM_picker,  # object of PM_selector,  PM_picker.initial_select() and .second_select() will be called
                 bm_ret,  # pd Series, benchmark return



                 ):

        self.ID = ID
        self.start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        self.invest_horizon = invest_horizon

        self.initial_capital = float(pd.Series(capital_allocation).sum())
        self.capital_allocation = capital_allocation
        self.benchmark_ret = bm_ret.loc[np.logical_and(
            bm_ret.index >= start_date, bm_ret.index <= end_date)]
        self.cur_manager = {}  # dict {manager_code: EID}
        # dict {date (datetime): {manager_code: EID}}
        self.PMSelection_log = {}
        # dict { manager_code:  invested capital for each manager (not market
        # value)}
        self.cur_investedCapital = None
        # df, col: manager_code, index: date, value: management fee charge
        # monthly amortization rate
        self.mgmFee = None
        # df, col: manager_code, index: date, value: manager gross return
        self.manager_returns_gross = None
        # df, col: manager_code, index: date, value: manager net return
        self.manager_returns_net = None
        # df, col: manager_code, index: date, value: manager accumulated value
        # (mkt value), gross
        self.manager_value_gross = None
        # df, col: manager code, index: date, value: manager accmulated value
        # (mkt value), net
        self.manager_value_net = None
        self.manager_exposure = None
        # df, col: manager_code, row: dimension, value: exposure
        self.allocation_index= None
        # int, the index of selected allocation from the list of feasible allocations.

        # initial PM selection
        PM_pool = list(PM_returns.loc[self.start_date, ].dropna().index)
        self.cur_manager, self.manager_exposure, self.allocation_index = PM_picker.initial_select(
            PM_pool=PM_pool, PM_returns=PM_returns, PM_Info=PM_Info, fwdperiod_start=self.start_date, fwdperiod_end=self.end_date, manager_weight_allocation={
                k: v / self.initial_capital for k, v in self.capital_allocation.items()})

        '''
        PM_picker.initial_select(PM_pool, PM_returns, PM_Info, fwdperiod_start, fwdperiod_end, manager_weight_allocation, cur_manager)

            PM_pool, list, the available PM pool
            PM_returns, df, the PM monthly return dataframe
            PM_Info, dict, {PM EID: {dimension: exposure }}, the exposure of dimensions for each PM
            fwdperiod_start, datetime.datetime, the start of forward looking period
            fwdperiod_end, datetime.datetime, the end of forward looking period
            manager_weight_allocation, {manager code ('PM_i'): percentage of total capital}

        RETURN: dict, {manager code ('PM_i'): manager EID },
                df, col: manager_code, row: dimension, value: exposure,
                int, index of selected allocation in the list of feasible allocations

        '''

        self.PMSelection_log[self.start_date] = copy.deepcopy(self.cur_manager)
        self.manager_returns_gross = pd.DataFrame(
            {k: PM_returns[v] for k, v in self.cur_manager.items()})
        self.manager_returns_gross = self.manager_returns_gross.loc[
            np.logical_and(self.manager_returns_gross.index >= self.start_date,
                           self.manager_returns_gross.index <= self.end_date), ]
        self.mgmFee = (
            self.manager_returns_gross -
            self.manager_returns_gross +
            1).mul(
            pd.Series(
                {
                    k: managerFee_calculator.mg_fee(
                        v,
                        invested_capital=self.capital_allocation[k]) for k,
                    v in self.cur_manager.items()}),
            axis=1)
        self.manager_returns_net = (
            1 + self.manager_returns_gross) * (1 - self.mgmFee) - 1
        self.cur_investedCapital = copy.deepcopy(self.capital_allocation)

        # fill all missing managers
        while self.manager_returns_gross.isna().any(axis=None):
            a = self.manager_returns_gross.isna().any(axis=1)
            cur_date = a[a].index[0]
            cur_date = datetime.datetime(
                cur_date.year, cur_date.month, cur_date.day)
            a = self.manager_returns_gross.loc[cur_date, ].isna()
            missing_managers = list(a[a].index)  # eg., ['PM_2' , 'PM_4']
            cur_managerAccValue = (
                1 + self.manager_returns_net).dropna().prod() * pd.Series(self.capital_allocation)
            PM_pool = list(PM_returns.loc[cur_date, ].dropna().index)
            cur_manager = copy.deepcopy(self.cur_manager)
            cur_manager.update({k: None for k in missing_managers})
            self.cur_manager = PM_picker.second_select(
                PM_pool=PM_pool,
                PM_returns=PM_returns,
                PM_Info=PM_Info,
                fwdperiod_start=cur_date,
                fwdperiod_end=self.end_date,
                cur_manager=cur_manager,
                manager_dimension_allocation=self.manager_exposure)

            '''
            PM_picker.second_select(PM_pool, PM_returns, PM_Info, fwdperiod_start, fwdperiod_end, manager_weight_allocation,
                                    cur_manager, manager_dimesion_allocation)

                PM_pool, list, the available PM pool
                PM_returns, df, the PM monthly return dataframe
                PM_Info, dict, {PM EID: {dimension: exposure }}, the exposure of dimensions for each PM
                fwdperiod_start, datetime.datetime, the start of forward looking period
                fwdperiod_end, datetime.datetime, the end of forward looking period
                manager_weight_allocation, {manager code ('PM_i'): percentage of total capital}
                cur_manager, {manager code ('PM_i'): manager EID }. The missing manager will have EID= None
                manager_dimension_allocation, df, col: manager_code, row: dimension, value: exposure


            RETURN: dict, {manager code ('PM_i'): manager EID }

            '''

            self.PMSelection_log[cur_date] = {
                k: self.cur_manager[k] for k in missing_managers}
            self.cur_investedCapital.update(
                {k: cur_managerAccValue[k] for k in missing_managers})
            new_PM = {k: self.cur_manager[k] for k in missing_managers}
            new_PM_return = pd.DataFrame(
                {k: PM_returns[v] for k, v in new_PM.items()})
            new_PM_return = new_PM_return.loc[np.logical_and(
                new_PM_return.index >= cur_date, new_PM_return.index <= self.end_date), ]
            new_PM_fee = (
                new_PM_return -
                new_PM_return +
                1).mul(
                pd.Series(
                    {
                        k: managerFee_calculator.mg_fee(
                            v,
                            invested_capital=self.cur_investedCapital[k]) for k,
                        v in new_PM.items()}),
                axis=1)
            new_PM_return_net = (1 + new_PM_return) * (1 - new_PM_fee) - 1
            self.manager_returns_gross.update(
                other=new_PM_return, overwrite=False)
            self.mgmFee.update(other=new_PM_fee, overwrite=False)
            self.manager_returns_net.update(
                other=new_PM_return_net, overwrite=False)

        # calculate manager accumulated value
        self.manager_value_gross = (
            self.manager_returns_gross +
            1) .rolling(
            window=self.manager_returns_gross.shape[0] +
            1,
            min_periods=1) .apply(
            lambda x: x.prod()) .mul(
                pd.Series(
                    self.capital_allocation),
            axis=1)

        self.manager_value_net = (
            self.manager_returns_net +
            1) .rolling(
            window=self.manager_returns_net.shape[0] +
            1,
            min_periods=1) .apply(
            lambda x: x.prod()) .mul(
                pd.Series(
                    self.capital_allocation),
            axis=1)

        # calculate portfolio performance metrics
        self.value_gross = self.manager_value_gross.sum(axis=1)
        self.value_net = self.manager_value_net.sum(axis=1)
        self.return_gross = self.value_gross.pct_change().fillna(
            self.value_gross.head(1) / self.initial_capital - 1)
        self.return_net = self.value_net.pct_change().fillna(
            self.value_net.head(1) / self.initial_capital - 1)
        self.returned_capital = float(self.value_net.tail(1))
        self.annualReturn_gross = float((self.value_gross.tail(
            1) / self.initial_capital) ** (1 / self.invest_horizon) - 1)
        self.annualReturn_net = float((self.value_net.tail(
            1) / self.initial_capital) ** (1 / self.invest_horizon) - 1)
        self.annualFee = self.annualReturn_gross - \
            self.annualReturn_net  # annual pct cost
        # annual vol, gross of fee
        self.volatility = float(self.return_gross.std() * (12 ** .5))
        self.benchmark_annualReturn = (
            self.benchmark_ret + 1).prod() ** (1 / self.invest_horizon) - 1
        self.alpha_net = self.annualReturn_net - \
            self.benchmark_annualReturn  # annual alpha, net of fee
        self.alpha_gross = self.annualReturn_gross - \
            self.benchmark_annualReturn  # annual alpha, gross of fee
        self.te = (self.return_gross - self.benchmark_ret).std() * \
            (12 ** .5)  # annual te, gross of fee
        self.IR_net = self.alpha_net / self.te

    def portfolio_exposure(self, manager_exposure):
        '''
        Calculate portfolio exposure using dynamic market value weight, given manager_exposure
        :param manager_exposure: {EID: {dimension: exposure }}
        :return: (portfolio dimension exposure, manager dimension exposure)
                portfolio dimension exposure:
                    df, col: dimension, row: date, value: exposure
                manager dimension exposure:
                    dict: {manager code: df, col: dimension, row: date, value: exposure}
        '''

        manager_record = pd.DataFrame(
            self.PMSelection_log).T.fillna(
            method='ffill')
        manager_exposure_df = pd.DataFrame(manager_exposure).T
        manager_dimension_exposure = {}

        for mg_code in manager_record.columns:
            a = manager_record[mg_code]
            tmp = pd.DataFrame({dim: a.apply(lambda x: None if x is None else manager_exposure_df.loc[x, dim])
                                for dim in manager_exposure_df.columns})\
                .reindex(self.return_gross.index)\
                .sort_index()\
                .fillna(method='ffill')

            manager_dimension_exposure[mg_code] = tmp

        manager_weight = self.manager_value_net.div(
            self.manager_value_net.sum(axis=1), axis=0)

        portfolio_dimension_exposure = 0
        for mg_code, exposure_df in manager_dimension_exposure.items():
            portfolio_dimension_exposure = exposure_df.mul(
                manager_weight[mg_code], axis=0) + portfolio_dimension_exposure

        return portfolio_dimension_exposure, manager_dimension_exposure

    @lazy
    def anchor_ratio_gross(self):
        tmp_s = ((self.return_gross - self.benchmark_ret)
                 > 0).astype(int).astype(str)
        return self.anchor_ratio_helper(''.join(tmp_s.values))

    @lazy
    def anchor_ratio_net(self):
        tmp_s = ((self.return_net - self.benchmark_ret)
                 > 0).astype(int).astype(str)
        return self.anchor_ratio_helper(''.join(tmp_s.values))

    def anchor_ratio_helper(self,
                            tmp_s  # 01 string series, 0 means miss and 1 means hit
                            ):

        hit_runs = []
        miss_runs = []
        self.func_ar(tmp_s, hit_runs=hit_runs, miss_runs=miss_runs)

        hit_scores = np.array([len(x) for x in hit_runs])
        miss_scores = np.array([len(x) for x in miss_runs])

        if len(miss_scores) == 0:
            return None
        else:
            return np.sqrt((hit_scores ** 2).sum() / (miss_scores ** 2).sum())

    def func_ar(self, tmp_s, hit_runs, miss_runs):

        if len(tmp_s) == 0:
            return ''
        elif tmp_s == '1':
            hit_runs.append('1')
            return ''
        elif tmp_s == '0':
            miss_runs.append('0')
            return ''
        else:
            cur_s = tmp_s
            i = 0
            while i < len(cur_s) - 1:
                if cur_s[i] == cur_s[i + 1]:
                    i += 1
                else:
                    break
            if i == len(cur_s) - 1:
                if cur_s[0] == '1':
                    hit_runs.append(cur_s)
                    return ''
                else:
                    miss_runs.append(cur_s)
                    return ''
            else:
                a = cur_s[0: i + 1]
                cur_s = cur_s[i + 1:]
                if a[0] == '1':
                    hit_runs.append(a)
                else:
                    miss_runs.append(a)

                cur_s = self.func_ar(cur_s, hit_runs, miss_runs)


class manager_selector(object):

    '''
    One instance of manager selector with flexible selection rules. Can be inherited.
    '''

    def __init__(self, selection_skill,
                 selection_survival,
                 manager_dimension_scheme,
                 manager_return_filler= None):
        '''

        :param selection_skill: float, 0-inf , the skill param for self.__selection__(). Equal to 0 implies random picking.
        :param selection_survival: bool. True if select manager with survival bias.
        Automatically True if selection_skill>0, i.e. not random picking.
        :param manager_dimension_scheme: dict {k: [df1, df2, ... ]}
        df, col: manager_code, row: dimension, value: exposure
        k, k= count( df.columns)
        :param manager_return_filler: pd df, col: EID, row: date, value: return
        The df to fill NA in manager returns if needed.
        '''
        self.selection_skill = selection_skill
        self.selection_survival = selection_survival
        self.manager_dimension_scheme = manager_dimension_scheme
        self.manager_return_filler= manager_return_filler


    def __selection__(self, N,
                      PM_pool,
                      PM_returns):
        '''

        :param N: int, the number of PM to be selected from the PM_pool
        :param PM_pool: list of EIDs
        :param PM_returns: df, col: EIDs, row, date, value: return MoM

        :return: list of EIDs with length N
        '''

        selected_manager = []
        if self.selection_skill == 0:
            if self.selection_survival:
                return_df = PM_returns[PM_pool].dropna(axis=1)
                selected_manager = np.random.choice(
                    a=list(return_df.columns), size=N, replace=False).tolist()
            else:
                selected_manager = np.random.choice(
                    a=PM_pool, size=N, replace=False).tolist()

        else:

            # calculate probability for each of PM_pool
            if self.selection_survival or ( self.manager_return_filler is None ):
                acc_returns = (PM_returns[PM_pool].dropna(axis=1) + 1).prod(axis=0)
            else:
                manager_returns_fixed= copy.deepcopy(PM_returns[PM_pool])
                manager_returns_fixed.update(other= self.manager_return_filler, overwrite= False)
                acc_returns = (manager_returns_fixed+1).prod(axis=0)

            selection_prob = (acc_returns.rank() /
                              acc_returns.shape[0]) ** self.selection_skill
            selection_prob = selection_prob / selection_prob.sum()
            selected_manager = np.random.choice(
                a=list(acc_returns.index),
                size=N,
                replace=False,
                p=selection_prob).tolist()

        return selected_manager

    def initial_select(self, PM_pool,
                       PM_returns,
                       PM_Info,
                       fwdperiod_start,
                       fwdperiod_end,
                       manager_weight_allocation):
        '''

        :param PM_pool:  list of EIDs
        :param PM_returns: df, col: EID, row: date, value: returns MoM
        :param PM_Info: dict, {EID: {dimension: exposure}}
        :param fwdperiod_start: datetime
        :param fwdperiod_end: datetime
        :param manager_weight_allocation: {manager_code: weight}
        :return: cur_manager, manager_exposure
        cur_manager, {manager_code: EID}
        manager_exposure, df, col: manager_code, row:dimension, value:exposure
        '''

        num_manager = len(manager_weight_allocation.keys())
        a = self.manager_dimension_scheme[num_manager]
        allocation_index= random.randrange(len(a))
        manager_exposure = copy.deepcopy(a[allocation_index])

        cur_manager_type = manager_exposure.sort_index().apply(
            lambda x: '_'.join([str(t) for t in x]), axis=0).to_dict()
        cur_manager_family = {}
        for k, v in cur_manager_type.items():
            if v not in cur_manager_family.keys():
                cur_manager_family[v] = []
            cur_manager_family[v].append(k)
        # cur_manager_type {manager_code: manager exposure str, eg. '-1_0_1'}
        # cur_manager_family {key ,eg. '-1_0_1': [manager_codes]}

        PM_type = pd.DataFrame(PM_Info)[PM_pool].sort_index().apply(
            lambda x: '_'.join([str(t) for t in x]), axis=0).to_dict()
        PM_family = {}
        for k, v in PM_type.items():
            if v not in PM_family.keys():
                PM_family[v] = []
            PM_family[v].append(k)
        # PM_type {EID: PM exposure str, eg. '-1_0_1' }
        # PM_family is a dict {key, eg '-1_0_1': [EIDs]}

        return_df = PM_returns.loc[np.logical_and(
            PM_returns.index >= fwdperiod_start, PM_returns.index <= fwdperiod_end), PM_pool]
        cur_manager = {}

        for k, v in cur_manager_family.items():
            a = self.__selection__(
                N=len(v),
                PM_pool=PM_family[k],
                PM_returns=return_df)
            cur_manager.update({v[i]: a[i] for i in range(len(a))})

        return cur_manager, manager_exposure, allocation_index

    def second_select(self, PM_pool,
                      PM_returns,
                      PM_Info,
                      fwdperiod_start,
                      fwdperiod_end,
                      cur_manager,
                      manager_dimension_allocation):
        '''

        :param PM_pool:  list of EIDs
        :param PM_returns: df, col: EID, row: date, value: returns MoM
        :param PM_Info: dict, {EID: {dimension: exposure}}
        :param fwdperiod_start: datetime
        :param fwdperiod_end: datetime
        :param cur_manager: {manager_code: EID (None if this manager is to be selected)}
        :param manager_dimension_allocation:  df, col: manager_code, row:dimension, value:exposure. The determined manager dimension exposure
        :return: new_manager, dict {manager_code: EID (the missing None is updated)}
        '''

        a = []
        for k, v in cur_manager.items():
            if v is None:
                a.append(k)
        missing_manager_type = manager_dimension_allocation[a].sort_index().apply(
            lambda x: '_'.join([str(t) for t in x]), axis=0).to_dict()
        missing_manager_family = {}
        for k, v in missing_manager_type.items():
            if v not in missing_manager_family.keys():
                missing_manager_family[v] = []
            missing_manager_family[v].append(k)
        # missing_manager_type {manager_code: manager exposure string, eg '-1_0_1'}
        # missing_manager_family {key, eg '-1_0_1': [EIDs]}

        PM_type = pd.DataFrame(PM_Info)[PM_pool].sort_index().apply(
            lambda x: '_'.join([str(t) for t in x]), axis=0).to_dict()
        PM_family = {}
        for k, v in PM_type.items():
            if v not in PM_family.keys():
                PM_family[v] = []
            PM_family[v].append(k)
        # PM_type {EID: PM exposure str, eg. '-1_0_1' }
        # PM_family is a dict {key, eg '-1_0_1': [EIDs]}
        return_df = PM_returns.loc[np.logical_and(
            PM_returns.index >= fwdperiod_start, PM_returns.index <= fwdperiod_end), PM_pool]
        missing_manager = {}
        for k, v in missing_manager_family.items():
            a = self.__selection__(
                N=len(v),
                PM_pool=PM_family[k],
                PM_returns=return_df)
            cur_manager.update({v[i]: a[i] for i in range(len(a))})

        new_manager = copy.deepcopy(cur_manager)
        new_manager.update(missing_manager)

        return new_manager


class managementFee_virtual (object):

    '''
    virtual class of management fee calculator. Must have mg_fee( EID, invested_capital) implemented.
    '''

    def __init__(self):
        None

    def mg_fee(self, EID, invested_capital):
        '''

        :param EID:  string, the EID of managers
        :param invested_capital: float, the base to calculate the management fee
        :return: monthly_fee, float, the monthly management fee cost as portion of invested_capital
        '''

        return None


class managementFee(managementFee_virtual):
    def __init__(self, manager_Info):
        '''
        manager_Info: dict {EID: {dimensions: exp}}
        '''
        self.manager_Info = copy.deepcopy(manager_Info)

    def mg_fee(self, EID, invested_capital):
        '''
        EID: string, manager's EID
        invested_capital: float, the base to calculate mg fee

        RETURN: float, the monthly mg fee cost as a portion of invested_capital
        '''

        return self.__fee_calculation(
            **self.manager_Info[EID],
            invested_capital=invested_capital)

    def __fee_calculation(self, style, approach, te, invested_capital):
        '''
        style=0 (Core), -1 (Value), 1 (Growth)
        approach= -1 (Fundamental), 1 (Quant)
        te= 1 (High), -1 (Low)

        '''
        a1 = 0
        a2 = 0
        a3 = 0
        if style == 0:
            if approach == -1:
                a1 = 55
                a2 = 50
                a3 = 45
            else:
                a1 = 40
                a2 = 38
                a3 = 33
            te_p = 10

            if te == 1:
                a1 += te_p
                a2 += te_p
                a3 += te_p

        elif style == -1:
            if approach == -1:
                a1 = 60
                a2 = 52
                a3 = 46
            else:
                a1 = 45
                a2 = 42
                a3 = 38

            te_p = 15
            if te == 1:
                a1 += te_p
                a2 += te_p
                a3 += te_p

        else:
            if approach == -1:
                a1 = 60
                a2 = 54
                a3 = 48

            else:
                a1 = 52
                a2 = 50
                a3 = 45

            te_p = 15
            if te == 1:
                a1 += te_p
                a2 += te_p
                a3 += te_p

        g = 0
        if invested_capital < 50:
            g = a1 * invested_capital
        elif invested_capital < 150:
            g = a1 * 50 + a2 * (invested_capital - 50)

        else:
            g = a1 * 50 + a2 * 100 + a3 * (invested_capital - 150)

        a = g / 10000 / invested_capital
        return 1 - (1 - a) ** (1 / 12)

from cio_portfolio import *
import pandas as pd
import sys
import pickle
import time

if __name__ == '__main__':

    '''
    Arguments to Parse:
    [input_file,
    D3/D1/Coreonly (str, D3/D1/Coreonly simulation),
    key of skill_set (str, key of skill_sets),
    index of sim_periods (int, 0,1,2,...,6),
    numPM, (int, number of PM),
    numPath, (int, number of paths to generate) 
    output_path, summary of the output result, pd.DataFrame
    outputraw_path, dict of CIO_portfolio_v2 objs. 
    ]
    '''
    t1 = time.time()

    # read args
    input_file= open(sys.argv[1], 'rb')
    input_data= pickle.load(input_file)
    AllManager_Info_D1= input_data['AllManager_Info_D1']
    AllManager_Info_D3= input_data['AllManager_Info_D3']
    AllManager_return=  input_data['AllManager_return']
    AllManager_return_filler= input_data['AllManager_return_filler']
    BM_returns= input_data['BM_returns']
    D1_opposet_scheme= input_data['D1_opposet_scheme']
    D3_opposet_scheme= input_data['D3_opposet_scheme']
    Coreonly_opposet_scheme= input_data['Coreonly_opposet_scheme']
    sim_periods= input_data['sim_periods']
    skill_sets= input_data['skill_sets']
    otherSimSettings= input_data['otherSimSettings']
    inv_horizon = otherSimSettings['inv_horizon']
    capital_initial = otherSimSettings['capital_initial']
    # sim_count = otherSimSettings['sim_count']

    opt_type = sys.argv[2]
    # D3_opt = False
    # if opt_type == 'D3':
    #     D3_opt = True
    b = sys.argv[3]
    skill_set = skill_sets[b]
    sim_period = sim_periods[int(sys.argv[4])]
    numPM = int(sys.argv[5])
    numPath= int( sys.argv[6])
    output_path= sys.argv[7]
    outputraw_path= sys.argv[8]

    sim_scenario_str = opt_type + '+' + \
        b + '+' +\
        sim_period[1] + '+' + \
        sys.argv[5]
    # sim_scenario_str takes form as 'D3+r0+2006-01-01+4'
    capital_allocation = {'PM_' + str(i): 1 / numPM * capital_initial
                          for i in range(1, numPM + 1, 1)}
    PM_Info = None
    if opt_type=='D3':
        PM_Info = AllManager_Info_D3
    else:
        PM_Info = AllManager_Info_D1

    PM_picker = manager_selector(
        selection_skill=skill_set[0],
        selection_survival=skill_set[1],
        manager_dimension_scheme=D3_opposet_scheme if opt_type=='D3' else (
            D1_opposet_scheme if opt_type=='D1' else Coreonly_opposet_scheme),
        manager_return_filler= AllManager_return_filler)
    managementFee_cal= managementFee ( manager_Info= AllManager_Info_D3)
    cio_portfolio_param = {'start_date': sim_period[0],
                           'end_date': sim_period[1],
                           'invest_horizon': inv_horizon,
                           'capital_allocation': capital_allocation,
                           'managerFee_calculator': managementFee_cal,
                           'PM_returns': AllManager_return,
                           'PM_Info': PM_Info,
                           'bm_ret': BM_returns['R1000'],
                           'PM_picker': PM_picker}

    # run the simulation
    result_dict = {sim_scenario_str + '+'+ str(j): CIO_portfolio_v2(ID=sim_scenario_str +'+' +str(j),
                                                                   **cio_portfolio_param) for j in range(numPath)}
    # organize the result df
    result_df = pd.DataFrame({k: {'Alpha': v.alpha_net,
                                  'TE': v.te,
                                  'IR': v.IR_net,
                                  'TR_net': v.annualReturn_net,
                                  'MgFee': v.annualFee,
                                  'count_selection': int(pd.DataFrame(v.PMSelection_log).count().sum()),
                                  'tmp_index': v.allocation_index,
                                  } for k,
                              v in result_dict.items()}).T
    exp_df = pd.DataFrame({k: v.portfolio_exposure(AllManager_Info_D3)[
                          0].mean() for k, v in result_dict.items()}).T
    exp_df.columns = ['exp_' + x for x in exp_df.columns]
    info_df = pd.DataFrame(
        {k: k.split('+') for k in result_df.index},
        index=[
            'sce_optDim',
            'sce_skillSet',
            'sce_periodEnd',
            'sce_numPM',
            'sce_runCount']).T
    result_df = pd.concat([result_df, exp_df, info_df], axis=1).astype(float, errors='ignore')
    result_df['sce_alloIndex']=  result_df['sce_numPM'].astype(int).astype(str)+ '_'+ result_df['tmp_index'].astype(int).astype(str)
    result_df.drop(columns= 'tmp_index', inplace= True)

    print(result_df)
    # print(result_df.dtypes)
    # print(result_df.mean())
    # pickle.dump(result_df, open(output_path, 'wb'))
    result_df.to_csv(output_path)
    if outputraw_path!= 'None':
        pickle.dump(result_dict, open(outputraw_path, 'wb'))
    print('runtime: ',time.time()- t1)




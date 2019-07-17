
import pandas as pd
import pickle
import os
import sys


def fetch_pickledDF(folder_name, resultDF_name, save_memory= True):
    '''
    fetch picked pd.DataFrame from folder folder_name and save the concatenated DF to pickled file resultDF_name
    :param folder_name: str, name of folder to fetch result from
    :param resultDF_name: str, name of pickle file to save the concatenated DF
    :param save_memory: True/False, slow loop if True
    :return:None
    '''

    print('Fetching pickled results from ', folder_name)
    file_list= [folder_name+ '/'+x  for x in os.listdir(folder_name)]
    if save_memory:
        # result_DF= pickle.load(open( file_list.pop(0), 'rb'))
        result_DF= pd.read_csv(file_list.pop(0), index_col= 0)
        for file_name in file_list:
            tmp_df= pd.read_csv(file_name, index_col= 0) # pickle.load( open( file_name, 'rb'))
            result_DF= result_DF.append( tmp_df, sort= False)  # pd.concat([result_DF, tmp_df], axis=0)
            del tmp_df
            print(file_name)

        result_DF.sort_index(inplace= True)
    else:
        result_DF= pd.concat([ pd.read_csv(file_name, index_col=0) for file_name in file_list], axis=0).sort_index()


    print( 'Fetch results successfully. Save consolidated DF to ', resultDF_name)
    pickle.dump(result_DF.astype(float, errors= 'ignore'), open( resultDF_name ,'wb'))

    return None


if __name__ =='__main__':

    folder_name= sys.argv[1]
    resultDF_name= sys.argv[2]

    fetch_pickledDF(folder_name, resultDF_name, save_memory=False)



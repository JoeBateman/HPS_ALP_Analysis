"""
This python file adapts the functions defined in Functions_create_pkl_FHC_RHC.ipynb
for use in create_pkls_nb.ipynb. The goal is to create a pkl that is correctly formatted for
use in Loading_pkls_test.ipynb, which can then be used throughout the main analysis.
"""

import uproot3 as uproot 
import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
import joblib

Main_variables = [
    'run', # Most crucial Selection var
    'sub', # Most crucial Selection var
    'evt', # Most crucial Selection var
    'shr_energy_tot', # An imp BDT variable for KDAR 2shr and KDAR 1shr in training
    'pfnplanehits_Y', # BDT variable
    'contained_fraction', # BDT variable
    'shr_pitch_u_v', # BDT variable
    'trk_energy', # BDT variable
    'contained_sps_ratio', # BDT variable
    'total_hits_y', # BDT variable
    'trk_energy_hits_tot', # BDT variable
    'shr_start_x', # BDT variable
    'nu_pdg', # Use in plotter
    'ccnc', # Used in plotter
    'nu_parent_pdg', # Will be imp in fixing KL flux
    'theta', # beamdump factor to fix flux for FHC and RHC by 8.0 and 8.6 respectively. WRONG VARIABLE FOR BKG
    'reco_nu_vtx_sce_x', # Very first cut and used in distance_bw_two_particles() Selection var
    'reco_nu_vtx_sce_y', # Very first cut and used in distance_bw_two_particles() Selection var
    'reco_nu_vtx_sce_z', # Very first cut and used in distance_bw_two_particles() Selection var
    'npi0', # Used in plotter to scale the pi0 by 0.759 factor from BNB
    'crtveto', # Used to veto cosmics in RHC
    'nu_purity_from_pfp', # Used in Selection_mc() to separate cosmics from signal
    'slpdg', # BDT variable
    'trk_score_v', # Most basic cut to separate showers from tracks
    'topological_score', # This might be important later I think
    'n_showers_contained_MCStool', # Most basic cut used in selecting 2 shr and 1 shr
    'DeltaMed', # Imp variable for KDAR 2shr 
    'NeutrinoEnergy2', # The most important variable in training the BDT.
    'shr_energy_y_v', # used in defining the leading shr
    'shr_pfp_id_v', # Most basic cut to eliminate inexplicable events in the ntuples
    'hits_v', # BDT variable
    'n_tracks', # BDT variable
    #'nu_flashmatch_score', # BDT variable for KDAR 2shr I think <<<<<<<<<<<<<<<<<<<<<<=============================================================
    'pi0_dir1_x', # BDT variable
    'pi0_dir1_z', # BDT variable
    'pi0_energy1_Y', # BDT variable
    'pi0_mass_Y', # BDT variable
    'shrclusdir0', # BDT variable
    'shrclusdir1', # BDT variable
    'shrclusdir2', # BDT variable
    'shr_openangle', # Used in calculating some other variable
    'shr_theta', # BDT variable
    'shr_pca_0', # BDT variable
    'shr_px', # BDT variable
    'shr_px_v', # BDT variable
    'shr_pz', # BDT variable
    'trk_len_v', # An imp variable to use in training the BDT
    'pfnhits', # To select leading shower
    'nslice', # Could be very first cut Selection var
    'shr_py_v', # shr_px_v is imp BDT variable but this is for calculating inv_mass
    'shr_pz_v', # Used in calculating inv_mass
    'shr_start_x_v', # Used in calculating distance_bw_two_particles ()
    'shr_start_y_v', # Used in calculating distance_bw_two_particles ()
    'shr_start_z_v', # Used in calculating distance_bw_two_particles ()
    'shr_dedx_y_v', # Used in calculating the energy asymmetry 
    'nu_e', # To calculate theta_nu
    ]

truth_variables = ['true_nu_vtx_x',
                   'true_nu_vtx_y',
                   'true_nu_vtx_z',
                   'true_nu_pz',
                   ] # To calculate theta_nu


# Separate variables as they are not compatible with flatten=True
mc_variables = [
    'mc_px',
    'mc_py',
    'mc_pz',
    'mc_pdg'
]

tune_variables = [
    'weightSplineTimesTune', # # Product of genie tune weights and spline weights
    'weightTune', # Genie tune weights
    'ppfx_cv', # Flux weights
    ]

weights_variables = [
    'weightsGenie', # Xsec uncertainty
    'weightsReint', # Reint uncertainty
    'weightsPPFX' # Flux uncertainty
    ]


def load_weights(loc_root, sav_name_pkl, main_input_dir):
    # Adapted from 3_Genie_background.ipynb
    tree_name='nuselection/NeutrinoSelectionFilter'
    
    if 'dirt' in loc_root:
        vars_to_save = ['run','sub','evt','nslice', 'ppfx_cv','weightsGenie','weightsReint']
        #vars_to_save = ['run','sub','evt']
    else:
        vars_to_save = ['run','sub','evt','nslice', 'ppfx_cv', 'weightsGenie','weightsPPFX','weightsReint']
        #vars_to_save = ['run','sub','evt','weightsPPFX']


    df_ovl = uproot.open(loc_root)[tree_name].pandas.df(vars_to_save,flatten=False)
    
    print ('Started storing dataframe')
    df_ovl.to_pickle(main_input_dir+sav_name_pkl)
    print('RSE and weights saved to ', main_input_dir+sav_name_pkl)
    return df_ovl


def weightTuneBool (df_cut):

    df_cut['weightSplineTimesTune'] = df_cut['weightSplineTimesTune'].replace(np.nan, 1.0)    
    df_cut.loc[ df_cut['weightSplineTimesTune'] <= 0, 'weightSplineTimesTune' ] = 1.
    df_cut.loc[ df_cut['weightSplineTimesTune'] == np.inf, 'weightSplineTimesTune' ] = 1.
    df_cut.loc[ df_cut['weightSplineTimesTune'] > 50, 'weightSplineTimesTune' ] = 1.
    df_cut.loc[ np.isnan(df_cut['weightSplineTimesTune']) == True, 'weightSplineTimesTune' ] = 1.
    
    return df_cut

# These functions are used for the preselection of the 1shr and 2shr samples, and identifying the leading shower
def preselection_1shr(df, print_bool = False):
    # df = df.loc[df['nslice'] == 1]
    
    # in_fv_query = "10<=reco_nu_vtx_sce_x<=246 and -106<=reco_nu_vtx_sce_y<=106 and 10<=reco_nu_vtx_sce_z<=1026"
    # #df = df.query(in_fv_query+' and nu_purity_from_pfp>0.5')
    
    # df = df.loc[df['crtveto'] != 1]
    
    # df = df.query('trk_score_v>=0 and trk_score_v<0.5 and shr_pfp_id_v<10000 and n_showers_contained_MCStool == 1', engine='python')#.index.unique(level='subentry')
    if print_bool==True:
        print('Before slice ID cut')
        print(len(df['run']))
    df = df.loc[df['nslice'] == 1]
    
    if print_bool==True:
        print('Before FV cut')
        print(len(df['run']))
    in_fv_query = "10<=reco_nu_vtx_sce_x<=246 and -106<=reco_nu_vtx_sce_y<=106 and 10<=reco_nu_vtx_sce_z<=1026"
    df = df.query(in_fv_query)    
    if print_bool==True:
        print('After FV cut')
        print(len(df['run']))
    df.query('nu_purity_from_pfp>0.5')
    if print_bool==True:
        print('After nu_purity_from_pfp cut')
        print(len(df['run']))
    df = df.loc[df['crtveto'] != 1]
    if print_bool==True:
        print('After CRT cut')
        print(len(df['run']))
    df = df.query('trk_score_v>=0 and trk_score_v<0.5', engine='python')#.index.unique(level='subentry')
    if print_bool==True:
        print('After trk_score cut')
        print(len(df['run']))
    df = df.query('shr_pfp_id_v<10000', engine='python')
    if print_bool==True:
        print('After shr_pfp_id_v cut')
        print(len(df['run']))
    df = df.query('n_showers_contained_MCStool == 1', engine='python')
    if print_bool==True:
        print('After n_showers_contained_MCStool cut')
        print(len(df['run']))

    df["temp_ID"] = df["run"].apply(str) +"_"+ df["sub"].apply(str) +"_"+ df["evt"].apply(str)

    df['Counts'] = df.groupby(['entry'])['temp_ID'].transform('count')

    df = df.query('Counts==1')
    if print_bool==True:
        print('After Counts==1 cut')
        print(len(df['run']))
    return df

def preselection_2shr(df, print_bool = False):
    df = df.loc[df['nslice'] == 1]
    in_fv_query = "10<=reco_nu_vtx_sce_x<=246 and -106<=reco_nu_vtx_sce_y<=106 and 10<=reco_nu_vtx_sce_z<=1026"
    df = df.query(in_fv_query)    
    if print_bool==True:
        print('After FV cut')
        print(len(df['run']))
    df.query('nu_purity_from_pfp>0.5')
    if print_bool==True:
        print('After nu_purity_from_pfp cut') 
        print(len(df['run'])) 
    if print_bool==True:
        print('After FV cut')
        print(len(df['run']))
    df = df.loc[df['crtveto'] != 1]
    if print_bool==True:
        print('After CRT cut')
        print(len(df['run']))
    df = df.query('trk_score_v>=0 and trk_score_v<0.5', engine='python') #.index.unique(level='subentry')
    if print_bool==True:
        print('After trk_score cut')
        print(len(df['run']))
    df = df.query('shr_pfp_id_v<10000', engine='python')
    if print_bool==True:
        print('After shr_pfp_id_v cut')
        print(len(df['run']))
    df = df.query('n_showers_contained_MCStool == 2', engine='python')
    if print_bool==True:
        print('After n_showers_contained_MCStool cut')
        print(len(df['run']))

    df["temp_ID"] = df["run"].apply(str) +"_"+ df["sub"].apply(str) +"_"+ df["evt"].apply(str)

    df['Counts'] = df.groupby(['entry'])['temp_ID'].transform('count')

    df = df.query('Counts==2')
    if print_bool==True:
        print('After Counts==2 cut')
        print(len(df['run']))

    
    return df

def leading_shr (df):

    df['leading_shr_hits'] = df['pfnhits'].groupby("entry").transform(max) == df['pfnhits']
    df['leading_shr_energy'] = df['shr_energy_y_v'].groupby("entry").transform(max) == df['shr_energy_y_v']
    df.eval('leading_shr = leading_shr_hits',inplace=True)
    
    
    
    
    print (df.query('leading_shr == True')['temp_ID'].nunique())
    print (df.query('leading_shr == True')['temp_ID'].count())
    print (df.query('leading_shr == False')['temp_ID'].nunique())
    print (df.query('leading_shr == False')['temp_ID'].count())
    # Needed because indices are repeated and causes issues with df.update later on!
    df.reset_index(drop=True, inplace=True)
    
    df_temp = df.copy()
    
    # df_temp is a df that contain events that are leading shr defined by leading shr hits
    # This will make it one row per event with an exception for events with both being true
    df_temp.query('leading_shr_hits == True',inplace=True)
    
    # On same temp_ID if there are 2 events with both being leading_shr due to them having same hits then
    # 
    df_temp_temp = df_temp[df_temp.duplicated(subset=['temp_ID'],keep=False)].copy()
    
    #print ('Fine until here 2')
    
    
    
    df_temp_temp['leading_shr'] = df_temp_temp['leading_shr_energy']
    #print ('Fine until here 2.5')
    #display(df_temp_temp.head(200))
    print('df')
    print(len(df['run']))
    print('df_temp_temp')
    print(len(df_temp_temp['run']))

    df.update(df_temp_temp)

    #print ('Fine until here 3')
    
    print (df.query('leading_shr == True')['temp_ID'].nunique())
    print (df.query('leading_shr == True')['temp_ID'].count())
    print (df.query('leading_shr == False')['temp_ID'].nunique())
    print (df.query('leading_shr == False')['temp_ID'].count())
    return df

# These functions calculate some additional kinematics and define scale factors to be determined later

def apply_bnb_flat_scale(df):
    df['bnb_scale_ls'] = np.where(((df['nu_pdg_ls'] == 14) | (df['nu_pdg_ls'] == -14)) & (df['npi0_ls']>=1), 0.759, 1)
    return df

def get_1shr_variable(df, is_data = 0):
    df = df.add_suffix('_ls')
    df.eval('dist_bw_koto_shower_x_ls = reco_nu_vtx_sce_x_ls - shr_start_x_v_ls',inplace = True)
    df.eval('dist_bw_koto_shower_y_ls = reco_nu_vtx_sce_y_ls - shr_start_y_v_ls',inplace = True)
    df.eval('dist_bw_koto_shower_z_ls = reco_nu_vtx_sce_z_ls - shr_start_z_v_ls',inplace = True)
    df.eval('dist_bw_koto_shower_ls = sqrt(dist_bw_koto_shower_x_ls**2 + dist_bw_koto_shower_y_ls**2 + dist_bw_koto_shower_z_ls**2)',inplace = True)
    df['shr_energy_y_v_corrected_ls']=np.where(df['shr_energy_y_v_ls']<=300,df.eval('shr_energy_y_v_ls/.74'),df.eval('shr_energy_y_v_ls/.83'))
    df['bnb_scale_ls'] = 1
    df['KDAR_scale'] = 1
    
    if is_data == 0:
        df = apply_bnb_flat_scale(df)
    
    return df

def get_2shr_variable(df, is_data = 0):
    
    df_new = row_per_event(df.copy())
    df_new = inv_mass(df_new)
    df_new = distance_bw_two_particles(df_new)
    df_new = energy_asymm(df_new)
    
    df_new['bnb_scale_ls'] = 1
    df_new['KDAR_scale'] = 1
    
    if is_data == 0:
        df_new = apply_bnb_flat_scale(df_new)
    
    return df_new

# Here we define the functions that are used in the get_2shr_variable function
def row_per_event(df):
    df_ls = df[df['leading_shr']==True].copy()

    df_sls = df[df['leading_shr']==False].copy()

    df_new = pd.merge(df_ls, df_sls, left_on='temp_ID',right_on='temp_ID',how='inner',suffixes=('_ls','_sls')).copy()

    return df_new

def inv_mass(df):
    df['shr_energy_y_v_corrected_ls']=np.where(df['shr_energy_y_v_ls']<=300,df.eval('shr_energy_y_v_ls/.74'),df.eval('shr_energy_y_v_ls/.83'))
    df['shr_energy_y_v_corrected_sls']=np.where(df['shr_energy_y_v_sls']<=300,df.eval('shr_energy_y_v_sls/.74'),df.eval('shr_energy_y_v_sls/.83'))


    df['recoDotProduct'] = df['shr_px_v_ls'].mul(df['shr_px_v_sls']) + df['shr_py_v_ls'].mul(df['shr_py_v_sls'])+df['shr_pz_v_ls'].mul(df['shr_pz_v_sls'])
    df['prod_shr_E_y'] = df['shr_energy_y_v_ls'].mul(df['shr_energy_y_v_sls'].values)

    df['prod_shr_E_y_corrected'] = df['shr_energy_y_v_corrected_ls'].mul(df['shr_energy_y_v_corrected_sls'].values)

    df.eval('recoOpeningAngle = arccos(recoDotProduct)',inplace=True)

    df.eval('inv_mass_corrected = sqrt(2*prod_shr_E_y_corrected*(1-recoDotProduct))',inplace=True)

    return df

# Two particles because we have koto to shower and shower to shower distance                                                                                                                
def distance_bw_two_particles(df):
    df['dist_bw_showers_x'] = df['shr_start_x_v_ls'].sub(df['shr_start_x_v_sls'].values)
    df['dist_bw_showers_y'] = df['shr_start_y_v_ls'].sub(df['shr_start_y_v_sls'].values)
    df['dist_bw_showers_z'] = df['shr_start_z_v_ls'].sub(df['shr_start_z_v_sls'].values)
    df.eval('dist_bw_showers = sqrt(dist_bw_showers_x**2 + dist_bw_showers_y**2 + dist_bw_showers_z**2)',inplace=True)

    """
    # reco_nu_vtx_sce_x_ls = reco_nu_vtx_sce_x_sls = reco_nu_vtx_sce_x                                                                                                                      

    df.eval('dist_bw_koto_shower_x_ls = reco_nu_vtx_sce_x_ls - shr_start_x_v_ls',inplace = True)
    df.eval('dist_bw_koto_shower_y_ls = reco_nu_vtx_sce_y_ls - shr_start_y_v_ls',inplace = True)
    df.eval('dist_bw_koto_shower_z_ls = reco_nu_vtx_sce_z_ls - shr_start_z_v_ls',inplace = True)
    df.eval('dist_bw_koto_shower_ls = sqrt(dist_bw_koto_shower_x_ls**2 + dist_bw_koto_shower_y_ls**2 + dist_bw_koto_shower_z_ls**2)',inplace = True)

    """
    df.eval('dist_bw_koto_shower_x_sls = reco_nu_vtx_sce_x_sls - shr_start_x_v_sls',inplace = True)
    df.eval('dist_bw_koto_shower_y_sls = reco_nu_vtx_sce_y_sls - shr_start_y_v_sls',inplace = True)
    df.eval('dist_bw_koto_shower_z_sls = reco_nu_vtx_sce_z_sls - shr_start_z_v_sls',inplace = True)
    df.eval('dist_bw_koto_shower_sls = sqrt(dist_bw_koto_shower_x_sls**2 + dist_bw_koto_shower_y_sls**2 + dist_bw_koto_shower_z_sls**2)',inplace = True)


    return df

def energy_asymm(df):
    df.eval('dEdx_asymm = (shr_dedx_y_v_ls - shr_dedx_y_v_sls)/(shr_dedx_y_v_ls + shr_dedx_y_v_sls)',inplace=True)
    df.eval('Energy_asymm = (shr_energy_y_v_ls - shr_energy_y_v_sls)/(shr_energy_y_v_ls + shr_energy_y_v_sls)',inplace=True)

    return df

# These functions calculates the KDAR_scale parameter for background and signal separately
def merge_true_theta_bkg (df_new, Run):
    # This is valid only for background
    # NOT VALID FOR SIGNAL
    
    # query_beamdump = 'theta_nu>1.8 and nu_e_ls > 0.2354 and nu_e_ls < 0.2356'
    
    
    # Pre-selection cuts has the following but still I will add it just to be safe
    df_new['KDAR_scale'] = 1
    
    df_new['theta_nu'] = np.arccos(df_new['true_nu_pz_ls']/df_new['nu_e_ls'])
    
    
    if Run=='Run1':
        KDAR_factor = 8.0
        df_new.loc[(df_new['theta_nu']>=2.14) & (df_new['nu_e_ls']>=0.2354) & (df_new['nu_e_ls']<=0.2356), ['KDAR_scale']] *= KDAR_factor
        
    elif Run=='Run3':
        KDAR_factor = 8.6
        df_new.loc[(df_new['theta_nu']>=2.14) & (df_new['nu_e_ls']>=0.2354) & (df_new['nu_e_ls']<=0.2356), ['KDAR_scale']] *= KDAR_factor
    else:
        print ('SOMETHING IS WRONG')
        return 0
    
    return df_new

def merge_true_theta_sig (df_new, Run):
    # Run this only for KDAR events and ignore this for KDIF signal
    # This is valid only for signal
    # NOT VALID FOR BACKGROUND
    
    
    # Pre-selection cuts has the following but still I will add it just to be safe
    df_new['KDAR_scale'] = 1
    
    
    if Run=='Run1':
        KDAR_factor = 8.0
        df_new.loc[(df_new['theta_ls']>=2.14), ['KDAR_scale']] *= KDAR_factor
        
    elif Run=='Run3':
        KDAR_factor = 8.6
        df_new.loc[(df_new['theta_ls']>=2.14), ['KDAR_scale']] *= KDAR_factor
    else:
        print ('SOMETHING IS WRONG')
        return 0
    return df_new

# These functions are used to tag the different background types (overlay, dirt, cosmic/EXT)
def tag_bkg_category (df, category):
    df_temp = df.copy()
    df_temp['bkg_category'] = category
    return df_temp

# We can use this function to merge our weights and BDT prediction pkls, ready to be input into the analysis code. 
def merge_pred(df_final_test_df, df_weights):
    # df_final_test_df contains signal, overlay, dirt and ext.
    # df_weights only contains overlay and dirt.
    # Split the background into three.
    print ('Length of original final_test_df...: '+str(len(df_final_test_df)))
    print ('Length of original df_weights...: '+str(len(df_weights)))
    
    df_mc = df_final_test_df.query('bkg_category=="o" or bkg_category=="d"')
    print ('Length of df_mc...: '+str(len(df_mc)))
    #display(df_mc)
    
    df_ext = df_final_test_df.query('bkg_category=="e"')
    print ('Length of df_ext...: '+str(len(df_ext)))
    
    # Create temp_ID for both.
    df_mc["temp_ID"] = df_mc["run_ls"].astype(int).apply(str) +"_"+ df_mc["sub_ls"].astype(int).apply(str) +"_"+ df_mc["evt_ls"].astype(int).apply(str)
    df_ext["temp_ID"] = df_ext["run_ls"].astype(int).apply(str) +"_"+ df_ext["sub_ls"].astype(int).apply(str) +"_"+ df_ext["evt_ls"].astype(int).apply(str)
    df_weights["temp_ID"] = df_weights["run"].apply(str) +"_"+ df_weights["sub"].apply(str) +"_"+ df_weights["evt"].apply(str)
    
    df_mc_weights = pd.merge(df_mc, df_weights,  how='left', left_on=['temp_ID'], right_on = ['temp_ID'])
    
    if len(df_mc_weights)==len(df_mc):
        print ('GOOD!!! df_mc_weights has same length as df_mc')
    
    df_final = pd.concat([df_mc_weights, df_ext])
    
    if len(df_final_test_df.query('bkg_category=="o" or bkg_category=="d" or bkg_category=="e"'))==len(df_final):
        print ('Amazing')
        return df_final
    else:
        print ('ERROR SOMETHING IS WRONG...:')
        return 0
    
# The selection cut function defined by Aditya in defined in Load_booster.ipynb 
# and updated in import_BDTs.ipynb
def Selection_mc(df_dataframe,weightTune=True,signalbool=False,print_bool = False, Run3 = False):
    
    if weightTune == False:
        in_fv_query = "10<=reco_nu_vtx_sce_x<=246 and -106<=reco_nu_vtx_sce_y<=106 and 10<=reco_nu_vtx_sce_z<=1026"
        out_fv_query = "((reco_nu_vtx_sce_x<10 or reco_nu_vtx_sce_x>246) or (reco_nu_vtx_sce_y<-106 or reco_nu_vtx_sce_y>106) or (reco_nu_vtx_sce_z<10 or reco_nu_vtx_sce_z>1026))"
        
    else:
        in_fv_query = "10<=true_nu_vtx_x<=246 and -106<=true_nu_vtx_y<=106 and 10<=true_nu_vtx_z<=1026"
        out_fv_query = "((true_nu_vtx_x<10 or true_nu_vtx_x>246) or (true_nu_vtx_y<-106 or true_nu_vtx_y>106) or (true_nu_vtx_z<10 or true_nu_vtx_z>1026))"

    if signalbool == True:
        in_fv_query = "10<=true_nu_vtx_x<=246 and -106<=true_nu_vtx_y<=106 and 10<=true_nu_vtx_z<=1026"
        out_fv_query = "((true_nu_vtx_x<10 or true_nu_vtx_x>246) or (true_nu_vtx_y<-106 or true_nu_vtx_y>106) or (true_nu_vtx_z<10 or true_nu_vtx_z>1026))"
        
        
    infv = df_dataframe.query(in_fv_query+' and nu_purity_from_pfp>0.5')
    cosmic = df_dataframe.query(in_fv_query+' and nu_purity_from_pfp<=0.5')
    outfv = df_dataframe.query(out_fv_query)
    
    
    
    # check that everything is accounted for 
    if len(df_dataframe)==len(infv)+len(cosmic)+len(outfv):
        if print_bool==True:
            print ('Number of COSMICS ' + str(len(cosmic)))
            print ('Number of Interactions INSIDE FV ' + str(len(infv)))
            print ('Number of Interactions OUTSIDE FV ' + str(len(outfv)))
            print ('Total in DATAFRAME ' + str(len(df_dataframe)))
            print ('Sum of FV + Cosmics + OutFV adds to the MC..')
    else:
        if print_bool==True:
            print ('Number of COSMICS ' + str(len(cosmic)))
            print ('Number of Interactions INSIDE FV ' + str(len(infv)))
            print ('Number of Interactions OUTSIDE FV ' + str(len(outfv)))
            print ('Total in DATAFRAME ' + str(len(df_dataframe)))
            print ('Sum of FV + Cosmics + OutFV DOES NOT add to the MC..')
            print ('Something is wrong...cosmics/outFV/inFV not summing to the original dataframe (Ignore this for Ext and On-Beam)')
            return 0
    
    df_dataframe = infv.copy()
    
    if print_bool==True:
        print ('Before Slice ID cut ' + str(df_dataframe['nslice'].count()))
    df_dataframe = df_dataframe.loc[df_dataframe['nslice'] == 1]
    if print_bool==True:
        print ('After Slice ID cut ' + str(df_dataframe['nslice'].count()))

    if Run3 == True:
        df_dataframe = df_dataframe.loc[df_dataframe['crtveto'] != 1]
        if print_bool==True:
            print ('After CRT VETO cut ' + str(df_dataframe['crtveto'].count()))

    #df_dataframe = df_dataframe.loc[df_dataframe['shr_pfp_id_v'] <= 1000]
    #df_dataframe.query('trk_score_v<0.5 and trk_score_v>=0.0',inplace=True)
    
    df_dataframe = df_dataframe.loc[df_dataframe['trk_score_v'] <= 0.5]
    df_dataframe = df_dataframe.loc[df_dataframe['trk_score_v'] >= 0.0]
    if print_bool==True:
        print ('After Track score cut ' + str(df_dataframe['nslice'].count()))

    df_dataframe.query('n_showers_contained_MCStool==2',inplace=True)
    if print_bool==True:
        print ('After n_showers_contained_MCStool cut ' + str(df_dataframe['nslice'].count()))

    # When will this be not the case. Only when I have the signal cuz it has overlays.
    if weightTune == True:
        print ('Setting weightTune right')
        df_dataframe['weightTune'] = df_dataframe['weightTune'].fillna(1.0)
        df_dataframe.loc[df_dataframe.weightTune > 50, 'weightTune'] = 1.0
        if print_bool==True:
            print (df_dataframe['weightTune'].count())
        print('Setting weightSplineTimesTune right')
        df_dataframe = weightTuneBool(df_dataframe)
        if print_bool==True:
            print (df_dataframe['weightSplineTimesTune'].count())
        

    # The following is increasing the no. of events. It is as if we are organising the data more correctly by having the following cut.
    # The weird part is that it is chopping just 6 events. How could 6 events give a significant different from 15717 to 14020.
    # Having a cut increases the number of events than having no cut at all WHAT??????????
    # Maybe I would get even more number of events  
    #df_dataframe['shr_pfp_id_v'] = df_dataframe['shr_pfp_id_v'].astype(np.uint64)


    df_dataframe = df_dataframe.loc[df_dataframe['shr_pfp_id_v'] <= 1000]
    if print_bool==True:
        print ('After shr_pfp_id_v cut ' + str(df_dataframe['nslice'].count()))

    df_dataframe["temp_ID"] = df_dataframe["run"].apply(str) +"_"+ df_dataframe["sub"].apply(str) +"_"+ df_dataframe["evt"].apply(str)
    # I think the following removes the single event and keep those that are >=2
    df_dataframe = df_dataframe[df_dataframe.duplicated(['run','sub','evt'], keep=False)]

    df_dataframe['Counts'] = df_dataframe.groupby(['entry'])['temp_ID'].transform('count')
    

    #df_dataframe = df_dataframe.loc[df_dataframe['Counts'] == 2]
    #df_dataframe.query('Counts<=2',inplace=True)
    df_dataframe = df_dataframe.loc[df_dataframe['Counts'] <= 2]
    if print_bool==True:
        print ('After counts<=2 cut................ ' + str(df_dataframe['nslice'].count()))

    # df_dataframe = df_dataframe.loc[df_dataframe['Counts'] == 2]
    # if print_bool==True:
    #     print ('After counts==2 cut................ ' + str(df_dataframe['nslice'].count()))

    df_dataframe['leading_shr_hits'] = df_dataframe['pfnhits'].groupby("entry").transform(max) == df_dataframe['pfnhits']

    df_dataframe['leading_shr_energy'] = df_dataframe['shr_energy_y_v'].groupby("entry").transform(max) == df_dataframe['shr_energy_y_v']

    #df_dataframe['leading_shr'] = np.logical_and(df_dataframe['leading_shr_hits'], df_dataframe['leading_shr_energy'])
    df_dataframe.eval('leading_shr = leading_shr_hits',inplace=True)

    #df_dataframe.query('sub == 200')

    #df_dataframe["temp_ID"] = df_dataframe["run"].apply(str) +"_"+ df_dataframe["sub"].apply(str) +"_"+ df_dataframe["evt"].apply(str)
    if print_bool==True:
        print (df_dataframe.query('leading_shr == True')['temp_ID'].nunique())
        print (df_dataframe.query('leading_shr == True')['temp_ID'].count())
        print (df_dataframe.query('leading_shr == False')['temp_ID'].nunique())
        print (df_dataframe.query('leading_shr == False')['temp_ID'].count())


    df_temp = df_dataframe.copy()

    # There must be many true than false as the code is thinking both are leading shr so more trues
    df_temp.query('leading_shr_hits == True',inplace=True)

    # Keep the ones that have both the values == True
    df_temp_temp = df_temp[df_temp.duplicated(subset=['temp_ID'],keep=False)].copy()
    #display (df_temp_temp)
    if print_bool==True:
        print ('Before is above')
    # These two true events must have the same leading_shr_energy for run 3 signal that's why an error
    
    if Run3 == True:
        df_dataframe = df_dataframe[~df_dataframe.temp_ID.isin(df_temp_temp.temp_ID.values)]
    else:
        df_temp_temp.eval('leading_shr = leading_shr_energy',inplace=True)
        #display (df_temp_temp)
        # We usually don't apply the leading shower query and set it inplace to true.
        # We have done this for temp query and seeing what are some of the temp_ids with two true for same event
        df_dataframe.update(df_temp_temp)
    df_dataframe = df_dataframe.drop_duplicates(subset=['temp_ID', 'trk_score_v','shr_px_v','pfnhits'], keep='first')

    if print_bool==True:
        print('leading shower')
        print (df_dataframe.query('leading_shr == True')['temp_ID'].nunique())
        print (df_dataframe.query('leading_shr == True')['temp_ID'].count())
        print (df_dataframe.query('leading_shr == False')['temp_ID'].nunique())
        print (df_dataframe.query('leading_shr == False')['temp_ID'].count())



    df_dataframe_ls = df_dataframe[df_dataframe['leading_shr']==True].copy()

    df_dataframe_sls = df_dataframe[df_dataframe['leading_shr']==False].copy()


    result_df_ls = pd.DataFrame(df_dataframe_ls['evt'].values - df_dataframe_sls['evt'].values)

    if print_bool==True:
        print(result_df_ls.count())
        print(result_df_ls.nunique())

    df_dataframe.index = range(len(df_dataframe['run']))
    return df_dataframe

# BDT prediction function, given the BDT model and a pandas dataframe. Adapted from Aditya's code.
def booster_pred_df(booster, df):

    df_untrained = df.copy()

    df_untrained_even = df_untrained[df_untrained['leading_shr']==True]
    df_untrained_odd = df_untrained[df_untrained['leading_shr']==False]

    # # I'm not sure this is necessary, commenting it out to see if everything works without it

    # df_untrained['recoDotProduct'] = df_untrained_even['shr_px_v'].mul(df_untrained_odd['shr_px_v'].values) + df_untrained_even['shr_py_v'].mul(df_untrained_odd['shr_py_v'].values) + df_untrained_even['shr_pz_v'].mul(df_untrained_odd['shr_pz_v'].values)
    # df_untrained['prod_shr_E_y'] = df_untrained_even['shr_energy_y_v'].mul(df_untrained_odd['shr_energy_y_v'].values)
    # df_untrained.eval('recoOpeningAngle = arccos(recoDotProduct)',inplace=True)
    # df_untrained.eval('inv_mass = sqrt((2/(0.802*0.802))*prod_shr_E_y*(1-recoDotProduct))',inplace=True)
    # df_untrained['dist_bw_showers_x'] = df_untrained_even['shr_start_x_v'].sub(df_untrained_odd['shr_start_x_v'].values)
    # df_untrained['dist_bw_showers_y'] = df_untrained_even['shr_start_y_v'].sub(df_untrained_odd['shr_start_y_v'].values)
    # df_untrained['dist_bw_showers_z'] = df_untrained_even['shr_start_z_v'].sub(df_untrained_odd['shr_start_z_v'].values)
    # df_untrained.eval('dist_bw_showers = sqrt(dist_bw_showers_x**2 + dist_bw_showers_y**2 + dist_bw_showers_z**2)',inplace=True)
    # df_untrained.eval('dist_bw_koto_shower_x = reco_nu_vtx_sce_x - shr_start_x_v',inplace = True)
    # df_untrained.eval('dist_bw_koto_shower_y = reco_nu_vtx_sce_y - shr_start_y_v',inplace = True)
    # df_untrained.eval('dist_bw_koto_shower_z = reco_nu_vtx_sce_z - shr_start_z_v',inplace = True)
    # df_untrained.eval('dist_bw_koto_shower = sqrt(dist_bw_koto_shower_x**2 + dist_bw_koto_shower_y**2 + dist_bw_koto_shower_z**2)',inplace = True)
    # df_untrained.eval('shr_theta_v_deg = shr_theta_v*(180/3.1415)',inplace=True)
    # df_untrained.eval('shr_phi_v_deg = shr_phi_v*(180/3.1415)',inplace=True)

    merged_df_untrnd_detvar_even_bdt = df_untrained[df_untrained['leading_shr']==True]
    merged_df_untrnd_detvar_odd_bdt = df_untrained[df_untrained['leading_shr']==False]

    df_untrnd_bdt = pd.merge(merged_df_untrnd_detvar_even_bdt, merged_df_untrnd_detvar_odd_bdt, left_on='temp_ID',right_on='temp_ID',how='outer',suffixes=('_ls','_sls')).copy()

    cols_for_booster = booster.feature_names

    untrnd_df = df_untrnd_bdt[cols_for_booster].copy()
    feature_names = untrnd_df.columns.to_list()
    untrnd_final = xgboost.DMatrix(data=untrnd_df[feature_names],missing=-999.0,feature_names=feature_names)
    
    pred = booster.predict(untrnd_final)

    return pred


from scipy.special import logit

def New_loader(Run, mass, nshr, KDIF_KDAR_str):
    """
    New function to load the .pkl files as dataframes. Mostly taken directly from Aditya's code.
    get_tof is a boolean that determines whether to load the time of flight values from the root file.
    """
    #Constants for normalisation factors
    
    mc_pot_fhc = 2.33652e+21 # Need to update this when using new flux
    mc_pot_rhc = 2.644072435807561e+21 # Need to update this when using new flux
    dirt_pot_fhc = 1.42143e+21
    dirt_pot_rhc = 1.03226e+21
    OnBeam_EA9CNT_wcut_fhc = 5268051.0
    OnBeam_EA9CNT_wcut_rhc = 10363728.0
    OffBeam_EXT_NUMIwin_FEMBeamTriggerAlgo_fhc = 9199232.74
    OffBeam_EXT_NUMIwin_FEMBeamTriggerAlgo_rhc = 32878305.25
    scale_ext_fhc = OnBeam_EA9CNT_wcut_fhc/OffBeam_EXT_NUMIwin_FEMBeamTriggerAlgo_fhc
    scale_ext_rhc = OnBeam_EA9CNT_wcut_rhc/OffBeam_EXT_NUMIwin_FEMBeamTriggerAlgo_rhc
    scale_ext_fhc = scale_ext_fhc*.98
    scale_ext_rhc = scale_ext_rhc*.98
    OnBeam_tortgt_wcut_fhc = 2.002e20
    OnBeam_tortgt_wcut_rhc = 5.009e20
    scale_nu_fhc = OnBeam_tortgt_wcut_fhc/mc_pot_fhc
    scale_nu_rhc = OnBeam_tortgt_wcut_rhc/mc_pot_rhc
    scale_dirt_fhc = OnBeam_tortgt_wcut_fhc/dirt_pot_fhc
    scale_dirt_rhc = OnBeam_tortgt_wcut_rhc/dirt_pot_rhc
    scale_dirt_fhc = scale_dirt_fhc*0.75
    scale_dirt_rhc = scale_dirt_rhc*0.35
    
    # # Params for new flux files
    mc_pot_rhc = 2.644072435807561e+21 # PoT for new flux, old flux val = 1.98937e+21
    OnBeam_EA9CNT_wcut_rhc = 1795139.0 # for new flux, old val = 10363728.0
    OffBeam_EXT_NUMIwin_FEMBeamTriggerAlgo_rhc = 5884692.975000 # for new flux, old val = 32878305.25
    scale_ext_rhc = OnBeam_EA9CNT_wcut_rhc/OffBeam_EXT_NUMIwin_FEMBeamTriggerAlgo_rhc
    scale_nu_rhc = OnBeam_tortgt_wcut_rhc/mc_pot_rhc

    if Run == 'Run1':
        fhc_rhc_str = 'fhc'
        OnBeam_tortgt_wcut = OnBeam_tortgt_wcut_fhc
        scale_nu = scale_nu_fhc
        scale_dirt = scale_dirt_fhc
        scale_ext = scale_ext_fhc
    elif Run == 'Run3':
        fhc_rhc_str = 'rhc'
        OnBeam_tortgt_wcut = OnBeam_tortgt_wcut_rhc
        scale_nu = scale_nu_rhc
        scale_dirt = scale_dirt_rhc
        scale_ext = scale_ext_rhc
    elif Run == 'Run4a':
        scale_nu = 1.0 # Need to set this once run 4a sample is ready
    # Path for old flux files
    root_dir = "/exp/uboone/data/users/jbateman/workdir/HPS_uboone_analysis/"
    main_input_dir = root_dir+"/BDT_inputs_pkl/"

    #### OVL ####


    
    loc_BDT_input = '/exp/uboone/data/users/jbateman/workdir/HPS_uboone_analysis/NewFlux/BDT_input_pkl/'
    # df_ovl = pd.read_pickle(main_input_dir+'AllVar_Selected_'+Run+'_NuMI_Overlay_'+KDIF_KDAR_str+'_'+mass+'_'+nshr+'_PPFX_pred_NEW.pkl') # Old overlay files
    df_ovl = pd.read_pickle(loc_BDT_input + f'AllVar_Selected_{Run}_NuMI_Overlay_{KDIF_KDAR_str}_{mass}_MeV_{nshr}_Weights_Updated_Flux_pred.pkl') # New flux overlay
    frac_ovl_test_sample = len(df_ovl.query('is_trained == 0'))/len(df_ovl) #Should be 0.4
    df_ovl['frac_test_sample'] = frac_ovl_test_sample
    df_ovl = df_ovl.query('is_trained==0')
    df_ovl.eval('scale_factor_ls = (1/frac_test_sample)*KDAR_scale*'+str(scale_nu),inplace=True)
    df_ovl['is_good_genie'] = 1
    df_ovl['is_good_ppfx'] = 1
    df_ovl['is_good_reint'] = 1
    
    # These should be pre-calculated! They add ages to the import time
    # df_ovl['thsnd_weight_genie'] = (['yes' if all(a == 1000 for a in i) else 'no' for i in df_ovl['weightsGenie']])
    # df_ovl['thsnd_weight_ppfx'] = (['yes' if all(a == 1000 for a in i) else 'no' for i in df_ovl['weightsPPFX']])
    # df_ovl['thsnd_weight_reint'] = (['yes' if all(a == 1000 for a in i) else 'no' for i in df_ovl['weightsReint']])
    
    # df_ovl['inf_weight_genie'] = (['yes' if any(a>65534 for a in i) else 'no' for i in df_ovl['weightsGenie']])
    # df_ovl['inf_weight_ppfx'] = (['yes' if any(a>65534 for a in i) else 'no' for i in df_ovl['weightsPPFX']])
    # df_ovl['inf_weight_reint'] = (['yes' if any(a>65534 for a in i) else 'no' for i in df_ovl['weightsReint']])

    # df_ovl['ones_weight_genie'] = (['yes' if all(a == 1 for a in i) else 'no' for i in df_ovl['weightsGenie']])
    # df_ovl['ones_weight_ppfx'] = (['yes' if all(a == 1 for a in i) else 'no' for i in df_ovl['weightsPPFX']])
    # df_ovl['ones_weight_reint'] = (['yes' if all(a == 1 for a in i) else 'no' for i in df_ovl['weightsReint']])
    
    # df_ovl['zero_weight_genie'] = (['yes' if all(a == 0 for a in i) else 'no' for i in df_ovl['weightsGenie']])
    # df_ovl['zero_weight_ppfx'] = (['yes' if all(a == 0 for a in i) else 'no' for i in df_ovl['weightsPPFX']])
    # df_ovl['zero_weight_reint'] = (['yes' if all(a == 0 for a in i) else 'no' for i in df_ovl['weightsReint']])

    df_ovl.loc[df_ovl['inf_weight_genie'] == "yes", 'is_good_genie'] = 0
    df_ovl.loc[df_ovl['zero_weight_genie'] == "yes", 'is_good_genie'] = 0
    df_ovl.loc[df_ovl['ones_weight_genie'] == "yes", 'is_good_genie'] = 0
    df_ovl.loc[df_ovl['thsnd_weight_genie'] == "yes", 'is_good_genie'] = 0
    
    df_ovl.loc[df_ovl['inf_weight_ppfx'] == "yes", 'is_good_ppfx'] = 0
    df_ovl.loc[df_ovl['zero_weight_ppfx'] == "yes", 'is_good_ppfx'] = 0
    df_ovl.loc[df_ovl['ones_weight_ppfx'] == "yes", 'is_good_ppfx'] = 0
    df_ovl.loc[df_ovl['thsnd_weight_ppfx'] == "yes", 'is_good_ppfx'] = 0
    
    df_ovl.loc[df_ovl['inf_weight_reint'] == "yes", 'is_good_reint'] = 0
    df_ovl.loc[df_ovl['zero_weight_reint'] == "yes", 'is_good_reint'] = 0
    df_ovl.loc[df_ovl['ones_weight_reint'] == "yes", 'is_good_reint'] = 0
    df_ovl.loc[df_ovl['thsnd_weight_reint'] == "yes", 'is_good_reint'] = 0
   
    # #### Detsys ####
    # if mass == '125' or mass == '130' or mass == '135' or mass == '140' or mass == '145':
    #     detsys_mass = '150'
    #     print("Loading 150 MeV detsys samples")
    # else: detsys_mass = mass
    
    # # detsys_sig_input_dir = './Final_detector_systematics_SIGNAL/'+detsys_mass+'_MeV/'
    # # detsys_bkg_input_dir = './Final_detector_systematics_BACKGROUND/'+detsys_mass+'_MeV/'
    # detsys_sig_input_dir = root_dir+'/Final_detector_systematics_SIGNAL/'+detsys_mass+'_MeV/'
    # detsys_bkg_input_dir = root_dir+'/Final_detector_systematics_BACKGROUND/'+detsys_mass+'_MeV/'

    # KDIF_KDAR_str_lower = KDIF_KDAR_str.lower()
    # df_sig_detsys = pd.read_pickle(detsys_sig_input_dir + 'Signal_' + fhc_rhc_str + '_' + detsys_mass + '_MeV_' + nshr + '_' + KDIF_KDAR_str_lower +'.pkl')
    # df_ovl_detsys = pd.read_pickle(detsys_bkg_input_dir + "bkg_" + fhc_rhc_str + '_' + detsys_mass + '_MeV_' + nshr + '_' + KDIF_KDAR_str_lower +'.pkl')

    df_ovl['logit_pred'] = logit(df_ovl['pred'])
    
    # pred_column_sig = [col for col in df_sig_detsys if col.startswith('pred')]
    # pred_column_ovl = [col for col in df_ovl_detsys if col.startswith('pred')]
    
    # df_sig_detsys_copy = df_sig_detsys.copy()
    # df_ovl_detsys_copy = df_ovl_detsys.copy()
    
    # for pred in pred_column_sig:
    #     df_sig_detsys_copy['logit_'+pred] = logit(df_sig_detsys[pred])
    
    # for pred in pred_column_ovl:
    #     df_ovl_detsys_copy['logit_'+pred] = logit(df_ovl_detsys[pred])

    print (f'SUCCESSFULLY LOADED THE PICKLE FILES {Run, mass, nshr, KDIF_KDAR_str}')
    return df_ovl



def New_loader_OLD(Run, mass, nshr, KDIF_KDAR_str):
    """
    New function to load the .pkl files as dataframes. Mostly taken directly from Aditya's code.
    get_tof is a boolean that determines whether to load the time of flight values from the root file.
    """
    #Constants for normalisation factors
    
    mc_pot_fhc = 2.33652e+21 # Need to update this when using new flux
    mc_pot_rhc = 1.98937e+21 # Need to update this when using new flux
    dirt_pot_fhc = 1.42143e+21
    dirt_pot_rhc = 1.03226e+21
    OnBeam_EA9CNT_wcut_fhc = 5268051.0
    OnBeam_EA9CNT_wcut_rhc = 10363728.0
    OffBeam_EXT_NUMIwin_FEMBeamTriggerAlgo_fhc = 9199232.74
    OffBeam_EXT_NUMIwin_FEMBeamTriggerAlgo_rhc = 32878305.25
    scale_ext_fhc = OnBeam_EA9CNT_wcut_fhc/OffBeam_EXT_NUMIwin_FEMBeamTriggerAlgo_fhc
    scale_ext_rhc = OnBeam_EA9CNT_wcut_rhc/OffBeam_EXT_NUMIwin_FEMBeamTriggerAlgo_rhc
    scale_ext_fhc = scale_ext_fhc*.98
    scale_ext_rhc = scale_ext_rhc*.98
    OnBeam_tortgt_wcut_fhc = 2.002e20
    OnBeam_tortgt_wcut_rhc = 5.009e20
    scale_nu_fhc = OnBeam_tortgt_wcut_fhc/mc_pot_fhc
    scale_nu_rhc = OnBeam_tortgt_wcut_rhc/mc_pot_rhc
    scale_dirt_fhc = OnBeam_tortgt_wcut_fhc/dirt_pot_fhc
    scale_dirt_rhc = OnBeam_tortgt_wcut_rhc/dirt_pot_rhc
    scale_dirt_fhc = scale_dirt_fhc*0.75
    scale_dirt_rhc = scale_dirt_rhc*0.35
    
    if Run == 'Run1':
        fhc_rhc_str = 'fhc'
        OnBeam_tortgt_wcut = OnBeam_tortgt_wcut_fhc
        scale_nu = scale_nu_fhc
        scale_dirt = scale_dirt_fhc
        scale_ext = scale_ext_fhc
    elif Run == 'Run3':
        fhc_rhc_str = 'rhc'
        OnBeam_tortgt_wcut = OnBeam_tortgt_wcut_rhc
        scale_nu = scale_nu_rhc
        scale_dirt = scale_dirt_rhc
        scale_ext = scale_ext_rhc
    
    root_dir = "/exp/uboone/data/users/jbateman/workdir/HPS_uboone_analysis/"
    main_input_dir = root_dir+"/BDT_inputs_pkl/"

    #### OVL ####

    df_ovl = pd.read_pickle(main_input_dir+'AllVar_Selected_'+Run+'_NuMI_Overlay_'+KDIF_KDAR_str+'_'+mass+'_'+nshr+'_PPFX_pred_NEW.pkl')
    frac_ovl_test_sample = len(df_ovl.query('is_trained == 0'))/len(df_ovl) #Should be 0.4
    df_ovl['frac_test_sample'] = frac_ovl_test_sample
    df_ovl = df_ovl.query('is_trained==0')
    df_ovl.eval('scale_factor_ls = (1/frac_test_sample)*KDAR_scale*'+str(scale_nu),inplace=True)

    df_ovl['is_good_genie'] = 1
    df_ovl['is_good_ppfx'] = 1
    df_ovl['is_good_reint'] = 1
    
    df_ovl['thsnd_weight_genie'] = (['yes' if all(a == 1000 for a in i) else 'no' for i in df_ovl['weightsGenie']])
    df_ovl['thsnd_weight_ppfx'] = (['yes' if all(a == 1000 for a in i) else 'no' for i in df_ovl['weightsPPFX']])
    df_ovl['thsnd_weight_reint'] = (['yes' if all(a == 1000 for a in i) else 'no' for i in df_ovl['weightsReint']])
    
    df_ovl.loc[df_ovl['inf_weight_genie'] == "yes", 'is_good_genie'] = 0
    df_ovl.loc[df_ovl['zero_weight_genie'] == "yes", 'is_good_genie'] = 0
    df_ovl.loc[df_ovl['ones_weight_genie'] == "yes", 'is_good_genie'] = 0
    df_ovl.loc[df_ovl['thsnd_weight_genie'] == "yes", 'is_good_genie'] = 0
    
    df_ovl.loc[df_ovl['inf_weight_ppfx'] == "yes", 'is_good_ppfx'] = 0
    df_ovl.loc[df_ovl['zero_weight_ppfx'] == "yes", 'is_good_ppfx'] = 0
    df_ovl.loc[df_ovl['ones_weight_ppfx'] == "yes", 'is_good_ppfx'] = 0
    df_ovl.loc[df_ovl['thsnd_weight_ppfx'] == "yes", 'is_good_ppfx'] = 0
    
    df_ovl.loc[df_ovl['inf_weight_reint'] == "yes", 'is_good_reint'] = 0
    df_ovl.loc[df_ovl['zero_weight_reint'] == "yes", 'is_good_reint'] = 0
    df_ovl.loc[df_ovl['ones_weight_reint'] == "yes", 'is_good_reint'] = 0
    df_ovl.loc[df_ovl['thsnd_weight_reint'] == "yes", 'is_good_reint'] = 0
    
    df_ovl['logit_pred'] = logit(df_ovl['pred'])

    print (f'SUCCESSFULLY LOADED THE PICKLE FILES {Run, mass, nshr, KDIF_KDAR_str}')
    
    return df_ovl

def format_for_step(hist, bins):
    hist_step = np.insert(hist,0,hist[0])
    return hist_step, bins
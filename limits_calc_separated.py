#Loading libraries
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pyhf
import scipy
from scipy import stats
import uproot3
import uproot
import math
import awkward as ak
import pickle
import csv
import copy

import argparse

from importlib import reload

def New_Load_pyhf_files(filenames, Params_pyhf, location='BDT_output/', HNL_masses=[100, 150, 200], use_test = False):
    loc_hists = location

    hist_dict_run1, hist_dict_run3, theta_dict = {}, {}, {}

    #Loading in the .root files
    for HNL_mass in HNL_masses:
        for shr in ["1shr", "2shr"]:
            for K in ["KDIF", "KDAR"]:
                name_ends = f"{HNL_mass}_{shr}_{K}"
                if use_test == True:
                    print('Using test files')
                    hist_dict_run1[name_ends] = uproot.open(loc_hists+f'run1_TEST_ALL_Capped_{name_ends}' + filenames)
                    hist_dict_run3[name_ends] = uproot.open(loc_hists+f'run3_TEST_ALL_Capped_{name_ends}' + filenames)
                else:
                    hist_dict_run1[name_ends] = uproot.open(loc_hists+f'run1_ALL_Capped_{name_ends}' + filenames)
                    hist_dict_run3[name_ends] = uproot.open(loc_hists+f'run3_ALL_Capped_{name_ends}' + filenames)
                theta_dict[HNL_mass] = hist_dict_run1[name_ends]["theta"].values()[0] #assuming scaled theta is the same for all runs, only 1 value saved

    all_hists_list = ['bkg_overlay;1', 'bkg_dirt;1', 'bkg_EXT;1', 'signal;1', 'data;1', 'theta;1',  
                      'ppfx_uncertainty_frac;1', 'Genie_uncertainty_frac;1', 'Reinteraction_uncertainty_frac;1', 
                      'overlay_DetVar_uncertainty_frac;1', 'signal_DetVar_uncertainty_frac;1', "signal_KDAR_frac;1"]
    missing_hists = []
    for hist_name in all_hists_list:
        if hist_name not in hist_dict_run1[name_ends].keys(): missing_hists.append(hist_name)
    if len(missing_hists) == 0: print("No missing histograms in Run1")
    else:
        print("Missing hists for Run1 are: ")
        print(missing_hists)

    for hist_name in all_hists_list:
        if hist_name not in hist_dict_run3[name_ends].keys(): missing_hists.append(hist_name)
    if len(missing_hists) == 0: print("No missing histograms in Run3")
    else:
        print("Missing hists for Run3 are: ")
        print(missing_hists)


    print("thetas are:")
    print(theta_dict)
    print("Done")
    
    return hist_dict_run1, hist_dict_run3, theta_dict

def add_hists_vals(hist_list):
    Total_hist = np.zeros_like(hist_list[0].values())
    for hist in hist_list:
        Total_hist += hist.values()
    return Total_hist

def add_all_errors_dict(err_dict): #adds in quadrature, assuming all hists are same shape
    list_keys = list(err_dict.keys())
    Total_hist = np.zeros_like(err_dict[list_keys[0]])
    for i in range(len(err_dict[list_keys[0]])): #Looping over the bins
        for errs in err_dict.keys(): #Looping over the histograms
            Total_hist[i] += err_dict[errs][i]**2 #Adding error from each hist in quadrature
        Total_hist[i] = np.sqrt(Total_hist[i])
    return Total_hist

def add_all_errors(err_list): #adds in quadrature, assuming all hists are same shape
    Total_hist = np.zeros_like(err_list[0])
    for i in range(len(err_list[0])): #Looping over the bins
        for errs in err_list: #Looping over the histograms
            Total_hist[i] += errs[i]**2 #Adding error from each hist in quadrature
        Total_hist[i] = np.sqrt(Total_hist[i])
    return Total_hist

def Full_calculate_total_uncertainty(Params, hist_dict, zero_bins_errs): #Takes the dictionary of all root files
    """
    Given parameters, hist dict and zero bins error.
    Returns a dict of all types of error and individual sample values for use in models.
    """
    OVERLAY_VALS, DIRT_VALS, BEAMOFF_VALS = {}, {}, {}
    OVERLAY_STAT, DIRT_STAT, BEAMOFF_STAT = {}, {}, {}
    TOT_BKG_ERR_dict, TOT_SIGNAL_ERR_dict = {}, {}
    BKG_STAT_ERR_dict, SIGNAL_STAT_ERR_dict = {}, {}
    BKG_DETVAR_ERR_dict, BKG_DIRT_ERR_dict, BKG_MULTISIM_ERR_dict = {}, {}, {}
    BKG_SHAPESYS_ERR_dict, SIGNAL_SHAPESYS_ERR_dict = {}, {}
    BKG_DETVAR_MULTISIM_dict = {}
    SIGNAL_DETVAR_ERR_dict = {}
    SIGNAL_NORMSYS_ERR_dict = {} #No normsys for this because currently background contributions are added together
    bkg_sample_names = ['bkg_overlay','bkg_EXT','bkg_dirt']
    overlay_sys_frac_names = ["ppfx_uncertainty_frac","Genie_uncertainty_frac","Reinteraction_uncertainty_frac","overlay_DetVar_uncertainty_frac"]
    for HNL_mass in hist_dict:
        bkg_stat_err_dict, bkg_sys_err_dict = {}, {} #Clean for each mass point
        
        OVERLAY_VALS[HNL_mass] = hist_dict[HNL_mass]['bkg_overlay'].values()
        DIRT_VALS[HNL_mass] = hist_dict[HNL_mass]['bkg_dirt'].values()
        BEAMOFF_VALS[HNL_mass] = hist_dict[HNL_mass]['bkg_EXT'].values()
        OVERLAY_STAT[HNL_mass] = np.add(hist_dict[HNL_mass]['bkg_overlay'].errors(), zero_bins_errs[HNL_mass]['bkg_overlay'])
        DIRT_STAT[HNL_mass] = np.add(hist_dict[HNL_mass]['bkg_dirt'].errors(), zero_bins_errs[HNL_mass]['bkg_dirt'])
        BEAMOFF_STAT[HNL_mass] = np.add(hist_dict[HNL_mass]['bkg_EXT'].errors(), zero_bins_errs[HNL_mass]['bkg_EXT'])
        
        for name in bkg_sample_names:
            bkg_stat_err_dict[name]=np.add(hist_dict[HNL_mass][name].errors(), zero_bins_errs[HNL_mass][name])
        sig_stat_err = np.add(hist_dict[HNL_mass]['signal'].errors(), zero_bins_errs[HNL_mass]['signal'])
        if Params["Stats_only"] == True: #Set all systematic errors to zero
            for name in bkg_sample_names:
                bkg_sys_err_dict[name] = np.zeros_like(hist_dict[HNL_mass][name].errors())
            sig_sys_err =  np.zeros_like(hist_dict[HNL_mass]['signal'].errors())
        elif Params["Use_flat_sys"] == True:
            for name in bkg_sample_names:
                bkg_sys_err_dict[name] = hist_dict[HNL_mass][name].values()*Params["Flat_"+name+"_frac"]
            sig_flux_err = hist_dict[HNL_mass]['signal'].values()*hist_dict[HNL_mass]["signal_KDAR_frac"].to_numpy()[0]
            sig_detvar_err = hist_dict[HNL_mass]['signal'].values()*Params["Flat_sig_detvar"]
            sig_sys_err = np.sqrt(sig_flux_err**2 + sig_detvar_err**2)
        #This is using the fully evaluated uncertainties
        elif Params["Use_flat_sys"] == False: 
            overlay_sys_dict = {}
            for sys in overlay_sys_frac_names:
                overlay_sys_dict[sys] = hist_dict[HNL_mass][sys].values()*hist_dict[HNL_mass]['bkg_overlay'].values()
            bkg_sys_err_dict['bkg_overlay'] = add_all_errors_dict(overlay_sys_dict)
            bkg_sys_err_dict['bkg_EXT'] = np.zeros_like(hist_dict[HNL_mass]['bkg_EXT'].errors())
            bkg_sys_err_dict['bkg_dirt'] = hist_dict[HNL_mass]['bkg_dirt'].values()*Params["Flat_bkg_dirt_frac"]
            
            # sig_detvar_err = hist_dict[HNL_mass]["signal_DetVar_uncertainty"].values()
            sig_detvar_err = hist_dict[HNL_mass]["signal_DetVar_uncertainty_frac"].values()*hist_dict[HNL_mass]['signal'].values()
            sig_flux_err = hist_dict[HNL_mass]['signal'].values()*hist_dict[HNL_mass]["signal_KDAR_frac"].to_numpy()[0]
            sig_sys_err = add_all_errors([sig_detvar_err,sig_flux_err])
            
        #Evaluating final stat+sys errors    
        bkg_stat_plus_sys_dict={}
        for name in bkg_sample_names:
            bkg_stat_plus_sys_dict[name]=add_all_errors([bkg_stat_err_dict[name],bkg_sys_err_dict[name]]) 
        
        total_bkg_err = add_all_errors_dict(bkg_stat_plus_sys_dict) #Now adding the errors of overlay, EXT and dirt in quadrature
        total_sig_err = add_all_errors([sig_stat_err,sig_sys_err])
        
        TOT_BKG_ERR_dict[HNL_mass] = total_bkg_err
        TOT_SIGNAL_ERR_dict[HNL_mass] = total_sig_err
        
        BKG_STAT_ERR_dict[HNL_mass] = add_all_errors_dict(bkg_stat_err_dict)
        BKG_SHAPESYS_ERR_dict[HNL_mass] = add_all_errors_dict(bkg_sys_err_dict)
        BKG_DETVAR_ERR_dict[HNL_mass] = overlay_sys_dict["overlay_DetVar_uncertainty_frac"]
        BKG_DIRT_ERR_dict[HNL_mass] = bkg_sys_err_dict['bkg_dirt']
        BKG_MULTISIM_ERR_dict[HNL_mass] = add_all_errors([overlay_sys_dict["ppfx_uncertainty_frac"],
                                                          overlay_sys_dict["Genie_uncertainty_frac"],
                                                          overlay_sys_dict["Reinteraction_uncertainty_frac"]])
        BKG_DETVAR_MULTISIM_dict[HNL_mass] = add_all_errors([BKG_DETVAR_ERR_dict[HNL_mass], BKG_MULTISIM_ERR_dict[HNL_mass]])
        
        SIGNAL_STAT_ERR_dict[HNL_mass] = sig_stat_err
        SIGNAL_SHAPESYS_ERR_dict[HNL_mass] = sig_detvar_err
        SIGNAL_DETVAR_ERR_dict[HNL_mass] = sig_detvar_err
    TOT_ERR_DICT = {}
    TOT_ERR_DICT["OVERLAY_VALS"], TOT_ERR_DICT["DIRT_VALS"], TOT_ERR_DICT["BEAMOFF_VALS"] = OVERLAY_VALS, DIRT_VALS, BEAMOFF_VALS
    TOT_ERR_DICT["OVERLAY_STAT"], TOT_ERR_DICT["DIRT_STAT"], TOT_ERR_DICT["BEAMOFF_STAT"] = OVERLAY_STAT, DIRT_STAT, BEAMOFF_STAT
    TOT_ERR_DICT["TOT_BKG_ERR"], TOT_ERR_DICT["TOT_SIGNAL_ERR"] = TOT_BKG_ERR_dict, TOT_SIGNAL_ERR_dict
    TOT_ERR_DICT["BKG_STAT"], TOT_ERR_DICT["BKG_SHAPESYS"] = BKG_STAT_ERR_dict, BKG_SHAPESYS_ERR_dict
    TOT_ERR_DICT["BKG_DETVAR"], TOT_ERR_DICT["BKG_DIRT"], TOT_ERR_DICT["BKG_MULTISIM"] = BKG_DETVAR_ERR_dict, BKG_DIRT_ERR_dict, BKG_MULTISIM_ERR_dict
    TOT_ERR_DICT["BKG_DETVAR_MULTISIM"] = BKG_DETVAR_MULTISIM_dict
    TOT_ERR_DICT["SIGNAL_STAT"], TOT_ERR_DICT["SIGNAL_SHAPESYS"] = SIGNAL_STAT_ERR_dict, SIGNAL_SHAPESYS_ERR_dict
    TOT_ERR_DICT["SIGNAL_DETVAR"] = SIGNAL_DETVAR_ERR_dict
    
    return TOT_ERR_DICT

def False_zero_bins(hist_dict):
    """
    This will set all "zero bin" errors as zero, i.e not accounting for them. 
    """
    print("Not accounting for zero bin count errors.")
    zero_bins_errors = {}
    bkg_sample_names = ['bkg_overlay','bkg_EXT','bkg_dirt']
    
    for HNL_mass in hist_dict:
        zero_bins_per_mass = {}
        for bkg in bkg_sample_names:
            zero_bins_per_mass[bkg] = np.zeros_like(hist_dict[HNL_mass][bkg].values())
                    
        zero_bins_per_mass["signal"] = np.zeros_like(hist_dict[HNL_mass]["signal"].values())
        zero_bins_errors[HNL_mass] = zero_bins_per_mass
        
    return zero_bins_errors

def Add_bkg_hists_make_signal(hist_dict):
    """
    Input dict of histgrams.
    Returns dicts of total BKG and total signal.
    """
    BKG_dict, SIGNAL_dict = {}, {}
    for HNL_mass in hist_dict:
        bkg_hists = [hist_dict[HNL_mass]['bkg_EXT'], hist_dict[HNL_mass]['bkg_overlay'], hist_dict[HNL_mass]['bkg_dirt']]
        
        total_bkg = add_hists_vals(bkg_hists)
        BKG_dict[HNL_mass] = total_bkg
        SIGNAL_dict[HNL_mass] = hist_dict[HNL_mass]['signal'].values()
 
    return BKG_dict, SIGNAL_dict

def remove_part_hist(hist_list, numbins):
        length = len(hist_list)
        slice_at = length - int(numbins)
        if slice_at < 0:
            # print("Trying to use greater number of bins than available, using full dist.")
            return hist_list
        else:
            sliced_hist = hist_list[slice_at:]
            return sliced_hist

def Make_into_lists(Params, BKG_dict, SIGNAL_dict, TOT_ERR_dict):
    """
    Takes parameters, the dicts of bkg and signal values and the error dict.
    Returns an output dict with bkg vals, signal vals and error vals all as lists with the correct number of bins.
    """
    BKG_dict_FINAL, SIGNAL_dict_FINAL= {}, {}
    ERR_dict_FINAL = {}
    for HNL_mass in BKG_dict:
        ERR_list_dict = {}
        BKG = np.ndarray.tolist(BKG_dict[HNL_mass])
        SIGNAL = np.ndarray.tolist(SIGNAL_dict[HNL_mass])
        for err_dict in TOT_ERR_dict:
            ERR_list_dict[err_dict]=np.ndarray.tolist(TOT_ERR_dict[err_dict][HNL_mass])
        if Params["Use_part_only"] == True:
            numbins = Params["Num_bins_for_calc"] #Number of bins in signal region to use for CLs calc
            BKG=remove_part_hist(BKG, numbins)
            SIGNAL=remove_part_hist(SIGNAL, numbins)
            for err_dict in ERR_list_dict:
                ERR_list_dict[err_dict]=remove_part_hist(ERR_list_dict[err_dict], numbins)
            
        BKG_dict_FINAL[HNL_mass] = BKG
        SIGNAL_dict_FINAL[HNL_mass] = SIGNAL
        ERR_dict_FINAL[HNL_mass] = ERR_list_dict

    # output_dict = {"BKG_dict":BKG_dict_FINAL, "SIGNAL_dict":SIGNAL_dict_FINAL}
    output_dict = {"TOT_BKG_VALS":BKG_dict_FINAL, "TOT_SIGNAL_VALS":SIGNAL_dict_FINAL}

    for err_dict in TOT_ERR_dict:
        new_err_dict_placeholder = {}
        for HNL_mass in BKG_dict:
            new_err_dict_placeholder[HNL_mass] = ERR_dict_FINAL[HNL_mass][err_dict]
        
        output_dict.update({err_dict:new_err_dict_placeholder})
        
    return output_dict

def append_list_of_lists(input_list):
        output_list = []
        for i in range(len(input_list)):
            output_list = output_list + input_list[i]
        return output_list

def Append_four_channels(Run_output):
    
    Merged_dict = {}
    Total_merged = {}
    
    all_keys = list(Run_output.keys())
    first_key = all_keys[0]
    
    all_hists = list(Run_output[first_key].keys())
    all_masses=[]
    for key in all_hists:
        # all_masses.append(key.split("_")[0])
        all_masses.append(int(key.split("_")[0]))
        
    unique_masses = set(all_masses)
        
    merge_list = ["1shr_KDIF", "1shr_KDAR", "2shr_KDIF", "1shr_KDAR"]
    
    for key in all_keys:
        Merged_dict={}
        for mass_val in unique_masses:
            list_placeholder = []
            for name in merge_list:
                list_placeholder.append(Run_output[key][f"{mass_val}_{name}"])
            Merged = append_list_of_lists(list_placeholder)
            Merged_dict[mass_val]=Merged
        Total_merged[key] = Merged_dict
        
    return Total_merged

def Append_kdif_channels(Run_output):
    
    Merged_dict = {}
    Total_merged = {}
    
    all_keys = list(Run_output.keys())
    first_key = all_keys[0]
    
    all_hists = list(Run_output[first_key].keys())
    all_masses=[]
    for key in all_hists:
        # all_masses.append(key.split("_")[0])
        all_masses.append(int(key.split("_")[0]))
        
    unique_masses = set(all_masses)
        
    merge_list = ["1shr_KDIF", "2shr_KDIF"]
    
    for key in all_keys:
        Merged_dict={}
        for mass_val in unique_masses:
            list_placeholder = []
            for name in merge_list:
                list_placeholder.append(Run_output[key][f"{mass_val}_{name}"])
            Merged = append_list_of_lists(list_placeholder)
            Merged_dict[mass_val]=Merged
        Total_merged[key] = Merged_dict
        
    return Total_merged

def Append_kdar_channels(Run_output):

    Merged_dict = {}
    Total_merged = {}

    all_keys = list(Run_output.keys())
    first_key = all_keys[0]

    all_hists = list(Run_output[first_key].keys())
    all_masses=[]
    for key in all_hists:
        # all_masses.append(key.split("_")[0])
        all_masses.append(int(key.split("_")[0]))

    unique_masses = set(all_masses)

    merge_list = ["1shr_KDAR", "2shr_KDAR"]

    for key in all_keys:
        Merged_dict = {}
        for mass_val in unique_masses:
            list_placeholder = []
            for name in merge_list:
                list_placeholder.append(Run_output[key][f"{mass_val}_{name}"])
            Merged = append_list_of_lists(list_placeholder)
            Merged_dict[mass_val] = Merged
        Total_merged[key] = Merged_dict
        
    return Total_merged


def Append_1shr_channels(Run_output):
    Merged_dict = {}
    Total_merged = {}

    all_keys = list(Run_output.keys())
    first_key = all_keys[0]

    all_hists = list(Run_output[first_key].keys())
    all_masses=[]
    for key in all_hists:
        # all_masses.append(key.split("_")[0])
        all_masses.append(int(key.split("_")[0]))

    unique_masses = set(all_masses)

    merge_list = ["1shr_KDAR", "1shr_KDIF"]

    for key in all_keys:
        Merged_dict = {}
        for mass_val in unique_masses:
            list_placeholder = []
            for name in merge_list:
                list_placeholder.append(Run_output[key][f"{mass_val}_{name}"])
            Merged = append_list_of_lists(list_placeholder)
            Merged_dict[mass_val] = Merged
        Total_merged[key] = Merged_dict
        
    return Total_merged

def Append_2shr_channels(Run_output):
    Merged_dict = {}
    Total_merged = {}

    all_keys = list(Run_output.keys())
    first_key = all_keys[0]

    all_hists = list(Run_output[first_key].keys())
    all_masses=[]
    for key in all_hists:
        # all_masses.append(key.split("_")[0])
        all_masses.append(int(key.split("_")[0]))

    unique_masses = set(all_masses)

    merge_list = ["2shr_KDAR", "2shr_KDIF"]

    for key in all_keys:
        Merged_dict = {}
        for mass_val in unique_masses:
            list_placeholder = []
            for name in merge_list:
                list_placeholder.append(Run_output[key][f"{mass_val}_{name}"])
            Merged = append_list_of_lists(list_placeholder)
            Merged_dict[mass_val] = Merged
        Total_merged[key] = Merged_dict
        
    return Total_merged

def Create_final_appended_runs_dict(list_input_dicts):

    Total_dict = {}
    all_keys = list(list_input_dicts[0].keys())
    first_key = all_keys[0]
    for HNL_mass in list_input_dicts[0][first_key]:
    # for HNL_mass in Constants.HNL_mass_samples:
        Appended_dict = {}
        # HNL_mass_val = HNL_mass.split("_")[0]
        HNL_mass_val = HNL_mass
        for dict_type in list_input_dicts[0].keys():
            list_placeholder = []
            for input_dict in list_input_dicts: #This loops over the dicts for different runs
                list_placeholder.append(input_dict[dict_type][HNL_mass]) 
            Appended = append_list_of_lists(list_placeholder)
            Appended_dict[dict_type] = Appended
        Total_dict[HNL_mass] = Appended_dict
        
    return Total_dict

def add_data(Total_dict, hists, numbins):
    """
    Given total dict, hist dict and number of bins.
    Returns total dict with data added for a single run.
    """
    for HNL_mass in Total_dict:
        hist_placeholder = list(hists[HNL_mass]["data"].values())
        hist_data = remove_part_hist(hist_placeholder, numbins)
        Total_dict[HNL_mass]["data"]=hist_data
    return Total_dict

def add_data_appended(Total_dict, hists_r1, hists_r3, numbins):
    """
    Given Total dict, r1 and r3 dicts and number of bins.
    Returns new Total dict with data added for both runs. 
    """
    for HNL_mass in Total_dict:
        r1_hist_placeholder = list(hists_r1[HNL_mass]["data"].values())
        r3_hist_placeholder = list(hists_r3[HNL_mass]["data"].values())
        r1_hist = remove_part_hist(r1_hist_placeholder, numbins)
        r3_hist = remove_part_hist(r3_hist_placeholder, numbins)
        appended = r1_hist+r3_hist
        Total_dict[HNL_mass]["data"]=appended
    return Total_dict

def add_data_merged_appended(Total_dict, hists_r1, hists_r3, numbins):
    """
    Given Total dict, r1 and r3 dicts and number of bins.
    Returns new Total dict with data added for both runs. 
    """
    merge_list = ["1shr_KDIF", "1shr_KDAR", "2shr_KDIF", "1shr_KDAR"]
    
    r1_merged, r3_merged = {}, {}
    Total_merged = {}
    
    all_keys = list(hists_r1.keys())
    first_key = all_keys[0]
    
    all_masses=[]
    #Looping over 100_KDIF_1shr
    for key in all_keys:
        all_masses.append(key.split("_")[0])
        
    unique_masses = set(all_masses)
        
    merge_list = ["1shr_KDIF", "1shr_KDAR", "2shr_KDIF", "1shr_KDAR"]
    #Looping over 100_KDIF_1shr
    for key in all_keys:
        r1_merged, r3_merged = {}, {}
        for mass_val in unique_masses:
            list_placeholder_r1, list_placeholder_r3 = [], []
            for name in merge_list:
                vals_r1 = hists_r1[f"{mass_val}_{name}"]["data"].values()
                vals_r3 = hists_r3[f"{mass_val}_{name}"]["data"].values()
                r1_hist_sliced = remove_part_hist(list(vals_r1), numbins)
                r3_hist_sliced = remove_part_hist(list(vals_r3), numbins)
                list_placeholder_r1.append(r1_hist_sliced)
                list_placeholder_r3.append(r3_hist_sliced)
            Merged_r1 = append_list_of_lists(list_placeholder_r1)
            Merged_r3 = append_list_of_lists(list_placeholder_r3)
            # Merged_dict[mass_val]=Merged
            r1_merged[mass_val]=Merged_r1
            r3_merged[mass_val]=Merged_r3
    
    for HNL_mass in Total_dict:
        r1_hist_placeholder = list(r1_merged[str(HNL_mass)])
        r3_hist_placeholder = list(r3_merged[str(HNL_mass)])
        # r1_hist = remove_part_hist(r1_hist_placeholder, numbins)
        # r3_hist = remove_part_hist(r3_hist_placeholder, numbins)
        appended = r1_hist_placeholder+r3_hist_placeholder
        Total_dict[HNL_mass]["data"]=appended
    return Total_dict

# merge_list = ["1shr_KDIF", "1shr_KDAR", "2shr_KDIF", "1shr_KDAR"]

def create_stat_unc_safe_hist(Total_dict):
    """
    Given total dict, which contains the stat uncertainties for TOTAL signal and TOTAL bkg.
    Returns dicts of the stat unc for signal and bkg. For use in models.
    Stops bins with zero stat error giving inf values. 
    """
    sig_stat, bkg_stat = {}, {}
    for HNL_mass in Total_dict['TOT_SIGNAL_VALS']:
        sig_stat[HNL_mass] = np.zeros_like(np.array(Total_dict['TOT_SIGNAL_VALS'][HNL_mass]))
        bkg_stat[HNL_mass] = np.zeros_like(np.array(Total_dict['TOT_BKG_VALS'][HNL_mass]))
        for i, val in enumerate(Total_dict['TOT_SIGNAL_VALS'][HNL_mass]):
            if val == 0: sig_stat[HNL_mass][i] = 0.0
            else: sig_stat[HNL_mass][i] = Total_dict['SIGNAL_STAT'][HNL_mass][i]/Total_dict['TOT_SIGNAL_VALS'][HNL_mass][i]
            
        for i, val in enumerate(Total_dict['TOT_BKG_VALS'][HNL_mass]):
            if val == 0: bkg_stat[HNL_mass][i] = 0.0
            else: bkg_stat[HNL_mass][i] = Total_dict['BKG_STAT'][HNL_mass][i]/Total_dict['TOT_BKG_VALS'][HNL_mass][i]
        sig_stat[HNL_mass], bkg_stat[HNL_mass] = list(sig_stat[HNL_mass]), list(bkg_stat[HNL_mass])
        # sig_stat[HNL_mass] = list(np.divide(np.array(Total_dict[HNL_mass]['SIGNAL_STAT']),np.array(Total_dict[HNL_mass]['SIGNAL_dict'])))
        # bkg_stat[HNL_mass] = list(np.divide(np.array(Total_dict[HNL_mass]['BKG_STAT']),np.array(Total_dict[HNL_mass]['BKG_dict'])))
    return sig_stat, bkg_stat

def create_stat_unc_safe(Total_dict):
    """
    Given total dict, which contains the stat uncertainties for TOTAL signal and TOTAL bkg.
    Returns dicts of the stat unc for signal and bkg. For use in models.
    Stops bins with zero stat error giving inf values. 
    """
    sig_stat, bkg_stat = {}, {}
    for HNL_mass in Total_dict:
        sig_stat[HNL_mass] = np.zeros_like(np.array(Total_dict[HNL_mass]['TOT_SIGNAL_VALS']))
        bkg_stat[HNL_mass] = np.zeros_like(np.array(Total_dict[HNL_mass]['TOT_BKG_VALS']))
        for i, val in enumerate(Total_dict[HNL_mass]['TOT_SIGNAL_VALS']):
            if val == 0: sig_stat[HNL_mass][i] = 0.0
            else: sig_stat[HNL_mass][i] = Total_dict[HNL_mass]['SIGNAL_STAT'][i]/Total_dict[HNL_mass]['TOT_SIGNAL_VALS'][i]
            
        for i, val in enumerate(Total_dict[HNL_mass]['TOT_BKG_VALS']):
            if val == 0: bkg_stat[HNL_mass][i] = 0.0
            else: bkg_stat[HNL_mass][i] = Total_dict[HNL_mass]['BKG_STAT'][i]/Total_dict[HNL_mass]['TOT_BKG_VALS'][i]
        sig_stat[HNL_mass], bkg_stat[HNL_mass] = list(sig_stat[HNL_mass]), list(bkg_stat[HNL_mass])
        # sig_stat[HNL_mass] = list(np.divide(np.array(Total_dict[HNL_mass]['SIGNAL_STAT']),np.array(Total_dict[HNL_mass]['SIGNAL_dict'])))
        # bkg_stat[HNL_mass] = list(np.divide(np.array(Total_dict[HNL_mass]['BKG_STAT']),np.array(Total_dict[HNL_mass]['BKG_dict'])))
    return sig_stat, bkg_stat

def create_individual_stat_unc_safe(Total_dict):
    """
    Given total dict, which contains the stat uncertainties for TOTAL signal, overlay, dirt and beamoff.
    Returns dicts of the stat unc for signal and bkgs. For use in models.
    Stops bins with zero stat error giving inf values. 
    """
    sig_stat, overlay_stat, dirt_stat, beamoff_stat = {}, {}, {}, {}
    for HNL_mass in Total_dict:
        sig_stat[HNL_mass] = np.zeros_like(np.array(Total_dict[HNL_mass]['TOT_SIGNAL_VALS']))
        overlay_stat[HNL_mass] = np.zeros_like(np.array(Total_dict[HNL_mass]['OVERLAY_VALS']))
        dirt_stat[HNL_mass] = np.zeros_like(np.array(Total_dict[HNL_mass]['DIRT_VALS']))
        beamoff_stat[HNL_mass] = np.zeros_like(np.array(Total_dict[HNL_mass]['BEAMOFF_VALS']))
        for i, val in enumerate(Total_dict[HNL_mass]['TOT_SIGNAL_VALS']):
            if val == 0: sig_stat[HNL_mass][i] = 0.0
            else: sig_stat[HNL_mass][i] = Total_dict[HNL_mass]['SIGNAL_STAT'][i]/Total_dict[HNL_mass]['TOT_SIGNAL_VALS'][i]
            
        for i, val in enumerate(Total_dict[HNL_mass]['OVERLAY_VALS']):
            if val == 0: overlay_stat[HNL_mass][i] = 0.0
            else: overlay_stat[HNL_mass][i] = Total_dict[HNL_mass]['OVERLAY_STAT'][i]/Total_dict[HNL_mass]['OVERLAY_VALS'][i]
        for i, val in enumerate(Total_dict[HNL_mass]['DIRT_VALS']):
            if val == 0: dirt_stat[HNL_mass][i] = 0.0
            else: dirt_stat[HNL_mass][i] = Total_dict[HNL_mass]['DIRT_STAT'][i]/Total_dict[HNL_mass]['DIRT_VALS'][i]
        for i, val in enumerate(Total_dict[HNL_mass]['BEAMOFF_VALS']):
            if val == 0: beamoff_stat[HNL_mass][i] = 0.0
            else: beamoff_stat[HNL_mass][i] = Total_dict[HNL_mass]['BEAMOFF_STAT'][i]/Total_dict[HNL_mass]['BEAMOFF_VALS'][i]
        
        sig_stat[HNL_mass], overlay_stat[HNL_mass] = list(sig_stat[HNL_mass]), list(overlay_stat[HNL_mass])
        dirt_stat[HNL_mass], beamoff_stat[HNL_mass] = list(dirt_stat[HNL_mass]), list(beamoff_stat[HNL_mass])
        
    return sig_stat, overlay_stat, dirt_stat, beamoff_stat

def Add_bkg_hists_make_signal(hist_dict):
    """
    Input dict of histgrams.
    Returns dicts of total BKG and total signal.
    """
    BKG_dict, SIGNAL_dict = {}, {}
    for HNL_mass in hist_dict:
        bkg_hists = [hist_dict[HNL_mass]['bkg_EXT'], hist_dict[HNL_mass]['bkg_overlay'], hist_dict[HNL_mass]['bkg_dirt']]
        
        total_bkg = add_hists_vals(bkg_hists)
        BKG_dict[HNL_mass] = total_bkg
        SIGNAL_dict[HNL_mass] = hist_dict[HNL_mass]['signal'].values()
 
    return BKG_dict, SIGNAL_dict

def remove_part_hist(hist_list, numbins):
        length = len(hist_list)
        slice_at = length - int(numbins)
        if slice_at < 0:
            # print("Trying to use greater number of bins than available, using full dist.")
            return hist_list
        else:
            sliced_hist = hist_list[slice_at:]
            return sliced_hist

def Make_into_lists(Params, BKG_dict, SIGNAL_dict, TOT_ERR_dict):
    """
    Takes parameters, the dicts of bkg and signal values and the error dict.
    Returns an output dict with bkg vals, signal vals and error vals all as lists with the correct number of bins.
    """
    BKG_dict_FINAL, SIGNAL_dict_FINAL= {}, {}
    ERR_dict_FINAL = {}
    for HNL_mass in BKG_dict:
        ERR_list_dict = {}
        BKG = np.ndarray.tolist(BKG_dict[HNL_mass])
        SIGNAL = np.ndarray.tolist(SIGNAL_dict[HNL_mass])
        for err_dict in TOT_ERR_dict:
            ERR_list_dict[err_dict]=np.ndarray.tolist(TOT_ERR_dict[err_dict][HNL_mass])
        if Params["Use_part_only"] == True:
            numbins = Params["Num_bins_for_calc"] #Number of bins in signal region to use for CLs calc
            BKG=remove_part_hist(BKG, numbins)
            SIGNAL=remove_part_hist(SIGNAL, numbins)
            for err_dict in ERR_list_dict:
                ERR_list_dict[err_dict]=remove_part_hist(ERR_list_dict[err_dict], numbins)
            
        BKG_dict_FINAL[HNL_mass] = BKG
        SIGNAL_dict_FINAL[HNL_mass] = SIGNAL
        ERR_dict_FINAL[HNL_mass] = ERR_list_dict

    # output_dict = {"BKG_dict":BKG_dict_FINAL, "SIGNAL_dict":SIGNAL_dict_FINAL}
    output_dict = {"TOT_BKG_VALS":BKG_dict_FINAL, "TOT_SIGNAL_VALS":SIGNAL_dict_FINAL}

    for err_dict in TOT_ERR_dict:
        new_err_dict_placeholder = {}
        for HNL_mass in BKG_dict:
            new_err_dict_placeholder[HNL_mass] = ERR_dict_FINAL[HNL_mass][err_dict]
        
        output_dict.update({err_dict:new_err_dict_placeholder})
        
    return output_dict

def append_list_of_lists(input_list):
        output_list = []
        for i in range(len(input_list)):
            output_list = output_list + input_list[i]
        return output_list

def Append_four_channels(Run_output):
    
    Merged_dict = {}
    Total_merged = {}
    
    all_keys = list(Run_output.keys())
    first_key = all_keys[0]
    
    all_hists = list(Run_output[first_key].keys())
    all_masses=[]
    for key in all_hists:
        # all_masses.append(key.split("_")[0])
        all_masses.append(int(key.split("_")[0]))
        
    unique_masses = set(all_masses)
        
    merge_list = ["1shr_KDIF", "1shr_KDAR", "2shr_KDIF", "1shr_KDAR"]
    
    for key in all_keys:
        Merged_dict={}
        for mass_val in unique_masses:
            list_placeholder = []
            for name in merge_list:
                list_placeholder.append(Run_output[key][f"{mass_val}_{name}"])
            Merged = append_list_of_lists(list_placeholder)
            Merged_dict[mass_val]=Merged
        Total_merged[key] = Merged_dict
        
    return Total_merged

def Create_final_appended_runs_dict(list_input_dicts):

    Total_dict = {}
    all_keys = list(list_input_dicts[0].keys())
    first_key = all_keys[0]
    for HNL_mass in list_input_dicts[0][first_key]:
    # for HNL_mass in Constants.HNL_mass_samples:
        Appended_dict = {}
        # HNL_mass_val = HNL_mass.split("_")[0]
        HNL_mass_val = HNL_mass
        for dict_type in list_input_dicts[0].keys():
            list_placeholder = []
            for input_dict in list_input_dicts: #This loops over the dicts for different runs
                list_placeholder.append(input_dict[dict_type][HNL_mass]) 
            Appended = append_list_of_lists(list_placeholder)
            Appended_dict[dict_type] = Appended
        Total_dict[HNL_mass] = Appended_dict
        
    return Total_dict

def add_data(Total_dict, hists, numbins):
    """
    Given total dict, hist dict and number of bins.
    Returns total dict with data added for a single run.
    """
    for HNL_mass in Total_dict:
        hist_placeholder = list(hists[HNL_mass]["data"].values())
        hist_data = remove_part_hist(hist_placeholder, numbins)
        Total_dict[HNL_mass]["data"]=hist_data
    return Total_dict

def add_data_appended(Total_dict, hists_r1, hists_r3, numbins):
    """
    Given Total dict, r1 and r3 dicts and number of bins.
    Returns new Total dict with data added for both runs. 
    """
    for HNL_mass in Total_dict:
        r1_hist_placeholder = list(hists_r1[HNL_mass]["data"].values())
        r3_hist_placeholder = list(hists_r3[HNL_mass]["data"].values())
        r1_hist = remove_part_hist(r1_hist_placeholder, numbins)
        r3_hist = remove_part_hist(r3_hist_placeholder, numbins)
        appended = r1_hist+r3_hist
        Total_dict[HNL_mass]["data"]=appended
    return Total_dict

def add_data_merged_appended(Total_dict, hists_r1, hists_r3, numbins):
    """
    Given Total dict, r1 and r3 dicts and number of bins.
    Returns new Total dict with data added for both runs. 
    """
    merge_list = ["1shr_KDIF", "1shr_KDAR", "2shr_KDIF", "1shr_KDAR"]
    
    r1_merged, r3_merged = {}, {}
    Total_merged = {}
    
    all_keys = list(hists_r1.keys())
    first_key = all_keys[0]
    
    all_masses=[]
    #Looping over 100_KDIF_1shr
    for key in all_keys:
        all_masses.append(key.split("_")[0])
        
    unique_masses = set(all_masses)
        
    merge_list = ["1shr_KDIF", "1shr_KDAR", "2shr_KDIF", "1shr_KDAR"]
    #Looping over 100_KDIF_1shr
    for key in all_keys:
        r1_merged, r3_merged = {}, {}
        for mass_val in unique_masses:
            list_placeholder_r1, list_placeholder_r3 = [], []
            for name in merge_list:
                vals_r1 = hists_r1[f"{mass_val}_{name}"]["data"].values()
                vals_r3 = hists_r3[f"{mass_val}_{name}"]["data"].values()
                r1_hist_sliced = remove_part_hist(list(vals_r1), numbins)
                r3_hist_sliced = remove_part_hist(list(vals_r3), numbins)
                list_placeholder_r1.append(r1_hist_sliced)
                list_placeholder_r3.append(r3_hist_sliced)
            Merged_r1 = append_list_of_lists(list_placeholder_r1)
            Merged_r3 = append_list_of_lists(list_placeholder_r3)
            # Merged_dict[mass_val]=Merged
            r1_merged[mass_val]=Merged_r1
            r3_merged[mass_val]=Merged_r3
    
    for HNL_mass in Total_dict:
        r1_hist_placeholder = list(r1_merged[str(HNL_mass)])
        r3_hist_placeholder = list(r3_merged[str(HNL_mass)])
        # r1_hist = remove_part_hist(r1_hist_placeholder, numbins)
        # r3_hist = remove_part_hist(r3_hist_placeholder, numbins)
        appended = r1_hist_placeholder+r3_hist_placeholder
        Total_dict[HNL_mass]["data"]=appended
    return Total_dict

# merge_list = ["1shr_KDIF", "1shr_KDAR", "2shr_KDIF", "1shr_KDAR"]

def create_stat_unc_safe_hist(Total_dict):
    """
    Given total dict, which contains the stat uncertainties for TOTAL signal and TOTAL bkg.
    Returns dicts of the stat unc for signal and bkg. For use in models.
    Stops bins with zero stat error giving inf values. 
    """
    sig_stat, bkg_stat = {}, {}
    for HNL_mass in Total_dict['TOT_SIGNAL_VALS']:
        sig_stat[HNL_mass] = np.zeros_like(np.array(Total_dict['TOT_SIGNAL_VALS'][HNL_mass]))
        bkg_stat[HNL_mass] = np.zeros_like(np.array(Total_dict['TOT_BKG_VALS'][HNL_mass]))
        for i, val in enumerate(Total_dict['TOT_SIGNAL_VALS'][HNL_mass]):
            if val == 0: sig_stat[HNL_mass][i] = 0.0
            else: sig_stat[HNL_mass][i] = Total_dict['SIGNAL_STAT'][HNL_mass][i]/Total_dict['TOT_SIGNAL_VALS'][HNL_mass][i]
            
        for i, val in enumerate(Total_dict['TOT_BKG_VALS'][HNL_mass]):
            if val == 0: bkg_stat[HNL_mass][i] = 0.0
            else: bkg_stat[HNL_mass][i] = Total_dict['BKG_STAT'][HNL_mass][i]/Total_dict['TOT_BKG_VALS'][HNL_mass][i]
        sig_stat[HNL_mass], bkg_stat[HNL_mass] = list(sig_stat[HNL_mass]), list(bkg_stat[HNL_mass])
        # sig_stat[HNL_mass] = list(np.divide(np.array(Total_dict[HNL_mass]['SIGNAL_STAT']),np.array(Total_dict[HNL_mass]['SIGNAL_dict'])))
        # bkg_stat[HNL_mass] = list(np.divide(np.array(Total_dict[HNL_mass]['BKG_STAT']),np.array(Total_dict[HNL_mass]['BKG_dict'])))
    return sig_stat, bkg_stat

def create_stat_unc_safe(Total_dict):
    """
    Given total dict, which contains the stat uncertainties for TOTAL signal and TOTAL bkg.
    Returns dicts of the stat unc for signal and bkg. For use in models.
    Stops bins with zero stat error giving inf values. 
    """
    sig_stat, bkg_stat = {}, {}
    for HNL_mass in Total_dict:
        sig_stat[HNL_mass] = np.zeros_like(np.array(Total_dict[HNL_mass]['TOT_SIGNAL_VALS']))
        bkg_stat[HNL_mass] = np.zeros_like(np.array(Total_dict[HNL_mass]['TOT_BKG_VALS']))
        for i, val in enumerate(Total_dict[HNL_mass]['TOT_SIGNAL_VALS']):
            if val == 0: sig_stat[HNL_mass][i] = 0.0
            else: sig_stat[HNL_mass][i] = Total_dict[HNL_mass]['SIGNAL_STAT'][i]/Total_dict[HNL_mass]['TOT_SIGNAL_VALS'][i]
            
        for i, val in enumerate(Total_dict[HNL_mass]['TOT_BKG_VALS']):
            if val == 0: bkg_stat[HNL_mass][i] = 0.0
            else: bkg_stat[HNL_mass][i] = Total_dict[HNL_mass]['BKG_STAT'][i]/Total_dict[HNL_mass]['TOT_BKG_VALS'][i]
        sig_stat[HNL_mass], bkg_stat[HNL_mass] = list(sig_stat[HNL_mass]), list(bkg_stat[HNL_mass])
        # sig_stat[HNL_mass] = list(np.divide(np.array(Total_dict[HNL_mass]['SIGNAL_STAT']),np.array(Total_dict[HNL_mass]['SIGNAL_dict'])))
        # bkg_stat[HNL_mass] = list(np.divide(np.array(Total_dict[HNL_mass]['BKG_STAT']),np.array(Total_dict[HNL_mass]['BKG_dict'])))
    return sig_stat, bkg_stat

def create_individual_stat_unc_safe(Total_dict):
    """
    Given total dict, which contains the stat uncertainties for TOTAL signal, overlay, dirt and beamoff.
    Returns dicts of the stat unc for signal and bkgs. For use in models.
    Stops bins with zero stat error giving inf values. 
    """
    sig_stat, overlay_stat, dirt_stat, beamoff_stat = {}, {}, {}, {}
    for HNL_mass in Total_dict:
        sig_stat[HNL_mass] = np.zeros_like(np.array(Total_dict[HNL_mass]['TOT_SIGNAL_VALS']))
        overlay_stat[HNL_mass] = np.zeros_like(np.array(Total_dict[HNL_mass]['OVERLAY_VALS']))
        dirt_stat[HNL_mass] = np.zeros_like(np.array(Total_dict[HNL_mass]['DIRT_VALS']))
        beamoff_stat[HNL_mass] = np.zeros_like(np.array(Total_dict[HNL_mass]['BEAMOFF_VALS']))
        for i, val in enumerate(Total_dict[HNL_mass]['TOT_SIGNAL_VALS']):
            if val == 0: sig_stat[HNL_mass][i] = 0.0
            else: sig_stat[HNL_mass][i] = Total_dict[HNL_mass]['SIGNAL_STAT'][i]/Total_dict[HNL_mass]['TOT_SIGNAL_VALS'][i]
            
        for i, val in enumerate(Total_dict[HNL_mass]['OVERLAY_VALS']):
            if val == 0: overlay_stat[HNL_mass][i] = 0.0
            else: overlay_stat[HNL_mass][i] = Total_dict[HNL_mass]['OVERLAY_STAT'][i]/Total_dict[HNL_mass]['OVERLAY_VALS'][i]
        for i, val in enumerate(Total_dict[HNL_mass]['DIRT_VALS']):
            if val == 0: dirt_stat[HNL_mass][i] = 0.0
            else: dirt_stat[HNL_mass][i] = Total_dict[HNL_mass]['DIRT_STAT'][i]/Total_dict[HNL_mass]['DIRT_VALS'][i]
        for i, val in enumerate(Total_dict[HNL_mass]['BEAMOFF_VALS']):
            if val == 0: beamoff_stat[HNL_mass][i] = 0.0
            else: beamoff_stat[HNL_mass][i] = Total_dict[HNL_mass]['BEAMOFF_STAT'][i]/Total_dict[HNL_mass]['BEAMOFF_VALS'][i]
        
        sig_stat[HNL_mass], overlay_stat[HNL_mass] = list(sig_stat[HNL_mass]), list(overlay_stat[HNL_mass])
        dirt_stat[HNL_mass], beamoff_stat[HNL_mass] = list(dirt_stat[HNL_mass]), list(beamoff_stat[HNL_mass])
        
    return sig_stat, overlay_stat, dirt_stat, beamoff_stat

def add_data_merge_channel(Total_dict, hists_r1, hists_r3, numbins, channel = 'both'):
    """
    Given Total dict, r1 and r3 dicts and number of bins.
    Returns new Total dict with data added for both runs. 
    """
    
    if channel == 'both' or channel == 'run1' or channel == 'run3':
        merge_list = ["1shr_KDIF", "1shr_KDAR", "2shr_KDIF", "1shr_KDAR"]
    elif channel == 'kdif':
        merge_list = ["1shr_KDIF", "2shr_KDIF"]
    elif channel == 'kdar':
        merge_list = ["1shr_KDAR", "2shr_KDAR"]
    elif channel == '1shr':
        merge_list = ["1shr_KDAR", "1shr_KDIF"]
    elif channel == '2shr':
        merge_list = ["2shr_KDAR", "2shr_KDIF"]
    else: return print("Channel not recognised.")

    r1_merged, r3_merged = {}, {}
    Total_merged = {}
    
    all_keys = list(hists_r1.keys())
    first_key = all_keys[0]
    
    all_masses=[]
    #Looping over 100_KDIF_1shr
    for key in all_keys:
        all_masses.append(key.split("_")[0])
        
    unique_masses = set(all_masses)

    #Looping over 100_KDIF_1shr
    for key in all_keys:
        r1_merged, r3_merged = {}, {}
        for mass_val in unique_masses:
            list_placeholder_r1, list_placeholder_r3 = [], []
            for name in merge_list:
                vals_r1 = hists_r1[f"{mass_val}_{name}"]["data"].values()
                vals_r3 = hists_r3[f"{mass_val}_{name}"]["data"].values()
                r1_hist_sliced = remove_part_hist(list(vals_r1), numbins)
                r3_hist_sliced = remove_part_hist(list(vals_r3), numbins)
                list_placeholder_r1.append(r1_hist_sliced)
                list_placeholder_r3.append(r3_hist_sliced)
            Merged_r1 = append_list_of_lists(list_placeholder_r1)
            Merged_r3 = append_list_of_lists(list_placeholder_r3)
            # Merged_dict[mass_val]=Merged
            r1_merged[mass_val]=Merged_r1
            r3_merged[mass_val]=Merged_r3
    
    for HNL_mass in Total_dict:
        r1_hist_placeholder = list(r1_merged[str(HNL_mass)])
        r3_hist_placeholder = list(r3_merged[str(HNL_mass)])
        # r1_hist = remove_part_hist(r1_hist_placeholder, numbins)
        # r3_hist = remove_part_hist(r3_hist_placeholder, numbins)
        if channel == 'run1':
            appended = r1_hist_placeholder
        elif channel ==  'run3':
            appended = r3_hist_placeholder
        else:
            appended = r1_hist_placeholder+r3_hist_placeholder
        
        
        Total_dict[HNL_mass]["data"]=appended
    return Total_dict

def create_stat_unc_safe_hist_inv_index(Total_dict):
    """
    Given total dict, which contains the stat uncertainties for TOTAL signal and TOTAL bkg.
    Returns dicts of the stat unc for signal and bkg. For use in models.
    Stops bins with zero stat error giving inf values. 
    """
    sig_stat, bkg_stat = {}, {}
    for HNL_mass in Total_dict.keys():
        sig_stat[HNL_mass] = np.zeros_like(np.array(Total_dict[HNL_mass]['TOT_SIGNAL_VALS']))
        bkg_stat[HNL_mass] = np.zeros_like(np.array(Total_dict[HNL_mass]['TOT_BKG_VALS']))
        for i, val in enumerate(Total_dict[HNL_mass]['TOT_SIGNAL_VALS']):
            if val == 0: sig_stat[HNL_mass][i] = 0.0
            else: sig_stat[HNL_mass][i] = Total_dict[HNL_mass]['SIGNAL_STAT'][i]/Total_dict[HNL_mass]['TOT_SIGNAL_VALS'][i]
            
        for i, val in enumerate(Total_dict[HNL_mass]['TOT_BKG_VALS']):
            if val == 0: bkg_stat[HNL_mass][i] = 0.0
            else: bkg_stat[HNL_mass][i] = Total_dict[HNL_mass]['BKG_STAT'][i]/Total_dict[HNL_mass]['TOT_BKG_VALS'][i]
        sig_stat[HNL_mass], bkg_stat[HNL_mass] = list(sig_stat[HNL_mass]), list(bkg_stat[HNL_mass])
        # sig_stat[HNL_mass] = list(np.divide(np.array(Total_dict[HNL_mass]['SIGNAL_STAT']),np.array(Total_dict[HNL_mass]['SIGNAL_dict'])))
        # bkg_stat[HNL_mass] = list(np.divide(np.array(Total_dict[HNL_mass]['BKG_STAT']),np.array(Total_dict[HNL_mass]['BKG_dict'])))
    return sig_stat, bkg_stat

def scale_signal(Total_dict, theta_dict, scaling_dict={}):
    """
    Scales the number of events by the number in the scaling dict.
    Returns the new dict of histograms and the new thetas.
    """
    if(scaling_dict=={}): raise Exception("Specify scalings")
    Total_dict_scaled, new_theta_dict = copy.deepcopy(Total_dict), {}
    
    for HNL_mass in Total_dict.keys():
        HNL_mass_label = int(HNL_mass.split("_")[0])
        new_signal_hist = np.array(Total_dict[HNL_mass]['TOT_SIGNAL_VALS'])*scaling_dict[HNL_mass_label]
        new_signal_err_hist = np.array(Total_dict[HNL_mass]['TOT_SIGNAL_ERR'])*scaling_dict[HNL_mass_label]
        new_signal_stat_err = np.array(Total_dict[HNL_mass]['SIGNAL_STAT'])*scaling_dict[HNL_mass_label]
        new_signal_shapesys = np.array(Total_dict[HNL_mass]['SIGNAL_SHAPESYS'])*scaling_dict[HNL_mass_label]
        new_signal_detvar = np.array(Total_dict[HNL_mass]['SIGNAL_DETVAR'])*scaling_dict[HNL_mass_label]
        new_theta = theta_dict[HNL_mass_label]*scaling_dict[HNL_mass_label]**(1/4) # Number of events is proportional to theta**4
        
        Total_dict_scaled[HNL_mass]['TOT_SIGNAL_VALS'] = list(new_signal_hist)
        Total_dict_scaled[HNL_mass]['TOT_SIGNAL_ERR'] = list(new_signal_err_hist)
        Total_dict_scaled[HNL_mass]['SIGNAL_STAT'] = list(new_signal_stat_err)
        Total_dict_scaled[HNL_mass]['SIGNAL_SHAPESYS'] = list(new_signal_shapesys)
        Total_dict_scaled[HNL_mass]['SIGNAL_DETVAR'] = list(new_signal_detvar)
        
        new_theta_dict[HNL_mass] = new_theta
        
    return Total_dict_scaled, new_theta_dict

def scale_signal_merged(Total_dict, theta_dict, scaling_dict={}):
    """
    Scales the number of events by the number in the scaling dict.
    Returns the new dict of histograms and the new thetas.
    """
    if(scaling_dict=={}): raise Exception("Specify scalings")
    Total_dict_scaled, new_theta_dict = copy.deepcopy(Total_dict), {}
    
    for HNL_mass in Total_dict.keys():
        # HNL_mass_label = int(HNL_mass.split("_")[0])
        new_signal_hist = np.array(Total_dict[HNL_mass]['TOT_SIGNAL_VALS'])*scaling_dict[HNL_mass]
        new_signal_err_hist = np.array(Total_dict[HNL_mass]['TOT_SIGNAL_ERR'])*scaling_dict[HNL_mass]
        new_signal_stat_err = np.array(Total_dict[HNL_mass]['SIGNAL_STAT'])*scaling_dict[HNL_mass]
        new_signal_shapesys = np.array(Total_dict[HNL_mass]['SIGNAL_SHAPESYS'])*scaling_dict[HNL_mass]
        new_signal_detvar = np.array(Total_dict[HNL_mass]['SIGNAL_DETVAR'])*scaling_dict[HNL_mass]
        new_theta = theta_dict[HNL_mass]*scaling_dict[HNL_mass]**(1/4) # Number of events is proportional to theta**4
        
        Total_dict_scaled[HNL_mass]['TOT_SIGNAL_VALS'] = list(new_signal_hist)
        Total_dict_scaled[HNL_mass]['TOT_SIGNAL_ERR'] = list(new_signal_err_hist)
        Total_dict_scaled[HNL_mass]['SIGNAL_STAT'] = list(new_signal_stat_err)
        Total_dict_scaled[HNL_mass]['SIGNAL_SHAPESYS'] = list(new_signal_shapesys)
        Total_dict_scaled[HNL_mass]['SIGNAL_DETVAR'] = list(new_signal_detvar)
        
        new_theta_dict[HNL_mass] = new_theta
        
    return Total_dict_scaled, new_theta_dict

def create_model_dict_separated(Total_dict, Params_pyhf, stat_arr, debug=False):
    """
    Creating a model where the background samples are fed in individually (takes longer than summing into one total bkg hist).
    """
    sig_stat, overlay_stat, dirt_stat, beamoff_stat = stat_arr
    model_dict = {}
    sig_norm = {"hi": 1.0+Params_pyhf["Signal_flux_error"], "lo": 1.0-Params_pyhf["Signal_flux_error"]}
    dirt_norm = {"hi": 1.0+Params_pyhf["Flat_bkg_dirt_frac"], "lo": 1.0-Params_pyhf["Flat_bkg_dirt_frac"]}
    for HNL_mass in Total_dict:
        if(debug):print(HNL_mass)
        model_dict[HNL_mass] = pyhf.Model(
        {
      "channels": [
        {
          "name": "singlechannel",
          "samples": [
            {
              "name": "signal",
              "data": Total_dict[HNL_mass]['TOT_SIGNAL_VALS'],
              "modifiers": [
                {"name": "mu", "type": "normfactor", "data": None}, #This is the scaling which is to be scanned over in the hypo tests
                {"name": "stat_siguncrt", "type": "staterror", "data": sig_stat[HNL_mass]},
                {"name": "norm_siguncrt", "type": "normsys", "data": sig_norm}, #NuMI absorber KDAR rate
                {"name": "Detvar_sig", "type": "shapesys", "data": Total_dict[HNL_mass]['SIGNAL_DETVAR']} #shapesys assumes uncorrelated
              ]
            },
            {
              "name": "overlay",
              "data": Total_dict[HNL_mass]['OVERLAY_VALS'],
              "modifiers": [
                {"name": "stat_overlay", "type": "staterror", "data": overlay_stat[HNL_mass]},
                {"name": "Multisim_overlay", "type": "shapesys", "data": Total_dict[HNL_mass]['BKG_MULTISIM']},
                {"name": "Detvar_overlay", "type": "shapesys", "data": Total_dict[HNL_mass]['BKG_DETVAR']} #shapesys assumes uncorrelated
              ]
            },
            {
              "name": "dirt",
              "data": Total_dict[HNL_mass]['DIRT_VALS'],
              "modifiers": [
                {"name": "stat_dirt", "type": "staterror", "data": dirt_stat[HNL_mass]},
                {"name": "dirt_norm", "type": "normsys", "data": dirt_norm}
              ]
            },
            {
              "name": "beamoff",
              "data": Total_dict[HNL_mass]['BEAMOFF_VALS'],
              "modifiers": [
                {"name": "stat_beamoff", "type": "staterror", "data": beamoff_stat[HNL_mass]}
              ]
            }
          ]
        }
      ]
    }
    )
    return model_dict


def create_model_dict_same(Total_dict, Params_pyhf, debug=False):
    """
    Creating a model where the uncertainties are all enveloped in one shapesys modifier for signal and bkg.
    The total errors are taken from \"TOT_SIGNAL_ERR\" and \"TOT_BKG_ERR\" respectively.
    """
    
    model_dict = {}
    
    for HNL_mass in Total_dict:
        if(debug):print(HNL_mass)
        model_dict[HNL_mass] = pyhf.Model(
        {
      "channels": [
        {
          "name": "singlechannel",
          "samples": [
            {
              "name": "signal",
              "data": Total_dict[HNL_mass]['TOT_SIGNAL_VALS'],
              "modifiers": [
                {"name": "mu", "type": "normfactor", "data": None, }, #This is the scaling which is to be calculated
                {"name": "uncorr_siguncrt", "type": "shapesys", "data": Total_dict[HNL_mass]["TOT_SIGNAL_ERR"]}
              ]
            },
            {
              "name": "background",
              "data": Total_dict[HNL_mass]['TOT_BKG_VALS'],
              "modifiers": [
                {"name": "uncorr_bkguncrt", "type": "shapesys", "data": Total_dict[HNL_mass]["TOT_BKG_ERR"]}
              ]
            }
          ]
        }
      ],
      "parameters": [
            {
                "name": "mu",
                "bounds": [[0, 1000000]],
            }
      ],

    }
    )
    return model_dict

def main():

    parser = argparse.ArgumentParser(description='Choose mass points for limits calculation, and whether to use ALP or HPS scores')
    parser.add_argument('-m','--masses', nargs="+", type=int, help='List of mass points to calculate limits for')
    parser.add_argument('--alp', type=str, default='False', help='Boolean flag for using ALP scores')
    parser.add_argument('--root', type=str, default='/Users/user/PhD/HPS_uboone_analysis/', help='Root directory path')
    parser.add_argument('-n','--num_steps', type=int, default=100, help='Number of steps for the reweighting')
    parser.add_argument('-s', '--separated', type=str, default='False', help='Boolean flag for using separated channels')
    parser.add_argument('--run', type=str, default='all', help='Which run to use for limits calculation')
    parser.add_argument('--use_test', action=argparse.BooleanOptionalAction)

    
    args = parser.parse_args()
    root_dir = args.root
    num_steps = args.num_steps
    TEST = args.use_test
    run = args.run

    if not (TEST or not TEST):
        return ValueError('Test flag must be a boolean')

    if (not isinstance(args.masses, list)):
        raise ValueError('Mass point array must be a list or array of ints')
    else: masses = args.masses

    if num_steps < 3:
        raise ValueError('Number of steps for calculating limts must be greater than 2')

    if args.alp not in ['True', 'False', 'true', 'false', '0', '1']:
        raise ValueError('ALP flag must be a boolean')
    else:
        if args.alp in ['True', 'true', '1']:
            REWEIGHTING= True
        elif args.alp in ['False', 'false', '0']:
            REWEIGHTING = False

    if args.separated not in ['True', 'False', 'true', 'false', '0', '1']:
        raise ValueError('ALP flag must be a boolean')
    else:
        if args.separated in ['True', 'true', '1']:
            use_separated = True
        elif args.separated in ['False', 'false', '0']:
            use_separated = False

    print('Root:', root_dir)
    print('Masses:', masses)
    print('Run:', run)
    print('Num steps:', num_steps)
    print('ALP:', REWEIGHTING)
    print('Using test:', TEST)
    print('Separated:', use_separated)


    Params_pyhf = {"Stats_only":False,
               "Use_flat_sys":False,
               "Num_bins_for_calc":6,
               "Use_part_only":True,
               "Use_toys":False,
               "Num_toys":1000,
               "Flat_bkg_overlay_frac":0.3,
               "Flat_bkg_dirt_frac":0.75,
               "Flat_bkg_EXT_frac":0.0,
               "Flat_sig_detvar":0.2, 
               "Signal_flux_error":0.3, 
               "Overlay_detvar_frac":0.3,
               "Load_single_r1_file":False}


    all_HPS_masses = [100, 125, 130, 135, 140, 145, 150, 200]
    c_phi_dict  = {
                    100:	0.0027014149387605906, 
                    125: 	0.0018106609327994778, 
                    130:	0.0009990304425688081, 
                    135:	4.52690931768051e-06, 
                    140:	0.0008568518547569688, 
                    145:	0.0014326989318147938, 
                    150:	0.001798938228520414, 
                    200:	0.002864767955135017
				} 

    # Retrieving BDT scores from either reweighted or HPS directory
    
    if REWEIGHTING == True:
        hist_dict_run1, hist_dict_run3, theta_dict = New_Load_pyhf_files(f".root", Params_pyhf, location=root_dir+"BDT_RW_output/", HNL_masses=all_HPS_masses, use_test=TEST)
    else: 
        hist_dict_run1, hist_dict_run3, theta_dict = New_Load_pyhf_files(f".root", Params_pyhf, location=root_dir+"BDT_output/", HNL_masses=all_HPS_masses, use_test=TEST)

    zero_bins_errors_run1 = False_zero_bins(hist_dict_run1)
    zero_bins_errors_run3 = False_zero_bins(hist_dict_run3)

    TOT_R1_ERR = Full_calculate_total_uncertainty(Params_pyhf, hist_dict_run1, zero_bins_errors_run1)
    TOT_R3_ERR = Full_calculate_total_uncertainty(Params_pyhf, hist_dict_run3, zero_bins_errors_run3)

    R1_BKG, R1_SIGNAL = Add_bkg_hists_make_signal(hist_dict_run1)
    R3_BKG, R3_SIGNAL = Add_bkg_hists_make_signal(hist_dict_run3)

    R1_output = Make_into_lists(Params_pyhf, R1_BKG, R1_SIGNAL, TOT_R1_ERR)
    R3_output = Make_into_lists(Params_pyhf, R3_BKG, R3_SIGNAL, TOT_R3_ERR)

    Merged_R1 = Append_four_channels(R1_output)
    Merged_R3 = Append_four_channels(R3_output)

    Merged_KDIF_R1 = Append_kdif_channels(R1_output)
    Merged_KDAR_R1 = Append_kdar_channels(R1_output)
    Merged_KDIF_R3 = Append_kdif_channels(R3_output)
    Merged_KDAR_R3 = Append_kdar_channels(R3_output)
    Merged_1shr_R1 = Append_1shr_channels(R1_output)
    Merger_1shr_R3 = Append_1shr_channels(R3_output)
    Merged_2shr_R1 = Append_2shr_channels(R1_output)
    Merged_2shr_R3 = Append_2shr_channels(R3_output)


    list_input_dicts = [Merged_R1, Merged_R3]

    Total_dict_both = Create_final_appended_runs_dict(list_input_dicts)

    Total_dict_run1 = Create_final_appended_runs_dict([Merged_R1])
    Total_dict_run3 = Create_final_appended_runs_dict([Merged_R3])

    Total_dict_kdif = Create_final_appended_runs_dict([Merged_KDIF_R1, Merged_KDIF_R3])
    Total_dict_kdar = Create_final_appended_runs_dict([Merged_KDAR_R1, Merged_KDAR_R3])

    Total_dict_1shr = Create_final_appended_runs_dict([Merged_1shr_R1, Merger_1shr_R3])
    Total_dict_2shr = Create_final_appended_runs_dict([Merged_2shr_R1, Merged_2shr_R3])

    mass_order = [100, 125, 130, 135, 140, 145, 150, 200]

    Total_dict_both_copy = copy.deepcopy(Total_dict_both)
    Total_dict_run1_copy = copy.deepcopy(Total_dict_run1)
    Total_dict_run3_copy = copy.deepcopy(Total_dict_run3)
    Total_dict_kdif_copy = copy.deepcopy(Total_dict_kdif)
    Total_dict_kdar_copy = copy.deepcopy(Total_dict_kdar)
    Total_dict_1shr_copy = copy.deepcopy(Total_dict_1shr)
    Total_dict_2shr_copy = copy.deepcopy(Total_dict_2shr)

    ordered_dict, ordered_dict_run1, ordered_dict_run3, ordered_dict_kdif, ordered_dict_kdar, ordered_dict_1shr, ordered_dict_2shr = {}, {}, {}, {}, {}, {}, {}


    for mass in mass_order:
        ordered_dict[mass] = Total_dict_both_copy[mass]
        ordered_dict_run1[mass] = Total_dict_run1_copy[mass]
        ordered_dict_run3[mass] = Total_dict_run3_copy[mass]
        ordered_dict_kdif[mass] = Total_dict_kdif_copy[mass]
        ordered_dict_kdar[mass] = Total_dict_kdar_copy[mass]
        ordered_dict_1shr[mass] = Total_dict_1shr_copy[mass]
        ordered_dict_2shr[mass] = Total_dict_2shr_copy[mass]    

    if Params_pyhf["Use_part_only"]==True:
        NUMBINS = Params_pyhf["Num_bins_for_calc"]
    else: NUMBINS=30 #This will just give the full hist
        
    if 'data;1' in hist_dict_run1["150_1shr_KDAR"]:
        Total_dict_both=add_data_merge_channel(ordered_dict, hist_dict_run1, hist_dict_run3, NUMBINS, channel='both')
        Total_dict_run1=add_data_merge_channel(ordered_dict_run1, hist_dict_run1, hist_dict_run3, NUMBINS,channel='run1')
        Total_dict_run3=add_data_merge_channel(ordered_dict_run3, hist_dict_run1, hist_dict_run3, NUMBINS,channel='run3')
        Total_dict_both=add_data_merge_channel(ordered_dict, hist_dict_run1, hist_dict_run3, NUMBINS, channel='both')
        Total_dict_kdif=add_data_merge_channel(ordered_dict_kdif, hist_dict_run1, hist_dict_run3, NUMBINS, channel='kdif')
        Total_dict_kdar=add_data_merge_channel(ordered_dict_kdar, hist_dict_run1, hist_dict_run3, NUMBINS, channel='kdar')
        Total_dict_1shr=add_data_merge_channel(ordered_dict_1shr, hist_dict_run1, hist_dict_run3, NUMBINS, channel='1shr')
        Total_dict_2shr=add_data_merge_channel(ordered_dict_2shr, hist_dict_run1, hist_dict_run3, NUMBINS, channel='2shr')
        
    #Create separate dirt normalisation uncertainties for Run1 and Run3 
    for HNL_mass in ordered_dict:

        dirt_vals = ordered_dict[HNL_mass]['BKG_DIRT']
        numbins = int(len(dirt_vals)/2)
        r1_dirt = dirt_vals[:numbins] + list(np.zeros(numbins))
        r3_dirt = list(np.zeros(numbins)) + dirt_vals[numbins:] 
        ordered_dict[HNL_mass]['BKG_DIRT_R1'] = r1_dirt
        ordered_dict[HNL_mass]['BKG_DIRT_R3'] = r3_dirt
        ordered_dict_run1[HNL_mass]['BKG_DIRT_R1'] = r1_dirt
        ordered_dict_run3[HNL_mass]['BKG_DIRT_R3'] = r3_dirt

    sig_stat, bkg_stat = create_stat_unc_safe(ordered_dict)

    sig_stat_r1, bkg_stat_r1 = create_stat_unc_safe(ordered_dict_run1)
    sig_stat_r3, bkg_stat_r3 = create_stat_unc_safe(ordered_dict_run3)

    sig_stat_kdif, bkg_stat_kdif = create_stat_unc_safe(ordered_dict_kdif)
    sig_stat_kdar, bkg_stat_kdar = create_stat_unc_safe(ordered_dict_kdar)

    sig_stat_1shr, bkg_stat_1shr = create_stat_unc_safe(ordered_dict_1shr)
    sig_stat_2shr, bkg_stat_2shr = create_stat_unc_safe(ordered_dict_2shr)



    sig_stat_r1_hists, bkg_stat_r1_hists = create_stat_unc_safe_hist(R1_output)
    sig_stat_r3_hists, bkg_stat_r3_hists = create_stat_unc_safe_hist(R3_output)

    sig_stat_kdif_hists, bkg_stat_kdif_hists = create_stat_unc_safe_hist_inv_index(Total_dict_kdif)
    sig_stat_kdar_hists, bkg_stat_kdar_hists = create_stat_unc_safe_hist_inv_index(Total_dict_kdar)

    sig_stat_1shr_hists, bkg_stat_1shr_hists = create_stat_unc_safe_hist_inv_index(Total_dict_1shr)
    sig_stat_2shr_hists, bkg_stat_2shr_hists = create_stat_unc_safe_hist_inv_index(Total_dict_2shr)

    sig_stat, overlay_stat, dirt_stat, beamoff_stat = create_individual_stat_unc_safe(ordered_dict)

    scaling_dict = {100:1.0,125: 1.0, 130: 1.0, 135: 1.0, 140: 1.0, 145: 1.0,150:1.0,200:1.0} #Scaling for both r1 and r3 combined
    scaling_dict_r1 = {100:1.0,125: 1.0, 130: 1.0, 135: 1.0, 140: 1.0, 145: 1.0,150:1.0,200:1.0}
    scaling_dict_r3 = {100:1.0,125: 1.0, 130: 1.0, 135: 1.0, 140: 1.0, 145: 1.0,150:1.0,200:1.0}


    Total_dict, theta_dict_scaled  = scale_signal_merged(ordered_dict, theta_dict, scaling_dict)
    Total_dict_run1_scaled, theta_dict_scaled_r1  = scale_signal_merged(ordered_dict_run1, theta_dict, scaling_dict_r1)
    Total_dict_run3_scaled, theta_dict_scaled_r3  = scale_signal_merged(ordered_dict_run3, theta_dict, scaling_dict_r3)
    Total_dict_kdif_scaled, theta_dict_scaled_kdif  = scale_signal_merged(ordered_dict_kdif, theta_dict, scaling_dict)
    Total_dict_kdar_scaled, theta_dict_scaled_kdar  = scale_signal_merged(ordered_dict_kdar, theta_dict, scaling_dict)
    Total_dict_1shr_scaled, theta_dict_scaled_1shr  = scale_signal_merged(ordered_dict_1shr, theta_dict, scaling_dict)
    Total_dict_2shr_scaled, theta_dict_scaled_2shr  = scale_signal_merged(ordered_dict_2shr, theta_dict, scaling_dict)

    # In this section, the aim is to use the full, separated uncertainties of the model, which is significantly slower.
    # Therefore, the point of this section is to verify how much the results change when using the full uncertainties,
    # ideally being consistent with the quicker method.
    print(run)

    if not use_separated:
        model_dict_both = create_model_dict_same(Total_dict, Params_pyhf)

        model_dict_run1 = create_model_dict_same(Total_dict_run1_scaled, Params_pyhf)
        model_dict_run3 = create_model_dict_same(Total_dict_run3_scaled, Params_pyhf)

        model_dict_kdif = create_model_dict_same(Total_dict_kdif_scaled, Params_pyhf)
        model_dict_kdar = create_model_dict_same(Total_dict_kdar_scaled, Params_pyhf)

        model_dict_1shr = create_model_dict_same(Total_dict_1shr_scaled, Params_pyhf)
        model_dict_2shr = create_model_dict_same(Total_dict_2shr_scaled, Params_pyhf)

        if run == 'all':
            model_dict = model_dict_both #Quicker, uncertainties are entered as one uncertainty which is the quadsum of components.
        elif run == 'run1':
            model_dict = model_dict_run1
        elif run == 'run3':
            model_dict = model_dict_run3
        elif run == 'kdif':
            model_dict = model_dict_kdif
        elif run == 'kdar':
            model_dict = model_dict_kdar
        elif run == '1shr':
            model_dict = model_dict_1shr
        elif run == '2shr':
            model_dict = model_dict_2shr

        DATA_OBS_dict = {}

        for HNL_mass in model_dict:
            init_pars = model_dict[HNL_mass].config.suggested_init()
            model_dict[HNL_mass].expected_actualdata(init_pars) #signal plus bkg

            bkg_pars = init_pars.copy()
            bkg_pars[model_dict[HNL_mass].config.poi_index] = 0
            model_dict[HNL_mass].expected_actualdata(bkg_pars) #bkg only
            list_keys = list(Total_dict[HNL_mass].keys())

            if run == 'all':
                list_keys = list(Total_dict[HNL_mass].keys())
                if "data" in list_keys: #haven't made this yet, need to test
                    DATA_OBS_dict[HNL_mass] = Total_dict[HNL_mass]["data"]+model_dict[HNL_mass].config.auxdata
                else:
                    DATA_OBS_dict[HNL_mass] = Total_dict[HNL_mass]['TOT_BKG_VALS']+model_dict[HNL_mass].config.auxdata
                    
            elif run == 'run1':
                list_keys = list(Total_dict_run1[HNL_mass].keys())
                if 'data' in list_keys:
                    DATA_OBS_dict[HNL_mass] = Total_dict_run1[HNL_mass]["data"]+model_dict[HNL_mass].config.auxdata
                else:
                    DATA_OBS_dict[HNL_mass] = Total_dict_run1[HNL_mass]['TOT_BKG_VALS']+model_dict[HNL_mass].config.auxdata
            elif run == 'run3':
                list_keys = list(Total_dict_run3[HNL_mass].keys())
                if 'data' in list_keys:
                    DATA_OBS_dict[HNL_mass] = Total_dict_run3[HNL_mass]["data"]+model_dict[HNL_mass].config.auxdata
                else: 
                    DATA_OBS_dict[HNL_mass] = Total_dict_run3[HNL_mass]['TOT_BKG_VALS']+model_dict[HNL_mass].config.auxdata
            elif run == 'kdif':
                list_keys = list(Total_dict_run3[HNL_mass].keys())
                if 'data' in list_keys:
                    DATA_OBS_dict[HNL_mass] = Total_dict_kdif[HNL_mass]["data"]+model_dict[HNL_mass].config.auxdata
                else: 
                    DATA_OBS_dict[HNL_mass] = Total_dict_kdif[HNL_mass]['TOT_BKG_VALS']+model_dict[HNL_mass].config.auxdata
            elif run == 'kdar':
                list_keys = list(Total_dict_run3[HNL_mass].keys())
                if 'data' in list_keys:
                    DATA_OBS_dict[HNL_mass] = Total_dict_kdar[HNL_mass]["data"]+model_dict[HNL_mass].config.auxdata
                else: 
                    DATA_OBS_dict[HNL_mass] = Total_dict_kdar[HNL_mass]['TOT_BKG_VALS']+model_dict[HNL_mass].config.auxdata
            elif run == '1shr':
                list_keys = list(Total_dict_run3[HNL_mass].keys())
                if 'data' in list_keys:
                    DATA_OBS_dict[HNL_mass] = Total_dict_1shr[HNL_mass]["data"]+model_dict[HNL_mass].config.auxdata
                else: 
                    DATA_OBS_dict[HNL_mass] = Total_dict_1shr[HNL_mass]['TOT_BKG_VALS']+model_dict[HNL_mass].config.auxdata
            elif run == '2shr':
                list_keys = list(Total_dict_run3[HNL_mass].keys())
                if 'data' in list_keys:
                    DATA_OBS_dict[HNL_mass] = Total_dict_2shr[HNL_mass]["data"]+model_dict[HNL_mass].config.auxdata
                else:
                    DATA_OBS_dict[HNL_mass] = Total_dict_2shr[HNL_mass]['TOT_BKG_VALS']+model_dict[HNL_mass].config.auxdata
            else: print('Error creating DATA_OBS_dict')
            model_dict[HNL_mass].logpdf(pars=bkg_pars, data=DATA_OBS_dict[HNL_mass])

    else:
        model_dict = create_model_dict_separated(Total_dict, Params_pyhf, [sig_stat, overlay_stat, dirt_stat, beamoff_stat]) #create_model_dict_separated(Total_dict, Params_pyhf, [sig_stat, overlay_stat, dirt_stat, beamoff_stat])
    
        DATA_OBS_dict = {}
        for HNL_mass in model_dict:
            init_pars = model_dict[HNL_mass].config.suggested_init()
            model_dict[HNL_mass].expected_actualdata(init_pars) #signal plus bkg

            bkg_pars = init_pars.copy()
            bkg_pars[model_dict[HNL_mass].config.poi_index] = 0
            model_dict[HNL_mass].expected_actualdata(bkg_pars) #bkg only
            list_keys = list(Total_dict[HNL_mass].keys())
            
            list_keys = list(Total_dict[HNL_mass].keys())
            if "data" in list_keys: #haven't made this yet, need to test
                DATA_OBS_dict[HNL_mass] = Total_dict[HNL_mass]["data"]+model_dict[HNL_mass].config.auxdata
            else:
                DATA_OBS_dict[HNL_mass] = Total_dict[HNL_mass]['TOT_BKG_VALS']+model_dict[HNL_mass].config.auxdata



    #Should take ~10 mins for each with 5 bins, 100 mu values
    obs_limit_dict = {}
    exp_limits_dict = {}
    print("If the output of an upper limit calculation is equal to the lowest or highest value of poi, the range needs to be extended")

    # poi_values = np.linspace(1000, 10000,10) #Values of mu which are scanned over in hypo tests. Default: np.linspace(0.001, 2, 100)
    # print("Max value is " + str(max(poi_values)))
    # print("Min value is " + str(min(poi_values)))
    # print("Next value is " + str(poi_values[1]))
    # print("Next value is " + str(poi_values[2]) + "\n")

    print("-----Starting Hypothesis tests-----" + "\n")
    CL_level = 0.05
    print(f"CLs CL is {(1-CL_level)*100}%")

    if CL_level != 0.05: print("WARNING, the CL should usually be 95%, here it is not.")
    print()


    if REWEIGHTING == True:
        # dict needs to cover a larger range depending on mass, due to axion-pion mixing 
        # which suppresses leptonic decays near 135 MeV.
        poi_dict = {
            100:np.linspace(0.001, 2, num_steps),
            125:np.linspace(0.001, 3, num_steps),
            130:np.linspace(0.1, 100, num_steps),
            135:np.linspace(0.1, 10000, num_steps),
            140:np.linspace(0.1, 200, num_steps),
            145:np.linspace(0.001, 5, num_steps),
            150:np.linspace(0.001, 2, num_steps),
            200:np.linspace(0.001, 2, num_steps),
        }
    else: 
        poi_dict = {
            100:np.linspace(0.001, 2, num_steps),
            125:np.linspace(0.001, 2, num_steps),
            130:np.linspace(0.001, 2, num_steps),
            135:np.linspace(0.001, 2, num_steps),
            140:np.linspace(0.001, 2, num_steps),
            145:np.linspace(0.001, 2, num_steps),
            150:np.linspace(0.001, 2, num_steps),
            200:np.linspace(0.001, 2, num_steps),
        }

    for HNL_mass in masses:
        poi_values = poi_dict[HNL_mass]
        print("Mass point: " + str(HNL_mass) + "MeV")
        print("Max value is " + str(max(poi_values)))
        print("Min value is " + str(min(poi_values)))
        print("Next value is " + str(poi_values[1]))
        print("Next value is " + str(poi_values[2]) + "\n")

        obs_limit_dict[HNL_mass], exp_limits_dict[HNL_mass], (scan, results) = pyhf.infer.intervals.upper_limits.upper_limit(
            DATA_OBS_dict[HNL_mass], model_dict[HNL_mass], poi_values, level=CL_level, return_results=True
        )

        print(f"Upper limit {HNL_mass}MeV (obs): μ = {obs_limit_dict[HNL_mass]:.6f}")
        print(f"Upper limit {HNL_mass}MeV (exp): μ = {exp_limits_dict[HNL_mass][2]:.6f}" + "\n")


    # obs_limit_dict[HNL_mass]

    exp_limit = [] #entry 2
    exp_1sig_up = [] #entry 3
    exp_1sig_down = [] #entry 1
    exp_2sig_up = [] #entry 4
    exp_2sig_down = [] #entry 0
    obs_limit = []

    for HNL_mass in obs_limit_dict:
        if REWEIGHTING == True:
            theta = c_phi_dict[HNL_mass]
        else:
            theta = theta_dict_scaled[HNL_mass]

        #N events scales with mixing angle squared
        obs_limit.append(theta*np.sqrt(obs_limit_dict[HNL_mass]))
        exp_limit.append(theta*np.sqrt(exp_limits_dict[HNL_mass][2]))

        exp_1sig_up.append(theta*np.sqrt(exp_limits_dict[HNL_mass][3]))
        exp_2sig_up.append(theta*np.sqrt(exp_limits_dict[HNL_mass][4]))
        exp_1sig_down.append(theta*np.sqrt(exp_limits_dict[HNL_mass][1]))
        exp_2sig_down.append(theta*np.sqrt(exp_limits_dict[HNL_mass][0]))



    if REWEIGHTING == True:
        save_loc = "limit_files/RW_Brazil_plot/"
    else: 
        save_loc = "limit_files/Brazil_plot/"
    to_save_names = ["exp_1sig_up","exp_1sig_down","exp_2sig_up","exp_2sig_down","exp_limit","obs_limit"]
    to_save_lists = [exp_1sig_up,exp_1sig_down,exp_2sig_up,exp_2sig_down,exp_limit,obs_limit]

    if use_separated == True:
        filename=f"_separated_ALL_Capped_HPS_rebinned_limit.csv"
    else:  
        if run == 'all':
            filename=f"_ALL_Capped_HPS_rebinned_limit.csv"
        elif run == 'run1':
            filename=f"_Run1_Capped_HPS_rebinned_limit.csv"
        elif run == 'run3':
            filename=f"_Run3_Capped_HPS_rebinned_limit.csv"
        elif run == 'kdif':
            filename=f"_KDIF_Capped_HPS_rebinned_limit.csv"
        elif run == 'kdar':
            filename=f"_KDAR_Capped_HPS_rebinned_limit.csv"
        elif run == '1shr':
            filename=f"_1shr_Capped_HPS_rebinned_limit.csv"
        elif run == '2shr':
            filename=f"_2shr_Capped_HPS_rebinned_limit.csv"

    if TEST == True:
        filename = f"_TEST"+filename

    for i, lim in enumerate(to_save_lists):

        r = zip(masses, lim)
        savestr=to_save_names[i]
        savename = root_dir+save_loc+savestr+filename
        with open(savename, "w") as s:
            w = csv.writer(s)
            for row in r:
                w.writerow(row)

    print("Last saved is " + savename)
    return 

main()
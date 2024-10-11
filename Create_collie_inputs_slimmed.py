import pandas as pd
import matplotlib.pyplot as plt
import numpy
import numpy as np
import math
import re
from matplotlib.lines import Line2D
from scipy.special import logit



def get_frac_ydata_std_dev_detsys(df, var, bins, plotRange, bdt_score_str='0.0'):

    
    y_CV, bin_CV = numpy.histogram(df.query('pred_CV >='+bdt_score_str)[var+'_CV'], bins=bins, range = plotRange)
    y_ly_atten,_ = numpy.histogram(df.query('pred_LYAttenuation >='+bdt_score_str)[var+'_LYAttenuation'], bins=bins, range = plotRange)

    y_ly_down,_ = numpy.histogram(df.query('pred_LYDown >='+bdt_score_str)[var+'_LYDown'], bins=bins, range = plotRange)
    y_ly_rayleigh,_ = numpy.histogram(df.query('pred_LYRayleigh >='+bdt_score_str)[var+'_LYRayleigh'], bins=bins, range = plotRange)
    y_recomb,_ = numpy.histogram(df.query('pred_Recomb2 >='+bdt_score_str)[var+'_Recomb2'], bins=bins, range = plotRange)
    y_sce,_  = numpy.histogram(df.query('pred_SCE >='+bdt_score_str)[var+'_SCE'], bins=bins, range = plotRange)
    y_wiremod_anglexz,_ = numpy.histogram(df.query('pred_WireModThetaXZ >='+bdt_score_str)[var+'_WireModThetaXZ'], bins=bins, range = plotRange)
    y_wiremod_angleyz,_ = numpy.histogram(df.query('pred_WireModThetaYZ >='+bdt_score_str)[var+'_WireModThetaYZ'], bins=bins, range = plotRange)
    y_wiremod_dEdx,_ = numpy.histogram(df.query('pred_WireModdEdX >='+bdt_score_str)[var+'_WireModdEdX'], bins=bins, range = plotRange)
    y_wiremod_x,_ = numpy.histogram(df.query('pred_WireModX >='+bdt_score_str)[var+'_WireModX'], bins=bins, range = plotRange)
    y_wiremod_yz,_ = numpy.histogram(df.query('pred_WireModYZ >='+bdt_score_str)[var+'_WireModYZ'], bins=bins, range = plotRange)


    y_all_arr = np.array([y_ly_atten, y_ly_down, y_ly_rayleigh, y_recomb, y_sce, y_wiremod_anglexz, y_wiremod_angleyz, y_wiremod_dEdx, y_wiremod_x, y_wiremod_yz, y_CV])

    ydata_std_dev = std_dev (y_all_arr)
    frac_ydata_std_dev_detsys_bkg = ydata_std_dev/y_CV
    
    if (y_CV == 0).any():
        if type(bins) is int:
            if bins == 1:
                print ('Not Working!! Tried with one bin for Detsys BACKGROUND but not working and therefore returning 1')
                return 1.0
        else:
            frac_ydata_std_dev_detsys_bkg = get_frac_ydata_std_dev_detsys(df, var, bins=1, plotRange=plotRange, bdt_score_str='0.0')
        
        
    #frac_ydata_std_dev_detsys_bkg[frac_ydata_std_dev_detsys_bkg == inf] = 1
    return frac_ydata_std_dev_detsys_bkg

def get_ydata_std_dev_reint(df_mc, var, nbins, plotRange):
    
    ydata_std_dev = []
    df_data_good = df_mc.query('is_good_reint == 1').copy()
    df_data_bad = df_mc.query('is_good_reint != 1').copy()
    
    pot_scale = 1
    
    weights_good = df_data_good['scale_factor_ls']*pot_scale*df_data_good['bnb_scale_ls']*df_data_good['ppfx_cv_ls']*df_data_good['weightSplineTimesTune_ls']
    weights_bad = df_data_bad['scale_factor_ls']*pot_scale*df_data_bad['bnb_scale_ls']*df_data_bad['ppfx_cv_ls']*df_data_bad['weightSplineTimesTune_ls']
    
    a_good, _ = numpy.histogram(df_data_good[var], weights = weights_good, bins=nbins, range = plotRange)
    a_bad, _ = numpy.histogram(df_data_bad[var], weights = weights_bad, bins=nbins, range = plotRange)


    a_good_arr, b_good_arr = get_n_bins_uni(df_data_good, var, nbins, plotRange, flux_xsec_str = 'reint')
    a_good_arr.append(a_good) # Since need the last value which will be CV in def std_dev () method
    
    ydata_std_dev_good = std_dev(a_good_arr)
    ydata_std_dev_bad = a_bad*0.3
    
    
    for sig_good, sig_bad, a_good_i, a_bad_i in zip(ydata_std_dev_good, ydata_std_dev_bad, a_good, a_bad):

        tot = a_good_i+a_bad_i
        
        #ydata_std_dev.append(math.sqrt((sig_good*a_good_i/tot)**2 + (sig_bad*a_bad_i/tot)**2))
        ydata_std_dev.append(math.sqrt((sig_good)**2 + (sig_bad)**2))
    
    #a_arr, b_arr = get_n_bins_uni(df_mc.copy(), var, nbins, plotRange, flux_xsec_str = 'reint')
    
    
    
    
    # CV is going to be the same for all.  
    #weights_sig_cv = df_mc['scale_factor_ls']*pot_scale*df_mc['bnb_scale_ls']*df_mc['ppfx_cv_ls']*df_mc['weightSplineTimesTune_ls']
    
    #a,b = numpy.histogram(df_mc[var], weights = weights_sig_cv, bins=nbins, range = plotRange)

    
    #a_arr.append(a)
    #b_arr.append(b)

    #ydata_std_dev = std_dev(a_arr)
    
    return ydata_std_dev

def get_ydata_std_dev_xsec(df_data, var, nbins, plotRange):
    # No need for correcting for KL flux here as that 100% uncertainty is only for KL events
    
    ydata_std_dev = []
    
    df_data_good = df_data.query('is_good_genie == 1').copy()
    df_data_bad = df_data.query('is_good_genie != 1').copy()
    

    pot_scale = 1
    #bnb_scale_ls = 0.759
    
    weights_good = df_data_good['scale_factor_ls']*pot_scale*df_data_good['bnb_scale_ls']*df_data_good['ppfx_cv_ls']*df_data_good['weightSplineTimesTune_ls']
    weights_bad = df_data_bad['scale_factor_ls']*pot_scale*df_data_bad['bnb_scale_ls']*df_data_bad['ppfx_cv_ls']*df_data_bad['weightSplineTimesTune_ls']
    
   

    a_good, _ = numpy.histogram(df_data_good[var], weights = weights_good, bins=nbins, range = plotRange)
    a_bad, _ = numpy.histogram(df_data_bad[var], weights = weights_bad, bins=nbins, range = plotRange)

    
    
    
    a_good_arr, b_good_arr = get_n_bins_uni(df_data_good, var, nbins, plotRange, flux_xsec_str = 'xsec')
    a_good_arr.append(a_good) # Since need the last value which will be CV in def std_dev () method
    
    
    ydata_std_dev_good = std_dev(a_good_arr)
    #ydata_std_dev_bad = a_bad
    ydata_std_dev_bad = a_bad

    for sig_good, sig_bad, a_good_i, a_bad_i in zip(ydata_std_dev_good, ydata_std_dev_bad, a_good, a_bad):

        tot = a_good_i+a_bad_i
        
        #ydata_std_dev.append(math.sqrt((sig_good*a_good_i/tot)**2 + (sig_bad*a_bad_i/tot)**2))
        ydata_std_dev.append(math.sqrt((sig_good)**2 + (sig_bad)**2))

    print ('**************** Final standard deviation (sigma)..............****************')
    print(ydata_std_dev)
    print ('******************************************************************************') 


    return ydata_std_dev

def get_ydata_std_dev_ppfx (df_data, var, nbins, plotRange):
    
    #display(df_data)
    
    ydata_std_dev = []

    df_data_KL = df_data.query('nu_parent_pdg_ls==130').copy()
    df_data_non_KL = df_data.query('nu_parent_pdg_ls!=130').copy()
    
    
    df_data_KL_good = df_data_KL.query('is_good_ppfx == 1').copy()
    df_data_KL_bad = df_data_KL.query('is_good_ppfx != 1').copy()
    
    df_data_non_KL_good = df_data_non_KL.query('is_good_ppfx == 1').copy()
    df_data_non_KL_bad = df_data_non_KL.query('is_good_ppfx != 1').copy()
    
    
    pot_scale = 1
    
    
    weights_KL_good = df_data_KL_good['scale_factor_ls']*pot_scale*df_data_KL_good['bnb_scale_ls']*df_data_KL_good['ppfx_cv_ls']*df_data_KL_good['weightSplineTimesTune_ls']
    weights_KL_bad = df_data_KL_bad['scale_factor_ls']*pot_scale*df_data_KL_bad['bnb_scale_ls']*df_data_KL_bad['ppfx_cv_ls']*df_data_KL_bad['weightSplineTimesTune_ls']


    weights_non_KL_good = df_data_non_KL_good['scale_factor_ls']*pot_scale*df_data_non_KL_good['bnb_scale_ls']*df_data_non_KL_good['ppfx_cv_ls']*df_data_non_KL_good['weightSplineTimesTune_ls']
    weights_non_KL_bad = df_data_non_KL_bad['scale_factor_ls']*pot_scale*df_data_non_KL_bad['bnb_scale_ls']*df_data_non_KL_bad['ppfx_cv_ls']*df_data_non_KL_bad['weightSplineTimesTune_ls']


    
    # Do we have to include the weight or not? Including weight reduces uncertainty due to larger stat no?
    a_KL_good, _ = numpy.histogram(df_data_KL_good[var], weights = weights_KL_good, bins=nbins, range = plotRange)
    
    #plt.hist(df_data_KL_good[var], weights = weights_KL_good, bins=nbins, range = plotRange)
    print ('==================================')
    print ('Following is height for KL good')
    print (a_KL_good)
    
    a_KL_bad, _ = numpy.histogram(df_data_KL_bad[var], weights = weights_KL_bad, bins=nbins, range = plotRange)
    print ('Following is height for KL bad')
    print (a_KL_bad)


    a_non_KL_good, _ = numpy.histogram(df_data_non_KL_good[var], weights = weights_non_KL_good, bins=nbins, range = plotRange)
    print ('Following is height for non KL events that are good')
    print (a_non_KL_good)
    
    
    a_non_KL_bad, _ = numpy.histogram(df_data_non_KL_bad[var], weights = weights_non_KL_bad, bins=nbins, range = plotRange)
    print ('Following is height for non KL events that are bad')
    print (a_non_KL_bad)
    print ('==================================')
    """
    I have uncertainty of 100% for KL events and 100% for NC Coherent Pi0 (aka bad) events. 
    Those events that are not NC Coherent are good events. In total I will have 4 histograms for a sample
    
    1. KL good. The uncertainty for which will be height of the bin as it’s 100% due to it being a KL event histogram.
    2. KL bad. The uncertainty for which will be sqrt(2)*height of the bin right?
    3. non KL good. RMS uncertainty for 600 universes
    4. non KL bad. 100% due to it being a bad event histogram
    
    then add them in quadrature for each bin
    
    Caution: NC Coherent Pi0 is only applied to xsec and not the ppfx so needs to fix it as this function
    is onl. I think I have fixed this. Not 100% sure
    
    I think what I'm doing here is 
    
    """

    ydata_std_dev_KL_good = a_KL_good
    ydata_std_dev_KL_bad = math.sqrt(2)*a_KL_bad # Think of it like a complete different histogram. It will have fraction of height

    # I need all the 600 universes for a_non_KL to be able to calculate the uncertainty.
    a_non_KL_good_arr, b_non_KL_good_arr = get_n_bins_uni(df_data_non_KL_good, var, nbins, plotRange, flux_xsec_str = 'ppfx')
    a_non_KL_good_arr.append(a_non_KL_good) # Since need the last value which will be CV in def std_dev () method
    
    
    ydata_std_dev_non_KL_good = std_dev(a_non_KL_good_arr)
    
    
    ydata_std_dev_non_KL_bad = a_non_KL_bad
    
    
    print ('Sigma KL good')
    print (ydata_std_dev_KL_good)
    
    print ('Sigma KL bad')
    print (ydata_std_dev_KL_bad)
    
    print ('Sigma non KL good')
    print (ydata_std_dev_non_KL_good)
    
    print ('Sigma non KL bad')
    print (ydata_std_dev_non_KL_bad)
    
    
    print ()

    # Sig here is for Sigma and not signal
    # Also was wondering if a_KL_good, a_KL_bad, a_non_KL_good, a_non_KL_bad needs the weight?
    for sig_KL_good, sig_KL_bad, sig_non_KL_good, sig_non_KL_bad, a_KL_good_i, a_KL_bad_i, a_non_KL_good_i, a_non_KL_bad_i in zip(ydata_std_dev_KL_good, ydata_std_dev_KL_bad, ydata_std_dev_non_KL_good, ydata_std_dev_non_KL_bad, a_KL_good, a_KL_bad, a_non_KL_good, a_non_KL_bad):
        #print ('Starting from here...')
        #print (bd)
        #print (t)
        tot = a_KL_good_i+a_KL_bad_i+a_non_KL_good_i+a_non_KL_bad_i
        
        #ydata_std_dev.append(math.sqrt((sig_KL_good*a_KL_good_i/tot)**2 + (sig_KL_bad*a_KL_bad_i/tot)**2 + (sig_non_KL_good*a_non_KL_good_i/tot)**2 + (sig_non_KL_bad*a_non_KL_bad_i/tot)**2))
        ydata_std_dev.append(math.sqrt((sig_KL_good)**2 + (sig_KL_bad)**2 + (sig_non_KL_good)**2 + (sig_non_KL_bad)**2))
    print ('**************** Final standard deviation (sigma)..............****************')
    print(ydata_std_dev)
    print ('******************************************************************************') 


    return ydata_std_dev


def plotter (df_data, df_mc, df_ext, df_detsys, var, ax1, ax2, f, counter_run,  bdt_score='0.0', nbins=25, plotRange=None, sub_plotRange=None, align = 'mid', bnb_flat_scale = True, data_driven_scale = False, dataplot=True, plotbar = False, ylim=None, leg_loc_num=0, ncol=1, xlabel='', sig_type_str='', flag_1bin=False, flag_logy=False):

    #in_fv_query = "10<=reco_nu_vtx_x<=246 and -106<=reco_nu_vtx_y<=106 and 10<=reco_nu_vtx_z<=1026"
    #out_fv_query = "((reco_nu_vtx_x<10 or reco_nu_vtx_x>246) or (reco_nu_vtx_y<-106 or reco_nu_vtx_y>106) or (reco_nu_vtx_z<10 or reco_nu_vtx_z>1026))"
    """
    global bkg_cv
    global bkg_up
    global bkg_down
    
    
    global data_cv
    global data_up
    global data_down
    """
    bkg_cv = 'Bkg_CV'+str(counter_run)
    bkg_up = 'Bkg_Fluct_Up'+str(counter_run)
    bkg_down = 'Bkg_Fluct_Down'+str(counter_run)
    bkg_ext_cv = 'Bkg_EXT_CV'+str(counter_run)

    data_cv = 'Data_CV'+str(counter_run)
    data_up = 'Data_Fluct_Up'+str(counter_run)
    data_down = 'Data_Fluct_Down'+str(counter_run)

    
    df_mc_org = df_mc.copy()
    df_mc = df_mc_org.reset_index(drop=True)
        
    #print ('Fixing universes weights')
    #df_mc = fix_universes_weight(df_mc.copy())
    
    print ('Calculating FLUX uncertainties')
    ydata_std_dev_ppfx = get_ydata_std_dev_ppfx (df_mc, var, nbins, plotRange)
    ydata_std_dev_ppfx = np.array(ydata_std_dev_ppfx)      


    # For xsec we have NC Coherent Pi0 events that needs to be fixed and not the ppfx.
    print ('Calculating XSEC uncertainties')
    ydata_std_dev_xsec = get_ydata_std_dev_xsec (df_mc, var, nbins, plotRange)
    ydata_std_dev_xsec = np.array(ydata_std_dev_xsec)      

    print ('Calculating REINT uncertainties')
    ydata_std_dev_reint = get_ydata_std_dev_reint (df_mc, var, nbins, plotRange)

    print ('Calculating DETECTOR systematics')
    frac_ydata_std_dev_detsys = get_frac_ydata_std_dev_detsys (df_detsys, var, nbins, plotRange, bdt_score_str = bdt_score)
    
    #if frac_ydata_std_dev_detsys == 0:
    #    frac_ydata_std_dev_detsys = get_frac_ydata_std_dev_detsys (df_detsys.copy(), var, nbins, plotRange, bdt_score_str = bdt_score, bool_1bin=True)
    


    final_bkg_ovl_df = df_mc.query('bkg_category=="o"')
    final_bkg_dirt_df = df_mc.query('bkg_category=="d"')
    final_bkg_ext_df = df_ext
    
    
    numu_CC_npi0_ls = '((nu_pdg_ls==14 or nu_pdg_ls==-14) and ccnc_ls==0 and npi0_ls>=1)'
    numu_CC_0pi0 = '((nu_pdg_ls==14 or nu_pdg_ls==-14) and ccnc_ls==0 and npi0_ls==0)'
    numu_NC_npi0_ls = '((nu_pdg_ls==14 or nu_pdg_ls==-14) and ccnc_ls==1 and npi0_ls>=1)'
    numu_NC_0pi0 = '((nu_pdg_ls==14 or nu_pdg_ls==-14) and ccnc_ls==1 and npi0_ls==0)'
    nue_cc_ls = '(nu_pdg_ls==12 or nu_pdg_ls==-12) and ccnc_ls==0'
    nue_nc_ls = '(nu_pdg_ls==12 or nu_pdg_ls==-12) and ccnc_ls==1'
    #other = 'nu_pdg_ls != nu_pdg_ls'
    
    
    stacked_hist = [final_bkg_ovl_df.query(numu_CC_npi0_ls)[var],
                    final_bkg_ovl_df.query(numu_CC_0pi0)[var],
                    final_bkg_ovl_df.query(numu_NC_npi0_ls)[var],
                    final_bkg_ovl_df.query(numu_NC_0pi0)[var],
                    final_bkg_ovl_df.query(nue_cc_ls)[var],
                    final_bkg_ovl_df.query(nue_nc_ls)[var],
                    final_bkg_dirt_df[var],
                    final_bkg_ext_df[var]]    



    #COLOR=['tomato','steelblue','lightskyblue','darkorange','bisque','green','palegreen','olive','y','dimgray','darkgrey']
    COLOR = ['steelblue','palegreen','green','y','olive','salmon','darkorange','c']

    #bnb_scale_ls, KDAR_scale, scale_factor


    
    pot_scale = 1
    
    WEIGHTS = [      
            final_bkg_ovl_df.query(numu_CC_npi0_ls)['weightSplineTimesTune_ls']*final_bkg_ovl_df.query(numu_CC_npi0_ls)['scale_factor_ls']*final_bkg_ovl_df.query(numu_CC_npi0_ls)['bnb_scale_ls']*pot_scale*final_bkg_ovl_df.query(numu_CC_npi0_ls)['ppfx_cv_ls'],
            final_bkg_ovl_df.query(numu_CC_0pi0)['weightSplineTimesTune_ls']*final_bkg_ovl_df.query(numu_CC_0pi0)['scale_factor_ls']*pot_scale*final_bkg_ovl_df.query(numu_CC_0pi0)['ppfx_cv_ls'],
            final_bkg_ovl_df.query(numu_NC_npi0_ls)['weightSplineTimesTune_ls']*final_bkg_ovl_df.query(numu_NC_npi0_ls)['scale_factor_ls']*final_bkg_ovl_df.query(numu_NC_npi0_ls)['bnb_scale_ls']*pot_scale*final_bkg_ovl_df.query(numu_NC_npi0_ls)['ppfx_cv_ls'],
            final_bkg_ovl_df.query(numu_NC_0pi0)['weightSplineTimesTune_ls']*final_bkg_ovl_df.query(numu_NC_0pi0)['scale_factor_ls']*pot_scale*final_bkg_ovl_df.query(numu_NC_0pi0)['ppfx_cv_ls'],
            final_bkg_ovl_df.query(nue_cc_ls)['weightSplineTimesTune_ls']*final_bkg_ovl_df.query(nue_cc_ls)['scale_factor_ls']*pot_scale*final_bkg_ovl_df.query(nue_cc_ls)['ppfx_cv_ls'],
            final_bkg_ovl_df.query(nue_nc_ls)['weightSplineTimesTune_ls']*final_bkg_ovl_df.query(nue_nc_ls)['scale_factor_ls']*pot_scale*final_bkg_ovl_df.query(nue_nc_ls)['ppfx_cv_ls'],
            final_bkg_dirt_df['weightSplineTimesTune_ls']*final_bkg_dirt_df['scale_factor_ls']*pot_scale*final_bkg_dirt_df['ppfx_cv_ls'],
            final_bkg_ext_df['scale_factor_ls']*pot_scale]




    if plotRange is None:
        counts_hist ,bins,_ = ax1.hist(stacked_hist, stacked=True, weights=WEIGHTS, histtype="stepfilled", bins=nbins,align = align, lw=1,color=COLOR)#, density=True)
    else:
        counts_hist ,bins,_ = ax1.hist(stacked_hist, stacked=True, weights=WEIGHTS, histtype="stepfilled", bins=nbins,align = align, lw=1,range = plotRange,color=COLOR)#, density=True)
        
    
    # The following will be the pure stacked histogram with no weights applid and not POT scaling applied.
    # This is just to calculate the stat uncertainty.
    
    
    #display(numpy.concatenate(stacked_hist))
    
    counts_hist_unwt, _ = numpy.histogram(numpy.concatenate(stacked_hist), bins=nbins, range = plotRange)#, density=True)
    
    #counts_hist_unwt ,_,_ = ax1.hist(stacked_hist, stacked=True, histtype="stepfilled", bins=nbins,align = align,label='_nolegend_', lw=0)#, density=True)

    
    """
    counts_hist_unwt_mc, _ = numpy.histogram(df_mc[var], bins=nbins, range = plotRange)#, density=True)

    frac_sigma_ppfx = 
    
    """
    
    
    """
    print ('Check the following')
    print ()
    print ('With POT scaling we have')
    print (counts_hist[-1])
    
    print ('WITHOUT POT scaling pure sample we have...')
    print (counts_hist_unwt)
    """
    
    print ('Calculating statistical uncertainties')
    counts_hist_unwt_sigma = numpy.sqrt(counts_hist_unwt)
    #print ('WITHOUT POT scaling pure sample square-root is...')
    #print (counts_hist_unwt_sigma)
    
    frac_counts_hist_unwt_sigma = counts_hist_unwt_sigma/counts_hist_unwt
    #print ('Fractional uncertainty is...')
    #print (frac_counts_hist_unwt_sigma)
    stat_sigma = numpy.multiply (counts_hist[-1], frac_counts_hist_unwt_sigma)
    
    
    
    
    ydata_std_dev_detsys = numpy.multiply (counts_hist[-3], frac_ydata_std_dev_detsys)
    
    
    #print ('After multiplying the fractional uncertainty to POT scaled histogram...')
    #print (stat_sigma)
   
    
    # say you have 900 events before POT and other weights applied. stat_uncertainty = +/- 30
    # 400 after applying POT scaling and the other weights. stat_uncertainty won't be +/- 20
    # But +/- 400*(30/900) = 13.33333 is what we would have for 400 and not +/- 20
    # That should be the shaded region
    
    
    
    # ADD IN QUADRATURE ALL THE UNCERTINATIES. STATISTICAL + FLUX AND XSEC (STILL HAVEN'T INCLUDED DETECTOR SYS)
    
    print ('==================================')
    print ('ALL UNCERTAINTIES')
    print ('==================================')
    print ('==================================')
    print ('Statistical')
    print (stat_sigma)
    print ('==================================')
    print ('FLUX')
    print (ydata_std_dev_ppfx)
    print ('==================================')
    print ('XSEC')
    print (ydata_std_dev_xsec)
    print ('==================================')
    print ('REINT')
    print (ydata_std_dev_reint)
    print ('==================================')
    print ('DETSYS fractional')
    print (frac_ydata_std_dev_detsys)
    print ('DETSYS')
    print (ydata_std_dev_detsys)
    print ('==================================')
    print ('==================================')
    
    #$#
    print ('==================================')
    print ('Adding in quadrature these uncertainties gives')
    print ('==================================')
    
    """
    stat_sigma = numpy.nan_to_num(stat_sigma, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    ydata_std_dev_ppfx = numpy.nan_to_num(ydata_std_dev_ppfx, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    ydata_std_dev_xsec = numpy.nan_to_num(ydata_std_dev_xsec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    ydata_std_dev_reint = numpy.nan_to_num(ydata_std_dev_reint, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    ydata_std_dev_detsys = numpy.nan_to_num(ydata_std_dev_detsys, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    """

    stat_sigma = numpy.array(stat_sigma)
    ydata_std_dev_ppfx = numpy.array(ydata_std_dev_ppfx)
    ydata_std_dev_xsec = numpy.array(ydata_std_dev_xsec)
    ydata_std_dev_reint = numpy.array(ydata_std_dev_reint)
    ydata_std_dev_detsys = numpy.array(ydata_std_dev_detsys)
    
    
    stat_sigma_2 = np.multiply(stat_sigma, stat_sigma)
    ydata_std_dev_ppfx_2 = np.multiply(ydata_std_dev_ppfx, ydata_std_dev_ppfx)
    ydata_std_dev_xsec_2 = np.multiply(ydata_std_dev_xsec, ydata_std_dev_xsec)
    ydata_std_dev_reint_2 = np.multiply(ydata_std_dev_reint, ydata_std_dev_reint)
    ydata_std_dev_detsys_2 = np.multiply(ydata_std_dev_detsys, ydata_std_dev_detsys)
    
    total_stat_sigma = np.array([stat_sigma_2, ydata_std_dev_ppfx_2, ydata_std_dev_xsec_2, ydata_std_dev_reint_2, ydata_std_dev_detsys_2])
    total_stat_sigma = total_stat_sigma.sum(axis=0)
    final_sigma = np.sqrt(total_stat_sigma)
    
    for idx, (h_i, h_s_i) in enumerate (zip(counts_hist[-3], final_sigma)):
        if h_s_i == np.inf or h_s_i == -np.inf or h_s_i > 1.3*h_i:
            final_sigma[idx] = h_i
            
        if math.isnan(h_s_i):
            final_sigma[idx] = 0

    
    print ('Background bar height')
    print ('==================================')
    print (counts_hist[-1])
    print ('==================================')
    
    
    print ('Background bar uncertainty')
    print ('==================================')
    
    #final_sigma = math.sqrt((stat_sigma)**2 + (ydata_std_dev_ppfx)**2 + (ydata_std_dev_xsec)**2 + (ydata_std_dev_reint)**2)
    print (final_sigma)
    
    print ('==================================')
    
    # draw gray area error
    # for the MC stacked histogram the gray area error represents the MC stat uncertainty
    

    vals, bine = numpy.histogram(df_data[var],bins=bins)
    binc = 0.5*(bine[1:]+bine[:-1])
    vals = vals.astype(float)
    errs = numpy.sqrt(vals)
    widths = (binc - bine[:-1])
    
    
    print ('Data bar height')
    print (vals)
    
    print ('Data bar uncertainty')
    print (errs)
    
    if flag_1bin == True:
        print ('NOT PLOTTING BAR')
        ax1.bar(x=binc, height=2*final_sigma, bottom=counts_hist[-1]-final_sigma, hatch='///', width=plotRange[-1] - plotRange[0],label='uncertainties', align='center', color='white', linewidth=1, alpha=0.4, zorder=2)
    else:
        #ax1.fill_between(binc, counts_hist[-1]-counts_hist_sigma, counts_hist[-1]+counts_hist_sigma, step='mid', color='gray', hatch='///', alpha=0.3, zorder=2)
        ax1.bar(x=binc, height=2*final_sigma, bottom=counts_hist[-1]-final_sigma, hatch='///', width=binc[1]-binc[0],label='uncertainties', align='center', color='white', linewidth=1, alpha=0.4, zorder=2)
 
        
    
    ##$$  
    w_plus_bkg = numpy.concatenate(WEIGHTS)
    w_minus_bkg = numpy.concatenate(WEIGHTS)
    
    inds_bkg = np.digitize(numpy.concatenate(stacked_hist), bins=bine)
        
    for n in range((numpy.concatenate(stacked_hist)).size):
        # print ('How many n are there?...: ' + str(n))
        # Get all the a[n] with its w[n] and find the bin each a[n], w[n] belong to using if condition
        for i in range (binc.size):
            # Find the bin i that has ydata_std_dev[i]

            # An event can be either in bin 1 or in bin 2 .... or in bin 25.
            if inds_bkg[n]-1 == i:

                # Following is for fractional uncertainties.
                # This has to go in the .root file that will be later used in the Collie.

                #print (counts_hist)
                
                wfactor_plus_bkg = final_sigma[i]/counts_hist[-1][i]
                w_plus_bkg[n] = w_plus_bkg[n]*wfactor_plus_bkg/counts_hist[-1][i]

                wfactor_minus_bkg = final_sigma[i]/counts_hist[-1][i]
                w_minus_bkg[n] = w_minus_bkg[n]*wfactor_minus_bkg/counts_hist[-1][i]
    
    
    
    
    
    
    if flag_1bin == True:
        
        #a,b = numpy.histogram(df_data[var],weights = weights_sig, bins=nbins, range = plotRange)
        
        # counts_hist_1bin,_  = numpy.histogram(numpy.concatenate(stacked_hist), weights=numpy.concatenate(WEIGHTS), bins=1, range = plotRange)
        
        
        print ('YES GOING IN THE flag_1bin ............................................. ')
        new_array = np.linspace(plotRange[0], plotRange[1], 25)
        print (new_array)
        print ('...............................................................................................')
        weights_1bin = np.ones_like(new_array)*counts_hist[-1]/ 25.0
        f[bkg_cv] = numpy.histogram(new_array, bins=25, weights = weights_1bin, range=plotRange)
        
        
        
        
        weights_up_down = np.ones_like(new_array)*(final_sigma/counts_hist[-1])
        f[bkg_up] = numpy.histogram(new_array, bins=25, weights = weights_up_down, range=plotRange)
        f[bkg_down] = numpy.histogram(new_array, bins=25, weights = weights_up_down, range=plotRange)

    else:
        f[bkg_cv] = numpy.histogram (numpy.concatenate(stacked_hist[:-1]), weights=numpy.concatenate(WEIGHTS[:-1]), bins=nbins, range = plotRange)
        f[bkg_up] = numpy.histogram(numpy.concatenate(stacked_hist), bins=bins, weights = w_plus_bkg)
        f[bkg_down] = numpy.histogram(numpy.concatenate(stacked_hist), bins=bins, weights = w_minus_bkg)
        f[bkg_ext_cv] = numpy.histogram (numpy.concatenate(stacked_hist[-1:]), weights=numpy.concatenate(WEIGHTS[-1:]), bins=nbins, range = plotRange)
        #f[bkg_ext_up] = numpy.histogram(numpy.concatenate(stacked_hist[-1:]), bins=bins, weights = w_plus_bkg)
        #f[bkg_ext_down] = numpy.histogram(numpy.concatenate(stacked_hist[-1:]), bins=bins, weights = w_minus_bkg)


    w_plus = np.ones(len(df_data[var]))
    w_minus = np.ones(len(df_data[var]))

    
    # a[n] is the height of the bin n for CV. So all events are extracted back again with their weights
    # For each event there's a weight and that weight needs to be changed depending on where the event is (which bin it is in)
    # in histogram. If an event lies in bin 2 with and if bin 2 has some uncertainty of 2.0 then it's weight 
    # will be changed by a factor so that it has a specific height.
    inds = np.digitize(np.array(df_data[var].values), bins=bine)


    for n in range(np.array(df_data[var].values).size):
        # print ('How many n are there?...: ' + str(n))
        # Get all the a[n] with its w[n] and find the bin each a[n], w[n] belong to using if condition
        for i in range (binc.size):
            # Find the bin i that has ydata_std_dev[i]

            # An event can be either in bin 1 or in bin 2 .... or in bin 25.
            if inds[n]-1 == i:

                # Following is for fractional uncertainties.
                # This has to go in the .root file that will be later used in the Collie.

                wfactor_plus = (errs[i])/vals[i]
                w_plus[n] = w_plus[n]*wfactor_plus/vals[i]

                wfactor_minus = (errs[i])/vals[i]
                w_minus[n] = w_minus[n]*wfactor_minus/vals[i]
    
    
    if flag_1bin == True:
        f[data_cv] = numpy.histogram(df_data[var], bins=25)
        #f[data_up] = numpy.histogram(df_data[var], bins=bins, weights = w_minus)
        #f[data_down] = numpy.histogram(df_data[var], bins=bins, weights = w_minus)
    else:
        f[data_cv] = numpy.histogram(df_data[var], bins=bins)
        #f[data_up] = numpy.histogram(df_data[var], bins=bins, weights = w_minus)
        #f[data_down] = numpy.histogram(df_data[var], bins=bins, weights = w_minus)
        
        

    count_data = numpy.sum(vals).round(2)
    
    Data_str = 'Data (' + str(count_data)+')'
    
    # Just to have the marker at the top of a histogram
    if dataplot == True:
    #ax1.errorbar(binc,vals,yerr=errs,fmt='o',color='k',markersize=3,label='Data – '+str(df_data[var].count()),elinewidth=1)
        ax1.errorbar(binc, vals, yerr=errs, fmt='o', color='k', markersize=3, elinewidth=1.25, label=Data_str)    
        
        
    numu_CCnpi0_ls_count = numpy.sum(counts_hist[0])
    numu_CC0pi0_count = numpy.sum(counts_hist[1]-counts_hist[0])
    numu_NCnpi0_ls_count = numpy.sum(counts_hist[2]-counts_hist[1])
    numu_NC0pi0_count = numpy.sum(counts_hist[3]-counts_hist[2])
    nue_cc_ls_count = numpy.sum(counts_hist[4]-counts_hist[3])
    nue_nc_ls_count = numpy.sum(counts_hist[5]-counts_hist[4])
    #other_count = numpy.sum(counts_hist[5]-counts_hist[4])
    Ext_count = numpy.sum(counts_hist[6]-counts_hist[5])
    Dirt_count = numpy.sum(counts_hist[7]-counts_hist[6])


    #nue_NC_0pi0_count = numpy.sum(counts_hist[7]-counts_hist[6]).round(2)
    #Ext_count = numpy.sum(counts_hist[8]-counts_hist[7]).round(2)
    #print('Total count for Overlay...: '+ str(numpy.sum(counts_hist[6])))


    count_all = nue_cc_ls_count + nue_nc_ls_count + numu_CCnpi0_ls_count + numu_CC0pi0_count + numu_NCnpi0_ls_count + numu_NC0pi0_count + Ext_count + Dirt_count
    #

    numu_CCnpi0_ls_str = '$\\nu_\mu$ CC N$\pi^{0}$ (' + str(numu_CCnpi0_ls_count.round(2)) + ')'#+ ' ' + str (infv.query(numu_CC_npi0_ls)[var].count()*0.07)
    numu_CC0pi0_str = '$\\nu_\mu$ CC (' + str(numu_CC0pi0_count.round(2))+ ')'#+ ' ' + str (infv.query(numu_CC_0pi0)[var].count()*0.07)
    numu_NCnpi0_ls_str = '$\\nu_\mu$ NC N$\pi^{0}$ (' + str(numu_NCnpi0_ls_count.round(2))+ ')'#+ ' ' + str (infv.query(numu_NC_npi0_ls)[var].count()*0.07)
    numu_NC0pi0_str = '$\\nu_\mu$ NC (' + str(numu_NC0pi0_count.round(2))+ ')'#+ ' ' + str (infv.query(numu_NC_0pi0)[var].count()*0.07)

    nue_cc_ls_str = '$\\nu_e$ CC (' + str(nue_cc_ls_count.round(2))+ ')'#+ ' ' + str (infv.query(nue_CC_npi0_ls)[var].count()*0.07)
    nue_nc_ls_str = '$\\nu_e$ NC (' + str(nue_nc_ls_count.round(2))+ ')'#+ ' ' + str (infv.query(nue_CC_npi0_ls)[var].count()*0.07)
    #other_str = 'Other (' +  str(other_count.round(2))+ ')'
    
    Ext_str = 'EXT (' + str(Ext_count.round(2))+ ')'#+ ' ' + str (df_ext[var].count()*0.584)
    Dirt_str = 'DIRT (' + str(Dirt_count.round(2))+ ')'

    total_bkg = 'Total Bkg (' + str(count_all.round(5))+ ')'
    
    
    
    finite_counts_hist = numpy.nan_to_num(counts_hist, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    finite_final_sigma = numpy.nan_to_num(final_sigma, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    #print ('I think lengths are not matching')
    #print (finite_counts_hist[-1])
    #print (finite_final_sigma)
    #print (ylim)
    """
    if np.max(finite_final_sigma+finite_counts_hist[-1]) > ylim:
        ax1.set_ylim(top=(np.max(finite_final_sigma+finite_counts_hist[-1]))*1.6, bottom=0)
    else:
        ax1.set_ylim(top=(ylim)*1.6, bottom=0)

    """
        
    finite_counts_hist = numpy.nan_to_num(counts_hist, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    finite_final_sigma = numpy.nan_to_num(final_sigma, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    
    finite_data_hist = numpy.nan_to_num(vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    finite_data_sigma = numpy.nan_to_num(errs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    
    ylim_sig = ylim
    ylim_bkg = np.max(finite_final_sigma+finite_counts_hist[-1])
    ylim_data = np.max(finite_data_sigma+finite_data_hist)
    
    #print ('I think lengths are not matching')
    #print (finite_counts_hist[-1])
    #print (finite_final_sigma)
    #print (ylim)

        
    if (ylim_sig >= ylim_bkg) and (ylim_sig >= ylim_data):
        ylim = ylim_sig
    elif (ylim_bkg >= ylim_sig) and (ylim_bkg >= ylim_data):
        ylim = ylim_bkg
    else:
        ylim = ylim_data
        
    if not flag_logy:
        ax1.set_ylim(top=(ylim)*1.6, bottom=0)
    else:
        ax1.set_ylim(top=10**(math.ceil(-3+2*math.log10(ylim)+3)), bottom=1e-3)

    
    leg_loc = {0: "best",
           1: "upper right",
           2: "upper left",
           3: "lower left",
           4: "lower right",
           5: "right",
           6: "center left",
           7: "center right",
           8: "lower center",
           9: "upper center",
           10: "center"}

    
    if dataplot==True:
        """
        # annotate_shr = xlabel
        # plt.rc('text', usetex=True)
        
        #string_inside_bracket = re.search('\(([^)]+)', xlabel).group(1)
        string_inside_bracket = re.findall('\(([^)]+)', xlabel)#.group(1)
        print ('string_inside_bracket')
        print (string_inside_bracket)
        if len(string_inside_bracket) > 1 :
            string_inside_bracket = string_inside_bracket[-1] # Ignores MeV if it exists
        
        if len(string_inside_bracket) == 1 and string_inside_bracket[0] == 'MeV':
            string_inside_bracket = None
        
        if string_inside_bracket:
            annotate_shr = string_inside_bracket.replace(' ','-')
            mass_str = xlabel.split(",")[-1]
            if sig_type_str == None:
                print ('RIGHT PLACE..................................................')
                right_title_leg_str = r'$\mathbf{' + annotate_shr+'\ events' + '}$'
            else:
                right_title_leg_str = r'$\mathbf{' + annotate_shr+'\ events' + '}$, ' + mass_str
            
        else:
            right_title_leg_str = ''

        
        """
        
        #string_inside_bracket = re.search('\(([^)]+)', xlabel).group(1)
        string_inside_bracket = re.findall('\(([^)]+)', xlabel)#.group(1)
        print ('xlabel','"',xlabel,'"')
        print ('string_inside_bracket')
        print (string_inside_bracket)
        
        if len(string_inside_bracket) > 1 :
            string_inside_bracket = string_inside_bracket[-1] # Ignores MeV if it exists
            
        elif len(string_inside_bracket) == 1 and string_inside_bracket[0] == 'MeV': # If MeV then don't label in legend
            string_inside_bracket = None
            
        else:
            string_inside_bracket = string_inside_bracket[0]
            
            
        print ('string_inside_bracket2')
        print (string_inside_bracket)
        
        if string_inside_bracket:
            if sig_type_str == None:
                if 'one' in string_inside_bracket or 'two' in string_inside_bracket:
                    annotate_shr = string_inside_bracket.replace(' ','-')
                    print ('RIGHT PLACE..................................................')
                    right_title_leg_str = r'$\mathbf{' + annotate_shr+'\ events' + '}$'
                else:
                    print ('string_inside_bracket3')
                    print (string_inside_bracket) 
                    annotate_shr = string_inside_bracket#.replace(' ', ' ')
                    print ('PERFECT PLACE..................................................')
                    right_title_leg_str = r'$\mathbf{' + annotate_shr + '}$'
            else:
                print ('Why here?')
                annotate_shr = string_inside_bracket.replace(' ','-')
                mass_str = xlabel.split(",")[-1]
                right_title_leg_str = r'$\mathbf{' + annotate_shr+'\ events' + '}$, ' + mass_str
                
            left_title_leg_str =  r'$\mathbf{\sum}$Data / $\mathbf{\sum}$MC = ' + str(round(count_data/count_all,3)) +', '
            title_leg_str = left_title_leg_str + ' ' + right_title_leg_str
            print ('string_inside_bracket4')
            print (title_leg_str) 
                
                
        else:
            mass_str = xlabel.split(",")[-1]
            right_title_leg_str = mass_str
            left_title_leg_str =  r'$\mathbf{\sum}$Data / $\mathbf{\sum}$MC = ' + str(round(count_data/count_all,3))
            
            print ('IS IT COMING HERE?')
            title_leg_str = left_title_leg_str + ', ' + right_title_leg_str
        
        #left_title_leg_str =  r'$\mathbf{\sum}$Data / $\mathbf{\sum}$MC = ' + str(round(count_data/count_all,3))+', '

        #title_leg_str = left_title_leg_str + ' ' + right_title_leg_str
        
        
        
        
        """
        left_alignment = "Left Text"
        center_alignment = "Centered Text"
        right_alignment = "Right Text"
        
        title_leg_str = '{:>30} {:>30}'.format("a", "b")
        
        
        title_leg_str = '{0: <16} StackOverflow!'.format('Hi')
        """
        
        
        # leg_str0_final = '{:>30} {:>30}'.format(leg_str0, title_leg_str)
        
        #'{:>12}  {:>12}  {:>12}'.format(word[0], word[1], word[2])
        

        # print(f"{left_alignment : <20}{center_alignment : ^15}{right_alignment : >20}")
        # leg_str0 = f"{'a':<20}  {'b':>20}"
        
        #leg_str0 = '{:<30} {:>30}'.format("a", "b")
        #leg_str0 = f"{'a' : <20}{'b' : ^15}{'c' : >20}"
        
        

        if sig_type_str == None:
            LABELS = [Data_str, 'EXT Stat.'+'\n' + 'MC Stat.+Syst. Uncertainty', numu_CCnpi0_ls_str, numu_CC0pi0_str, numu_NCnpi0_ls_str, numu_NC0pi0_str, nue_cc_ls_str, nue_nc_ls_str ,Ext_str, Dirt_str]
        else:
            LABELS = [Data_str, 'EXT Stat.'+'\n' + 'MC Stat.+Syst. Uncertainty', numu_CCnpi0_ls_str, numu_CC0pi0_str, numu_NCnpi0_ls_str, numu_NC0pi0_str, nue_cc_ls_str, nue_nc_ls_str ,Ext_str, Dirt_str, 'Signal '+sig_type_str]

        # 'MC Stat.+Syst. Uncertainty'
        
        LABELS = LABELS[::-1]
        #LABELS.append(Data_str)
        #LABELS.append('SIGNAL')
    
        
        plt.rcParams['legend.loc'] = leg_loc[leg_loc_num]
                
        #leg = ax1.legend(LABELS, title=title_leg_str, fontsize=13, title_fontsize=14, ncol=ncol, columnspacing=1.7, frameon=False)
        leg = ax1.legend(LABELS, title=title_leg_str, fontsize=14, title_fontsize=15, ncol=ncol, columnspacing=1.7, frameon=False)
        leg._legend_box.align = "left"
        #leg.legendHandles[-1].set_color('k')

        
        
        #leg2 = ax1.legend(title=title_leg_str, fontsize=15, title_fontsize=16)
        #leg2._legend_box.align = "right"
        
        # offset = matplotlib.text.OffsetFrom(leg, (1.0, 0.0))
        # Create annotation. Top right corner located -20 pixels below the offset point 
        # (lower right corner of legend).
        
        

        
        # ax1.annotate(annotate_shr+' events, '+ mass_str, xy=(0,0),size=16,
        #             xycoords='figure fraction', xytext=(-320, 91), textcoords=offset)
                    

        #leg.set_title(leg_str0,prop={'size':14})
        #handles, labels = ax1.get_legend_handles_labels()
        #handles.append(mpatches.Patch(color='none', label=leg_str0))
        #ax1.legend(handles=handles)

        #leg_str1 = r"$\chi^{2}$"+ "/dof("+ str(nbins) +"): " + str(round(scipy.stats.chisquare(vals,counts_hist[-1])[0]/nbins,3))
        #leg_str2 = "p-value: " + str (scipy.stats.chisquare(vals,counts_hist[-1])[1].round(2))

        
        # vals data and counts_hist[-1] is pred. Therefore, ( data - pred )/ pred is what we want
        # errs = numpy.sqrt(vals) and we divide the errors by pred.
        
        
        #ax2.errorbar(binc,     # this is what makes it comparable
        #    (vals-counts_hist[-1]) / counts_hist[-1], # maybe check for div-by-zero!
        #    fmt='o',color='k',alpha=1.0, yerr=errs/counts_hist[-1], markersize=3,elinewidth=1)

        ax2.errorbar(binc,     # this is what makes it comparable
            vals / counts_hist[-1], # maybe check for div-by-zero!
            fmt='o',color='k',alpha=1.0, yerr=errs/counts_hist[-1], markersize=3,elinewidth=1)

        
        
        #ax2.plot([], [], ' ', label=leg_str2)
        #ax2.legend(markerscale=1,handletextpad=-1000,handlelength=1000.2,fontsize=14)

        ax2.axhline(y=1.0, linewidth = 1, color='black')
        ax2.set_ylim(sub_plotRange)
        
        
        
        
        """
        counts_hist_unwt, _ = numpy.histogram(numpy.concatenate(stacked_hist), bins=nbins)#, density=True)
        print ('Imp to look below')
        print (counts_hist_unwt)
        print (counts_hist[-1])
        """
        
        #sub_bar_plot = ax2.bar(binc, numpy.sqrt(counts_hist_unwt)/counts_hist_unwt, width = binc[1]-binc[0], fill=True, color = 'grey', alpha=0.7)
        #sub_bar_plot = ax2.bar(binc, -numpy.sqrt(counts_hist_unwt)/counts_hist_unwt, width = binc[1]-binc[0], fill=True, color = 'grey', alpha=0.7)
        
        if flag_1bin == False:

            sub_bar_plot = ax2.bar(binc, ydata_std_dev_detsys/counts_hist[-1], bottom=1, width = binc[1]-binc[0], fill=False, edgecolor = 'navy', label='Detector')#, alpha=0.4)
            sub_bar_plot = ax2.bar(binc, -ydata_std_dev_detsys/counts_hist[-1], bottom=1, width = binc[1]-binc[0], fill=False, edgecolor = 'navy')#, alpha=0.4)

            sub_bar_plot = ax2.bar(binc, ydata_std_dev_reint/counts_hist[-1], bottom=1, width = binc[1]-binc[0], fill=False, edgecolor = 'darkgreen', label='Re-interaction')#, alpha=0.4)
            sub_bar_plot = ax2.bar(binc, -ydata_std_dev_reint/counts_hist[-1], bottom=1, width = binc[1]-binc[0], fill=False, edgecolor = 'darkgreen')#, alpha=0.4)


            sub_bar_plot = ax2.bar(binc, ydata_std_dev_xsec/counts_hist[-1], bottom=1, width = binc[1]-binc[0], fill=False, edgecolor = 'darkred', label='Cross-section')#, alpha=0.4)
            sub_bar_plot = ax2.bar(binc, -ydata_std_dev_xsec/counts_hist[-1], bottom=1, width = binc[1]-binc[0], fill=False, edgecolor = 'darkred')#, alpha=0.4)


            sub_bar_plot = ax2.bar(binc, ydata_std_dev_ppfx/counts_hist[-1], bottom=1, width = binc[1]-binc[0], fill=False, edgecolor = 'darkorange', label='Flux')#, alpha=0.4)
            sub_bar_plot = ax2.bar(binc, -ydata_std_dev_ppfx/counts_hist[-1], bottom=1, width = binc[1]-binc[0], fill=False, edgecolor = 'darkorange')#, alpha=0.4)

            sub_bar_plot = ax2.bar(binc, stat_sigma/counts_hist[-1], bottom=1, width = binc[1]-binc[0], fill=False, edgecolor = 'c', label='Statistical')#, alpha=0.4)
            sub_bar_plot = ax2.bar(binc, -stat_sigma/counts_hist[-1], bottom=1, width = binc[1]-binc[0], fill=False, edgecolor = 'c')#, alpha=0.4)


            sub_bar_plot = ax2.bar(binc, final_sigma/counts_hist[-1], bottom=1, width = binc[1]-binc[0], fill=True, color = 'grey',edgecolor = 'grey', alpha=0.18, label='Total')#, alpha=0.4)
            sub_bar_plot = ax2.bar(binc, -final_sigma/counts_hist[-1], bottom=1, width = binc[1]-binc[0], fill=True, color = 'grey',edgecolor = 'grey', alpha=0.18)#, alpha=0.4)


            """
            sub_bar_plot = ax2.bar(binc, final_sigma/counts_hist[-1], bottom=1, hatch='///', width=binc[1]-binc[0], label='Total', align='center', color='white', linewidth=1, alpha=0.4)
            sub_bar_plot = ax2.bar(binc, -final_sigma/counts_hist[-1], bottom=1, hatch='///', width=binc[1]-binc[0], label='Total', align='center', color='white', linewidth=1, alpha=0.4)

            """
            
        else:

            sub_bar_plot = ax2.bar(binc, ydata_std_dev_detsys/counts_hist[-1], bottom=1, width=plotRange[-1] - plotRange[0], align='center', fill=False, edgecolor = 'navy', label='Detector')#, alpha=0.4)
            sub_bar_plot = ax2.bar(binc, -ydata_std_dev_detsys/counts_hist[-1], bottom=1, width=plotRange[-1] - plotRange[0], align='center', fill=False, edgecolor = 'navy')#, alpha=0.4)

            sub_bar_plot = ax2.bar(binc, ydata_std_dev_reint/counts_hist[-1], bottom=1, width=plotRange[-1] - plotRange[0],align='center', fill=False, edgecolor = 'darkgreen', label='Re-interaction')#, alpha=0.4)
            sub_bar_plot = ax2.bar(binc, -ydata_std_dev_reint/counts_hist[-1], bottom=1, width=plotRange[-1] - plotRange[0],align='center', fill=False, edgecolor = 'darkgreen')#, alpha=0.4)


            sub_bar_plot = ax2.bar(binc, ydata_std_dev_xsec/counts_hist[-1], bottom=1, width=plotRange[-1] - plotRange[0],align='center', fill=False, edgecolor = 'darkred', label='Cross-section')#, alpha=0.4)
            sub_bar_plot = ax2.bar(binc, -ydata_std_dev_xsec/counts_hist[-1], bottom=1, width=plotRange[-1] - plotRange[0],align='center', fill=False, edgecolor = 'darkred')#, alpha=0.4)


            sub_bar_plot = ax2.bar(binc, ydata_std_dev_ppfx/counts_hist[-1], bottom=1, width=plotRange[-1] - plotRange[0],align='center', fill=False, edgecolor = 'darkorange', label='Flux')#, alpha=0.4)
            sub_bar_plot = ax2.bar(binc, -ydata_std_dev_ppfx/counts_hist[-1], bottom=1, width=plotRange[-1] - plotRange[0],align='center', fill=False, edgecolor = 'darkorange')#, alpha=0.4)

            sub_bar_plot = ax2.bar(binc, stat_sigma/counts_hist[-1], bottom=1,width=plotRange[-1] - plotRange[0],align='center', fill=False, edgecolor = 'c', label='Statistical')#, alpha=0.4)
            sub_bar_plot = ax2.bar(binc, -stat_sigma/counts_hist[-1], bottom=1, width=plotRange[-1] - plotRange[0],align='center', fill=False, edgecolor = 'c')#, alpha=0.4)


            sub_bar_plot = ax2.bar(binc, final_sigma/counts_hist[-1], bottom=1, width=plotRange[-1] - plotRange[0],align='center', fill=True, color = 'grey',edgecolor = 'grey', alpha=0.18, label='Total')#, alpha=0.4)
            sub_bar_plot = ax2.bar(binc, -final_sigma/counts_hist[-1], bottom=1, width=plotRange[-1] - plotRange[0],align='center', fill=True, color = 'grey',edgecolor = 'grey', alpha=0.18)#, alpha=0.4)


                

        custom_lines = [Line2D([0], [0], color='navy', lw=1),
                Line2D([0], [0], color='darkgreen', lw=1),
                Line2D([0], [0], color='darkred', lw=1),
                Line2D([0], [0], color='darkorange', lw=1),
                Line2D([0], [0], color='c', lw=1),
                Line2D([0], [0], color='grey', lw=5, alpha=0.3)]

        #ax2.legend(custom_lines, ['Detector', 'Re-interaction', 'Cross-section', 'Flux', 'MC + EXT Statistical', 'Total'], loc='upper center', bbox_to_anchor=(0.5, 1.40), ncol=3, fancybox=True, fontsize=13, borderpad=0.35)
        #ax2.legend(custom_lines, ['Detector', 'Re-interaction', 'Cross-section', 'Flux', 'MC + EXT Statistical', 'Total'], loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fancybox=False,frameon=False, fontsize=13, borderpad=0.35)
        ax2.legend(custom_lines, ['Detector', 'Re-interaction', 'Cross-section', 'Flux', 'MC + EXT Statistical', 'Total'], loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=3, fancybox=False,frameon=False, fontsize=14, borderpad=0.35)
        #ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3, fancybox=True, shadow=False)

    else:
        LABELS = [numu_CCnpi0_ls_str, numu_CC0pi0_str, numu_NCnpi0_ls_str, numu_NC0pi0_str, nue_cc_ls_str, nue_nc_ls_str ,Ext_str, Dirt_str]

        LABELS = LABELS[::-1]
        ax1.legend(LABELS)
        
        
    # return leg

    #plt.legend(LABELS, fontsize=legfontsize, ncol = 2)
    #plt.text(0.1, 0.9,'matplotlib', ha='center', va='center', transform=plt.transAxes)

def std_dev_SIGNAL(ydata):
    #print ('I hope we have the right ydata ')
    #print (ydata)
    
    # sum(uni-CV)^2/N
    # For KDIF ppfx, For KDAR from beam dump 30%, KDAR target use ppfx.
    ydata_std_dev = []
    numrtr_sqrd = []
    
    
#     if len(ydata) != 11:
#         print ('Length is NOT eleven ')
        
#     elif len(ydata) == 11:
#         print ('We will hardly ever see this. Length = 11 ')
    
    for i in range (0, len(ydata)-1):
        
        # numerator_subtracted = ydata[i] substracted from ppfx_cv (ydata[-1])
        numrtr_sub = ydata[i] - ydata[-1] # x - mu. This is entire array's individual value subtracted
        #print (numrtr_sub)
        numrtr_sqrd.append(numpy.square(numrtr_sub)) # square an entire array and append it into a new array

    # square root of [sum of (x_i - mu)^2 / N-1]
    # minus 1 in denominator because last value is ppfx_cv
    ydata_std_dev = numpy.sqrt(numpy.sum(numrtr_sqrd, axis=0)/(len(ydata)-1))
    
    
    return ydata_std_dev
    
def get_frac_ydata_std_dev_detsys_SIGNAL(df, var, bins, plotRange, bdt_score_str='0.0'):
    
    print ('SIGNAL BDT SCORE IN DETSYS IS...' + str (bdt_score_str))
    #display(df)
    weights_cv = df.query('pred_CV >='+bdt_score_str)['KDAR_scale_CV']
    weights_ly_atten = df.query('pred_ly_atten >='+bdt_score_str)['KDAR_scale_ly_atten']
    weights_ly_down = df.query('pred_ly_down >='+bdt_score_str)['KDAR_scale_ly_down']
    weights_ly_rayleigh = df.query('pred_ly_rayleigh >='+bdt_score_str)['KDAR_scale_ly_rayleigh']
    weights_recomb = df.query('pred_recomb >='+bdt_score_str)['KDAR_scale_recomb']
    weights_sce = df.query('pred_sce >='+bdt_score_str)['KDAR_scale_sce']    
    weights_wiremod_anglexz = df.query('pred_wiremod_anglexz >='+bdt_score_str)['KDAR_scale_wiremod_anglexz']
    weights_wiremod_angleyz = df.query('pred_wiremod_angleyz >='+bdt_score_str)['KDAR_scale_wiremod_angleyz']
    weights_wiremod_dEdx = df.query('pred_wiremod_dEdx >='+bdt_score_str)['KDAR_scale_wiremod_dEdx']
    weights_wiremod_x = df.query('pred_wiremod_x >='+bdt_score_str)['KDAR_scale_wiremod_x']
    weights_wiremod_yz = df.query('pred_wiremod_yz >='+bdt_score_str)['KDAR_scale_wiremod_yz']
    
    
    
    y_CV, bin_CV = numpy.histogram(df.query('pred_CV >='+bdt_score_str)[var+'_CV'], weights=weights_cv, bins=bins, range = plotRange)
    y_ly_atten,_= numpy.histogram(df.query('pred_ly_atten >='+bdt_score_str)[var+'_ly_atten'],weights=weights_ly_atten, bins=bins, range = plotRange)
    y_ly_down,_= numpy.histogram(df.query('pred_ly_down >='+bdt_score_str)[var+'_ly_down'],weights=weights_ly_down, bins=bins, range = plotRange)
    y_ly_rayleigh,_= numpy.histogram(df.query('pred_ly_rayleigh >='+bdt_score_str)[var+'_ly_rayleigh'],weights=weights_ly_rayleigh, bins=bins, range = plotRange)
    y_recomb,_= numpy.histogram(df.query('pred_recomb >='+bdt_score_str)[var+'_recomb'],weights=weights_recomb, bins=bins, range = plotRange)
    y_sce,_ = numpy.histogram(df.query('pred_sce >='+bdt_score_str)[var+'_sce'],weights=weights_sce, bins=bins, range = plotRange)
    y_wiremod_anglexz,_= numpy.histogram(df.query('pred_wiremod_anglexz >='+bdt_score_str)[var+'_wiremod_anglexz'],weights=weights_wiremod_anglexz, bins=bins, range = plotRange)
    y_wiremod_angleyz,_= numpy.histogram(df.query('pred_wiremod_angleyz >='+bdt_score_str)[var+'_wiremod_angleyz'],weights=weights_wiremod_angleyz, bins=bins, range = plotRange)
    y_wiremod_dEdx,_= numpy.histogram(df.query('pred_wiremod_dEdx >='+bdt_score_str)[var+'_wiremod_dEdx'],weights=weights_wiremod_dEdx, bins=bins, range = plotRange)
    y_wiremod_x,_= numpy.histogram(df.query('pred_wiremod_x >='+bdt_score_str)[var+'_wiremod_x'],weights=weights_wiremod_x, bins=bins, range = plotRange)
    y_wiremod_yz,_= numpy.histogram(df.query('pred_wiremod_yz >='+bdt_score_str)[var+'_wiremod_yz'],weights=weights_wiremod_yz, bins=bins, range = plotRange)


    #print ('CV in denominator for DETSYS (SIGNAL)')
    #print (y_CV)


    y_all_arr = np.array([y_ly_atten, y_ly_down, y_ly_rayleigh, y_recomb, y_sce, y_wiremod_anglexz, y_wiremod_angleyz, y_wiremod_dEdx, y_wiremod_x, y_wiremod_yz, y_CV])

    
    #print ('Displaying the big array for Detsys')
    #print (y_all_arr)
    
    ydata_std_dev = std_dev_SIGNAL(y_all_arr)
    ydata_std_dev = np.array(ydata_std_dev)
    
    
    frac_ydata_std_dev_detsys_SIGNAL = ydata_std_dev/y_CV
    
    #frac_ydata_std_dev_detsys_SIGNAL[frac_ydata_std_dev_detsys_SIGNAL == inf] = 1
    
    if (y_CV == 0).any():
        if type(bins) is int:
            if bins == 1:
                print ('Not Working!! Tried with one bin for Detsys Signal but not working and therefore returning 1')
                return 1.0
        else:
            print ('Trying with one bin for detsys Signal')
            frac_ydata_std_dev_detsys_SIGNAL = get_frac_ydata_std_dev_detsys_SIGNAL(df, var, bins=1, plotRange=plotRange, bdt_score_str=bdt_score_str)

        

        
        
    return frac_ydata_std_dev_detsys_SIGNAL

def std_dev (ydata):
    
    
    # print (ydata)
    # There'd be an array with columns 25 and rows 601.
    # print (ydata)
    
    # sum(uni-CV)^2/N
    # For KDIF ppfx, For KDAR from beam dump 30%, KDAR target use ppfx.
    ydata_std_dev = []
    numrtr_sqrd = []
    
    
    for i in range (0,len(ydata)-1):
        
        # numerator_subtracted = ydata[i] substracted from ppfx_cv (ydata[-1])
        numrtr_sub = ydata[i] - ydata[-1] # x - mu. This is entire array's individual value subtracted
        #print (numrtr_sub)
        numrtr_sqrd.append(numpy.square(numrtr_sub)) # square an entire array and append it into a new array

    # square root of [sum of (x_i - mu)^2 / N-1]
    # minus 1 in denominator because last value is ppfx_cv

    ydata_std_dev = numpy.sqrt(numpy.sum(numrtr_sqrd, axis=0)/(len(ydata)-1))
    
    # print ('Following should be 600 or 1000 always...'+str((len(ydata)-1)))
    
    return ydata_std_dev



def get_ydata_std_dev_KDAR (df_data, var, nbins, plotRange):
    # We have two types of KDAR. One from target and the other from beamdump. Then we have KDIF which is defined
    # at get_ydata_std_dev_KDIF
    
    print ('IS IT EVEN GOING HERE??? ')
    
    # display(df_data)
    
    ydata_std_dev = []

    df_data_KL = df_data.query('nu_parent_pdg_ls==130').copy()
    df_data_non_KL = df_data.query('nu_parent_pdg_ls!=130').copy()
    
    df_data_KL_target = df_data_KL.query('KDAR_scale < 5').copy()
    df_data_KL_beamdump = df_data_KL.query('KDAR_scale > 5').copy()
    
    df_data_non_KL_target = df_data_non_KL.query('KDAR_scale < 5').copy()
    df_data_non_KL_beamdump = df_data_non_KL.query('KDAR_scale > 5').copy()
    


    weights_KL_target = df_data_KL_target['scale_factor_ls']*df_data_KL_target['ppfx_cv_ls']
    weights_KL_beamdump = df_data_KL_beamdump['scale_factor_ls']*df_data_KL_beamdump['ppfx_cv_ls']


    weights_non_KL_target = df_data_non_KL_target['scale_factor_ls']*df_data_non_KL_target['ppfx_cv_ls']
    weights_non_KL_beamdump = df_data_non_KL_beamdump['scale_factor_ls']*df_data_non_KL_beamdump['ppfx_cv_ls']


    
    
    a_KL_target, _ = numpy.histogram(df_data_KL_target[var], weights = weights_KL_target, bins=nbins, range = plotRange)
    
    #plt.hist(df_data_KL_target[var], weights = weights_KL_target, bins=nbins, range = plotRange)
    print ('Following is height for KL target')
    print (a_KL_target)
    
    a_KL_beamdump, _ = numpy.histogram(df_data_KL_beamdump[var], weights = weights_KL_beamdump, bins=nbins, range = plotRange)



    a_non_KL_target, _ = numpy.histogram(df_data_non_KL_target[var], weights = weights_non_KL_target, bins=nbins, range = plotRange)
    a_non_KL_beamdump, _ = numpy.histogram(df_data_non_KL_beamdump[var], weights = weights_non_KL_beamdump, bins=nbins, range = plotRange)

    
    """
    I have uncertainty of 100% for KL events and 100% for NC Coherent Pi0 (aka beamdump) events. 
    Those events that are not NC Coherent are target events. In total I will have 4 histograms for a sample
    
    1. KL target. The uncertainty for which will be height of the bin as it’s 100% due to it being a KL event histogram.
    2. KL beamdump. The uncertainty for which will be sqrt(2)*height of the bin right?
    3. non KL target. RMS uncertainty for 600 universes
    4. non KL beamdump. 100% due to it being a beamdump event histogram
    
    then add them in quadrature for each bin
    
    Caution: NC Coherent Pi0 is only applied to xsec and not the ppfx so needs to fix it as this function
    is onl.
    
    """

    ydata_std_dev_KL_target = a_KL_target
    ydata_std_dev_KL_beamdump = math.sqrt(1.3)*a_KL_beamdump # Think of it like a complete different histogram. It will have fraction of height

    # I need all the 600 universes for a_non_KL to be able to calculate the uncertainty.
    a_non_KL_target_arr, b_non_KL_target_arr = get_n_bins_uni(df_data_non_KL_target, var, nbins, plotRange, flux_xsec_str='ppfx', bool_is_signal=True)
    a_non_KL_target_arr.append(a_non_KL_target) # Since need the last value which will be CV in def std_dev () method


    
    ydata_std_dev_non_KL_target = std_dev(a_non_KL_target_arr)
    
    
    ydata_std_dev_non_KL_beamdump = 0.3*a_non_KL_beamdump
    
    
    print ('Sigma KL target')
    print (ydata_std_dev_KL_target)
    
    print ('Sigma KL beamdump')
    print (ydata_std_dev_KL_beamdump)
    
    print ('Sigma non KL target')
    print (ydata_std_dev_non_KL_target)
    
    print ('Sigma non KL beamdump')
    print (ydata_std_dev_non_KL_beamdump)


    for sig_KL_target, sig_KL_beamdump, sig_non_KL_target, sig_non_KL_beamdump, a_KL_target_i, a_KL_beamdump_i, a_non_KL_target_i, a_non_KL_beamdump_i in zip(ydata_std_dev_KL_target, ydata_std_dev_KL_beamdump, ydata_std_dev_non_KL_target, ydata_std_dev_non_KL_beamdump, a_KL_target, a_KL_beamdump, a_non_KL_target, a_non_KL_beamdump):
        #print ('Starting from here...')
        #print (bd)
        #print (t)
        tot = a_KL_target_i+a_KL_beamdump_i+a_non_KL_target_i+a_non_KL_beamdump_i
        
        #ydata_std_dev.append(math.sqrt((sig_KL_target*a_KL_target_i/tot)**2 + (sig_KL_beamdump*a_KL_beamdump_i/tot)**2 + (sig_non_KL_target*a_non_KL_target_i/tot)**2 + (sig_non_KL_beamdump*a_non_KL_beamdump_i/tot)**2))
        ydata_std_dev.append(math.sqrt((sig_KL_target)**2 + (sig_KL_beamdump)**2 + (sig_non_KL_target)**2 + (sig_non_KL_beamdump)**2))
    
    return ydata_std_dev

def get_ydata_std_dev_KDIF(df_data, var, nbins, plotRange):
    # For KDIF we won't have 30% uncertainty due to KDAR from beamdump
    # Just need to fix the uncertainty due to KL in flux for Signal now on this function
    
    ydata_std_dev = []
    
    df_data_non_KL = df_data.query('nu_parent_pdg_ls!=130').copy()
    df_data_KL = df_data.query('nu_parent_pdg_ls==130').copy()
    
    weights_non_KL = df_data_non_KL['scale_factor_ls']*df_data_non_KL['ppfx_cv_ls']
    weights_KL = df_data_KL['scale_factor_ls']*df_data_KL['ppfx_cv_ls']

    a_non_KL, _ = numpy.histogram(df_data_non_KL[var], weights = weights_non_KL, bins=nbins, range = plotRange)
    a_KL, _ = numpy.histogram(df_data_KL[var], weights = weights_KL, bins=nbins, range = plotRange)    
    
    a_non_KL_arr, b_non_KL_arr = get_n_bins_uni(df_data_non_KL, var, nbins, plotRange,flux_xsec_str='ppfx', bool_is_signal=True)
    a_non_KL_arr.append(a_non_KL) # Since need the last value which will be CV in def std_dev () method
    
    
    ydata_std_dev_non_KL = std_dev(a_non_KL_arr)
    ydata_std_dev_KL = a_KL


    for sig_non_KL, sig_KL, a_non_KL_i, a_KL_i in zip(ydata_std_dev_non_KL, ydata_std_dev_KL, a_non_KL, a_KL):

        tot = a_non_KL_i+a_KL_i
        
        #ydata_std_dev.append(math.sqrt((sig_non_KL*a_non_KL_i/tot)**2 + (sig_KL*a_KL_i/tot)**2))
        ydata_std_dev.append(math.sqrt((sig_non_KL)**2 + (sig_KL)**2))
    
    
    
    return ydata_std_dev

def get_n_bins_uni(df_mc, var, nbins, plotRange, flux_xsec_str, bool_is_signal=False):

    n_arr = []
    bin_arr = []
    pot_scale = 1
    
    if flux_xsec_str == 'ppfx' or flux_xsec_str == 'xsec':
        
        if bool_is_signal == False:
            if flux_xsec_str == 'ppfx':
                for uni in range (0, 600):
                # For flux uncertainties
                    weights_bkg_ppfx = df_mc['scale_factor_ls']*pot_scale*df_mc['bnb_scale_ls']*df_mc['weightSplineTimesTune_ls']*df_mc.weightsPPFX.map(lambda x: x[uni])/1000
                    a_ppfx, b_ppfx = numpy.histogram(df_mc[var], weights = weights_bkg_ppfx, bins=nbins, range = plotRange)

                    n_arr.append(a_ppfx)
                    bin_arr.append(b_ppfx)


            elif flux_xsec_str == 'xsec':
                for uni in range (0, 600):
                    # For genie xsec uncertainties
                    weights_bkg_xsec = df_mc['scale_factor_ls']*pot_scale*df_mc['bnb_scale_ls']*df_mc['ppfx_cv_ls']*df_mc.weightsGenie.map(lambda x: x[uni])/1000
                    # foolowing is test which gave me worse results with ppfx removed
                    # weights_bkg_xsec = df_mc['scale_factor_ls']*pot_scale*df_mc['bnb_scale_ls']*df_mc.weightsGenie_ls.map(lambda x: x[uni])/1000

                    # Originally when copied this fn I had the following
                    #weights_sig = df_data['final_scale']*df_data['pot_scale']*df_data['bnb_scale']*df_data.weightsGenie.map(lambda x: x[uni])/1000

                    a_xsec, b_xsec = numpy.histogram(df_mc[var], weights = weights_bkg_xsec, bins=nbins, range = plotRange)

                    n_arr.append(a_xsec)
                    bin_arr.append(b_xsec)
                    
                    
        elif bool_is_signal == True:
            print ('Final final')
            #display(df_mc)
                
            for uni in range (0, 600):
                
                weights_sig_ppfx = df_mc['scale_factor_ls']*df_mc['ppfx_cv_ls']*df_mc.weightsPPFX.map(lambda x: x[uni])/1000

                a_sig_ppfx, b_sig_ppfx = numpy.histogram(df_mc[var],weights = weights_sig_ppfx, bins=nbins,range = plotRange)
                #a,b,c = plt.bar(binc, vals, width=widths, yerr=errs, weights = weights_sig, color=data_color, range = plotRange)
                #df_data[var].to_root('out.root', key='mytree')

                n_arr.append(a_sig_ppfx)
                bin_arr.append(b_sig_ppfx)
                
        else:
            print ('Something went wrong. Cannot get n and bins array.')
            return 0,0
            
    elif flux_xsec_str == 'reint':
                
        for uni in range (0, 1000):
            # For reinteraction uncertainties
            weights_bkg_reint = df_mc['scale_factor_ls']*pot_scale*df_mc['bnb_scale_ls']*df_mc['ppfx_cv_ls']*df_mc['weightSplineTimesTune_ls']*df_mc.weightsReint.map(lambda x: x[uni])/1000

            a_reint, b_reint = numpy.histogram(df_mc[var], weights = weights_bkg_reint, bins=nbins, range = plotRange)

            n_arr.append(a_reint)
            bin_arr.append(b_reint)
            
            
    return n_arr, bin_arr



def plotter_signal_ax1 (df_data, df_data_detsys, var, ax1, ax2, nbins, f, counter_run, bdt_score = "0.0", plotRange=None, align = 'mid', sig_type_str = '', data_color='red',linestyle='solid', bnb_flat_scale = False, loc_leg = 2, data_driven_scale = False, dataplot=True, plotbar = False, legfontsize=15, plotlegend=True, plot_uni=True):
    #KDAR_KDIF_str_temp = sig_type_str.split(', ')[0]
    
    #global sig_cv
    #global sig_up
    #global sig_down
    
    
    sig_cv = 'Signal_CV'+str(counter_run)
    sig_up = 'Signal_Fluct_Up'+str(counter_run)
    sig_down = 'Signal_Fluct_Down'+str(counter_run)


    df_data_org = df_data.copy()
    df_data = df_data_org.reset_index(drop=True)
    #df_data = df_data.query('pred>='+bdt_score)
    
    
    
    a_arr, b_arr = get_n_bins_uni(df_data, var, nbins, plotRange,flux_xsec_str='ppfx', bool_is_signal=True)
    # Still need last element which is CV
    
    weights_sig_cv = df_data['scale_factor_ls']*df_data['ppfx_cv_ls']
    
    
    if plotRange is None:
        a,b,c = ax1.hist(df_data[var], label="koto signal", histtype="step",weights = weights_sig_cv, bins=nbins,align = align, lw=1.3, color=data_color,linestyle=linestyle,zorder=2)

    else:
        a,b,c = ax1.hist(df_data[var], label="koto signal", histtype="step",weights = weights_sig_cv, bins=nbins,align = align, lw=1.3, color=data_color, linestyle=linestyle, range = plotRange,zorder=2)

    f[sig_cv] = numpy.histogram(np.array(df_data[var].values), bins=nbins, weights = np.array(weights_sig_cv.values), range = plotRange)
            
    a = numpy.array(a)
    
    # Adding last variable CV to compare others with
    a_arr.append(a)
    b_arr.append(b)

    
    
    print ('==================================')
    print ('HEIGHT OF THE BINS ARE (SIGNAL)')
    print ('==================================')
    print (a)
    print ('==================================')
    print ('==================================')

    
    
    
    print ('Calculating FLUX uncertainties SIGNAL')
    
    if sig_type_str == 'KDAR':
        print ('CALCULATING KDAR FLUX UNCERTAINTY')
        ydata_std_dev_ppfx_SIGNAL = get_ydata_std_dev_KDAR(df_data, var, nbins, plotRange)
        
    else:
        print ('CALCULATING KDIF FLUX UNCERTAINTY')
        ydata_std_dev_ppfx_SIGNAL = get_ydata_std_dev_KDIF(df_data, var, nbins, plotRange)
    
    
    


    print ('Calculating statistical uncertainties SIGNAL')
    
    
    counts_hist_unwt_SIGNAL, _ = numpy.histogram(df_data[var], bins=nbins, range = plotRange)#, density=True)
   
    
    print ('==============================================')
    print ('ORIGINAL UNWEIGHTED HEIGHT OF THE BINS ARE (SIGNAL)')
    print ('==============================================')
    print (counts_hist_unwt_SIGNAL)
    print ('==============================================')
    print ('==============================================')


    
    counts_hist_unwt_SIGNAL_sigma = numpy.sqrt(counts_hist_unwt_SIGNAL)
    counts_hist_unwt_SIGNAL_sigma = numpy.array(counts_hist_unwt_SIGNAL_sigma)
    #print ('WITHOUT POT scaling pure sample square-root is...')
    #print (counts_hist_unwt_sigma)
    
    frac_counts_hist_unwt_SIGNAL_sigma = counts_hist_unwt_SIGNAL_sigma/counts_hist_unwt_SIGNAL
    frac_counts_hist_unwt_SIGNAL_sigma = numpy.array(frac_counts_hist_unwt_SIGNAL_sigma)
    
    print ('Fractional uncertainty (SIGNAL) is...')
    print (frac_counts_hist_unwt_SIGNAL_sigma)
    print ('Multiplying this fraction with height of each bin')
    stat_sigma_SIGNAL = numpy.multiply (a, frac_counts_hist_unwt_SIGNAL_sigma)
    stat_sigma_SIGNAL = numpy.array(stat_sigma_SIGNAL)
    
    
    print (stat_sigma_SIGNAL)
 




    print ('SIGNAL Calculating DETECTOR systematics SIGNAL')
    frac_ydata_std_dev_detsys_SIGNAL = get_frac_ydata_std_dev_detsys_SIGNAL (df_data_detsys.copy(), var, bins=1, plotRange=plotRange, bdt_score_str = bdt_score)
    frac_ydata_std_dev_detsys_SIGNAL = numpy.array(frac_ydata_std_dev_detsys_SIGNAL)
    
    ydata_std_dev_detsys_SIGNAL = numpy.multiply (a, frac_ydata_std_dev_detsys_SIGNAL)
    ydata_std_dev_detsys_SIGNAL = numpy.array(ydata_std_dev_detsys_SIGNAL)

    print ('==================================')
    print ('ALL UNCERTAINTIES (SIGNAL)')
    print ('==================================')
    print ('==================================')
    print ('Flux (SIGNAL)')
    print (ydata_std_dev_ppfx_SIGNAL)
    print ('Statistical (SIGNAL)')
    print (stat_sigma_SIGNAL)
    print ('DETSYS fractional (SIGNAL)')
    print (frac_ydata_std_dev_detsys_SIGNAL)
    print ('DETSYS (SIGNAL)')
    print (ydata_std_dev_detsys_SIGNAL)
    print ('==================================')
    print ('==================================')
    
    #$#
    print ('==================================')
    print ('Adding in quadrature these uncertainties gives for stat, flux and detsys  (SIGNAL)')
    print ('==================================')
    
    
    stat_sigma_SIGNAL_2 = np.multiply(stat_sigma_SIGNAL, stat_sigma_SIGNAL)
    ydata_std_dev_ppfx_SIGNAL_2 = np.multiply(ydata_std_dev_ppfx_SIGNAL, ydata_std_dev_ppfx_SIGNAL)
    ydata_std_dev_detsys_SIGNAL_2 = np.multiply(ydata_std_dev_detsys_SIGNAL, ydata_std_dev_detsys_SIGNAL)
    
    total_stat_sigma_SIGNAL = np.array([stat_sigma_SIGNAL_2, ydata_std_dev_ppfx_SIGNAL_2, ydata_std_dev_detsys_SIGNAL_2])
    total_stat_sigma_SIGNAL = total_stat_sigma_SIGNAL.sum(axis=0)
    final_sigma_SIGNAL = np.sqrt(total_stat_sigma_SIGNAL)
    
    final_sigma_SIGNAL = numpy.array(final_sigma_SIGNAL)
    
    print (final_sigma_SIGNAL)
    print ('==================================')
    
    for idx, (h_i, h_s_i) in enumerate (zip(a, final_sigma_SIGNAL)):
        if h_s_i == np.inf or h_s_i == -np.inf or h_s_i > 1.3*h_i:
            final_sigma_SIGNAL[idx] = h_i
        if math.isnan(h_s_i):
            final_sigma_SIGNAL[idx] = 0
    
    # ydata_std_dev
    
    binc = 0.5*(b[1:]+b[:-1])
    ax1.errorbar(binc, a, yerr=final_sigma_SIGNAL, fmt='o', color=data_color, markersize=3, label='_nolegend_', elinewidth=1, zorder=3)

    w_plus = np.array(weights_sig_cv.values)
    w_minus = np.array(weights_sig_cv.values)

    
    # a[n] is the height of the bin n for CV. So all events are extracted back again with their weights
    # For each event there's a weight and that weight needs to be changed depending on where the event is (which bin it is in)
    # in histogram. If an event lies in bin 2 with and if bin 2 has some uncertainty of 2.0 then it's weight 
    # will be changed by a factor so that it has a specific height.
    inds = np.digitize(np.array(df_data[var].values), bins=b)


    for n in range(np.array(df_data[var].values).size):
        # print ('How many n are there?...: ' + str(n))
        # Get all the a[n] with its w[n] and find the bin each a[n], w[n] belong to using if condition
        for i in range (binc.size):
            # Find the bin i that has ydata_std_dev[i]

            # An event can be either in bin 1 or in bin 2 .... or in bin 25.
            if inds[n]-1 == i:
                """
                For each event I asked what should be multiplied to the event so that they are 
                Take an example. I had a bin with CV = 4 and uncertainty = 1.8 and I wanted that bin 
                to be 5.8 for plus so I asked 4*x=5.8 should give me x=5.8/4 = (4+1.8)/4 = (a+std_dev)/a

                For fractional uncertainty I wanted 1.8/4 for both plus and minus as uncertainties were
                symmetrical. That gives me 

                # Each a[n]'s weight (stored in w[n]) are modified by a factor above to get the desired plot
                
                
                wfactor_plus = (final_sigma_SIGNAL[i]+a[i])/a[i]
                w_plus[n] = w_plus[n]*wfactor_plus

                wfactor_minus = (a[i]-final_sigma_SIGNAL[i])/a[i]
                w_minus[n] = w_minus[n]*wfactor_minus


                """
                # Following is for fractional uncertainties.
                # This has to go in the .root file that will be later used in the Collie.

                wfactor_plus = (final_sigma_SIGNAL[i])/a[i]
                w_plus[n] = w_plus[n]*wfactor_plus/a[i]

                wfactor_minus = (final_sigma_SIGNAL[i])/a[i]
                w_minus[n] = w_minus[n]*wfactor_minus/a[i]
                
    ##$$
    
    
    # y_plus,_,_ = ax1.hist(df_data[var], label="NuMI Overlay Background", histtype="step",linestyle="dashed",weights = w_plus, bins=nbins,align = align, lw=1.3,color=data_color,range = plotRange, zorder=2)
    f[sig_up] = numpy.histogram(df_data[var], bins=nbins, weights = w_plus, range=plotRange)


    # ax1.hist(df_data[var], label="NuMI Overlay Background", histtype="step",linestyle="dashed",weights = w_minus, bins=nbins,align = align, lw=1.3,color=data_color,range = plotRange, zorder=2)
    f[sig_down] = numpy.histogram(df_data[var], bins=nbins, weights = w_minus, range=plotRange)
    
    #$$
    # offset = matplotlib.text.OffsetFrom(leg, (1.0, 0.0))
    # ax1.annotate(Data_str.replace('_',' ')+' – ' + str(numpy.sum(a).round(2)), xy=(0,0),size=16,
    #                 xycoords='figure fraction', xytext=(-170,-50), textcoords=offset)
                    


    """
    sub_bar_plot = ax2.bar(binc, (final_sigma_SIGNAL)/a, width = binc[1]-binc[0], fill=False, linewidth=2)
    ax2.set_ylim([.0, 1.0])
    ax2.set_ylabel(r'Fractional uncertainty',fontsize=17)
    #print ('Where is NaN')
    #print (sub_bar_plot)
    #autolabel(sub_bar_plot, ax2)
    
    
    
    Data_str = Data_str.replace('_',' ')+' – ' + str(numpy.sum(a).round(2))

    LABELS = []
    LABELS.append(Data_str)
    
    leg_loc = {0: "best",
           1: "upper right",
           2: "upper left",
           3: "lower left",
           4: "lower right",
           6: "center left",
           7: "center right",
           8: "lower center",
           9: "upper center",
           10: "center"}
    leg_loc_num = 0
    plt.rcParams['legend.loc'] = leg_loc[leg_loc_num]

    
    
    ax1.legend(LABELS, fontsize=legfontsize)
    """
    # return a_arr,b_arr, math.ceil(max(y_plus)) #,c_arr
    
    finite_a = numpy.nan_to_num(a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    #finite_a_indx = finite_a.index(max(finite_a))
    print ('Finite a')
    print (finite_a)
    
    finite_sigma_SIGNAL = numpy.nan_to_num(final_sigma_SIGNAL, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    print ('Finite sigma SIGNAL')
    print (finite_sigma_SIGNAL)
    
    # ax1.set_ylim(top=(np.max(finite_final_sigma+finite_counts_hist[-1]))*1.6, bottom=0)
    
    print ('We will therefore return')
    print (np.max(finite_sigma_SIGNAL+finite_a))
    
    
    return np.max(finite_sigma_SIGNAL+finite_a), b, str(numpy.sum(a).round(2))


    

def labelaxis (ax1, ax2):
    label_fontsize = 20
    figsize_x=6*1.618
    figsize_y=10.0
    ax1.set_ylabel('Entries',fontsize=label_fontsize)
    ax1.tick_params(axis="y", labelsize=17)
    ax1.tick_params(axis="y", labelsize=17)
    ax2.tick_params(axis="x", labelsize=17)
    ax2.tick_params(axis="y", labelsize=17)
    ax2.set_ylabel(r'$\frac{Data}{EXT + MC}$',fontsize=label_fontsize+4)

def Main_loader(Run, mass, nshr, KDIF_KDAR_str, bool_full_sample=True):
    
    mc_pot_fhc = 2.33652e+21
    mc_pot_rhc = 1.98937e+21

    dirt_pot_fhc = 1.42143e+21
    # Old
    #dirt_pot_rhc = 4.65831e20

    # New Genie Tune
    dirt_pot_rhc = 1.03226e+21

    # The EXT scaling would then be scale = EA9CNT[_wcut] /  EXT_NUMIwin_FEMBeamTriggerAlgo
    # On beam's EA9CNT_wcut/Off beam's EXT (leared from Krishan's wiki that it is EXT_NUMIwin_FEMBeamTriggerAlgo)

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

    # On beam's tortgt_wcut/MC_POT
    scale_nu_fhc = OnBeam_tortgt_wcut_fhc/mc_pot_fhc
    scale_nu_rhc = OnBeam_tortgt_wcut_rhc/mc_pot_rhc


    #On beam's tortgt_wcut/MC_dirt_POT
    scale_dirt_fhc = OnBeam_tortgt_wcut_fhc/dirt_pot_fhc
    scale_dirt_rhc = OnBeam_tortgt_wcut_rhc/dirt_pot_rhc

    scale_dirt_fhc = scale_dirt_fhc*0.75
    scale_dirt_rhc = scale_dirt_rhc*0.35

    
    

    """
    #So i scale EXT down by *0.98 in both run1 and run3

    #for dirt
    #i scale down by
    #*0.75 (run1) and *0.35 (run3)

    """


    #Run = 'Run1' or 'Run3'
    #mass = '100' or '150' or '200'
    #nshr = '1shr' or '2shr'
    #KDIF_KDAR_str = 'KDIF' or 'KDAR'
    
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
    
    main_input_dir = './BDT_inputs_pkl/'
    
    
    ############
    ### DATA ###
    ############
    
    if bool_full_sample == True:
        print ('No data')
        df_data = pd.read_pickle(main_input_dir+'AllVar_Selected_'+Run+'_NuMI_opendata_'+nshr+'_NEW.pkl')
        df_data ['pred'] = 1.0

    else:
        df_data = pd.read_pickle(main_input_dir+'AllVar_Selected_'+Run+'_NuMI_Opendata_'+KDIF_KDAR_str+'_'+mass+'_'+nshr+'_PPFX_pred_NEW.pkl')
        #df_data = pd.read_pickle(main_input_dir+'AllVar_Selected_'+Run+'_NuMI_opendata_'+nshr+'_NEW.pkl')
        #df_data ['pred'] = 1.0
    
    #############
    #### SIG ####
    #############
    if bool_full_sample == True:
        print ('No signal')
        
    else:
        df_sig = pd.read_pickle(main_input_dir+'AllVar_Selected_'+Run+'_NuMI_Signal_'+KDIF_KDAR_str+'_'+mass+'_'+nshr+'_PPFX_pred_NEW.pkl')
        # We are already using the POT of the test_sample so no need of finding how many were trained for Signal.
        # test_sample_POT_ls has fraction*original_signal_pot
        # but notice that eventually goes in the denominator. so 2e20/1e22 = 1/100 becomes 2e20/1e21 = 1/10

        frac_sig_test_sample = len(df_sig.query('is_trained == 0'))/len(df_sig)
        # The above will be about 0.4
        df_sig['frac_test_sample'] = frac_sig_test_sample

        df_sig = df_sig.query('is_trained==0')
        df_sig.eval('scale_factor_ls = (1/frac_test_sample)*KDAR_scale*'+str(OnBeam_tortgt_wcut)+'/test_sample_POT_ls',inplace=True)


    #print ('ORIGINAL SIGNAL')
    #display(df_sig)
    
    #############
    #### OVL ####
    #############
    print ('SUCCESSFULLY LOADED THE PICKLE FILES OVERLAY')

    if bool_full_sample == True:
        df_ovl = pd.read_pickle(main_input_dir+'AllVar_Selected_'+Run+'_NuMI_Overlay_'+nshr+'_WEIGHTS_NEW.pkl')
        df_ovl.eval('scale_factor_ls = KDAR_scale*'+str(scale_nu),inplace=True)
        df_ovl ['pred'] = 1.0
    else:
        df_ovl = pd.read_pickle(main_input_dir+'AllVar_Selected_'+Run+'_NuMI_Overlay_'+KDIF_KDAR_str+'_'+mass+'_'+nshr+'_PPFX_pred_NEW.pkl')
        frac_ovl_test_sample = len(df_ovl.query('is_trained == 0'))/len(df_ovl)
        # The above will be about 0.4
        df_ovl['frac_test_sample'] = frac_ovl_test_sample

        df_ovl = df_ovl.query('is_trained==0')
        df_ovl.eval('scale_factor_ls = (1/frac_test_sample)*KDAR_scale*'+str(scale_nu),inplace=True)
    
    # Let us understand this frac_test_sample
    # say I originally had 1000 events = 2e21 POT of which I trained 600 events and test 400 events
    # The POT of the sample will now become .4*2e21 POT and this is 8e20 POT
    # Now you have to scale more so to make it equivalent to data. We are scaling more as frac_test_sample < 1.0
    
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

    
    #print ('Following is sample for OVERLAY...$$$')
    #display(df_ovl)
    
    
    ############
    ### DIRT ###
    ############
    
    
    if bool_full_sample == True:
        df_dirt = pd.read_pickle(main_input_dir+'AllVar_Selected_'+Run+'_NuMI_Dirt_'+nshr+'_WEIGHTS_NEW.pkl')
        df_dirt.eval('scale_factor_ls = KDAR_scale*'+str(scale_dirt),inplace=True)
        df_dirt ['pred'] = 1.0
    else:
        df_dirt = pd.read_pickle(main_input_dir+'AllVar_Selected_'+Run+'_NuMI_Dirt_'+KDIF_KDAR_str+'_'+mass+'_'+nshr+'_PPFX_pred_NEW.pkl')
        frac_dirt_test_sample = len(df_dirt.query('is_trained == 0'))/len(df_dirt)
        df_dirt['frac_test_sample'] = frac_dirt_test_sample

        df_dirt = df_dirt.query('is_trained==0')
        df_dirt.eval('scale_factor_ls = (1/frac_test_sample)*KDAR_scale*'+str(scale_dirt),inplace=True)
    
    df_dirt['is_good'] = 0
    
    #print ('Following is sample for DIRT...$$$')
    #display(df_dirt)
    
    
    #############
    #### EXT ####
    #############
    

    if bool_full_sample == True:
        df_ext = pd.read_pickle(main_input_dir+'AllVar_Selected_'+Run+'_NuMI_Ext_'+nshr+'_WEIGHTS_NEW.pkl')
        df_ext.eval('scale_factor_ls = KDAR_scale*'+str(scale_ext),inplace=True)
        df_ext ['pred'] = 1.0
    else:
        df_ext = pd.read_pickle(main_input_dir+'AllVar_Selected_'+Run+'_NuMI_Ext_'+KDIF_KDAR_str+'_'+mass+'_'+nshr+'_PPFX_pred_NEW.pkl')
        frac_ext_test_sample = len(df_ext.query('is_trained == 0'))/len(df_ext)
        df_ext['frac_test_sample'] = frac_ext_test_sample

        df_ext = df_ext.query('is_trained==0')
        df_ext.eval('scale_factor_ls = (1/frac_test_sample)*KDAR_scale*'+str(scale_ext),inplace=True)
   



    ################
    #### Detsys ####
    ################

    if mass == '125' or mass == '130' or mass == '135' or mass == '140' or mass == '145':
        mass = '150'
    
    detsys_sig_input_dir = './Final_detector_systematics_SIGNAL/'+mass+'_MeV/'
    detsys_bkg_input_dir = './Final_detector_systematics_BACKGROUND/'+mass+'_MeV/'

    KDIF_KDAR_str_lower = KDIF_KDAR_str.lower()
    df_sig_detsys = pd.read_pickle(detsys_sig_input_dir + 'Signal_' + fhc_rhc_str + '_' + mass + '_MeV_' + nshr + '_' + KDIF_KDAR_str_lower +'.pkl')
    df_ovl_detsys = pd.read_pickle(detsys_bkg_input_dir + "bkg_" + fhc_rhc_str + '_' + mass + '_MeV_' + nshr + '_' + KDIF_KDAR_str_lower +'.pkl')


    
    
    ############
    #### MC ####
    ############
    #df_ovl = fix_weightSplineTimesTune(df_ovl)
    #df_dirt = fix_weightSplineTimesTune(df_dirt)
    
    df_mc = pd.concat([df_ovl, df_dirt], ignore_index=True)

    # For signal we are not using is_trained==0 because these are set later and some RSEs are overlapping between
    # raw data we used in the training and the test sample. The kinematics are different, I have checked that 
    # so don't worry
    # 
    # We cannot send the mc and ext to be to_train/is_trained == 0 because we wanna plot data/MC for full to validate
    # 
    #print ('ORIGINAL SIGNAL')
    #display (df_sig)
    
    # print ('ORIGINAL OVL')
    # display (df_ovl)
    
    if bool_full_sample == True:
        print ('NO LOGIT FOR SIGNAL..')
    else:
        df_sig['logit_pred'] = logit(df_sig['pred'])
    
    df_ovl['logit_pred'] = logit(df_ovl['pred'])
    df_dirt['logit_pred'] = logit(df_dirt['pred'])
    df_mc['logit_pred'] = logit(df_mc['pred'])
    df_ext['logit_pred'] = logit(df_ext['pred'])
    df_data['logit_pred'] = logit(df_data['pred'])
    
    pred_column_sig = [col for col in df_sig_detsys if col.startswith('pred')]
    pred_column_ovl = [col for col in df_ovl_detsys if col.startswith('pred')]
    
    for pred in pred_column_sig:
        #    print (pred)
        df_sig_detsys['logit_'+pred] = logit(df_sig_detsys[pred])
    
    for pred in pred_column_ovl:
        #    print (pred)
        df_ovl_detsys['logit_'+pred] = logit(df_ovl_detsys[pred])

    
    
    #return df_sig.query('to_train == 0'), df_ovl, df_dirt, df_mc, df_ext, df_data, df_sig_detsys, df_ovl_detsys
    print ('SUCCESSFULLY LOADED THE PICKLE FILES')
    
    
    
    if bool_full_sample == True:
        return df_ovl, df_dirt, df_mc, df_ext, df_data, df_sig_detsys, df_ovl_detsys
    else:
        #df_sig = fix_weightSplineTimesTune(df_sig)
        return df_sig, df_ovl, df_dirt, df_mc, df_ext, df_data, df_sig_detsys, df_ovl_detsys
    
    

def get_min_var(var,df1,df2,df3,df4):
    mx=min([df[var].min() for df in [df1,df2,df3,df4]])
    print('found min ',var,' = ',mx)
    return mx
    
def get_max_var(var,df1,df2,df3,df4):
    mx=max([df[var].max() for df in [df1,df2,df3,df4]])
    print('found max ',var,' = ',mx)
    return mx


def Create_collie_inputs(xlabel_main, nbins, var, plotRange, f, bdt_score, mass, KDIF_KDAR_str, Run, shr, counter_run, bool_full_sample=False, flag_1bin=False, flag_logy = False):
    # For variable pred it is collie inputs
    
    leg_loc_num = 2
    ncol = 3
    
    if bool_full_sample == False:
        sub_plotRange = [0.0, 2.0]
    else:
        sub_plotRange = [0.5, 1.5]
    
    if bool_full_sample == True:
        df_ovl, df_dirt, df_mc, df_ext, df_data, df_sig_detsys, df_ovl_detsys = Main_loader(Run, mass, shr, KDIF_KDAR_str, bool_full_sample=bool_full_sample)
    
    else:
        df_sig, df_ovl, df_dirt, df_mc, df_ext, df_data, df_sig_detsys, df_ovl_detsys = Main_loader(Run, mass, shr, KDIF_KDAR_str, bool_full_sample=bool_full_sample)

    label_fontsize = 20
    figsize_x=6*1.618
    figsize_y=10.0

    fig, (ax1, ax2) = plt.subplots(nrows=2,figsize=(figsize_x,figsize_y),gridspec_kw = {'height_ratios':[3, 1]}, sharex=True)
    print (Run, shr)
    
    

    if shr == '1shr':
        xlabel = xlabel_main + r' (one shower) , $M_S =$ ' + mass +' MeV'
    elif shr == '2shr':
        xlabel = xlabel_main + r' (two showers) , $M_S =$ ' + mass +' MeV '

    if Run == 'Run1':
        title = 'Run1, ' + r'$2.002 \times 10^{20}$ POT'

        
    elif Run == 'Run3':
        title = 'Run3, ' + r'$5.009 \times 10^{20}$ POT'


    #xlabel = xlabel.replace(' (one shower)','')
    #xlabel = xlabel.replace(' (two showers)','')

    ax2.set_xlabel(xlabel_main,fontsize=label_fontsize)

    ax1.set_title('MicroBooNE NuMI data',  loc='left', fontsize=22)
    ax1.set_title(title,  loc='right', fontsize=22)
    if flag_logy: ax1.set_yscale('log')
    
    labelaxis (ax1, ax2)

    # Main_plotter (df_data, df_mc, df_ext, var=var, xlabel = xlabel, title=title, nbins=nbins, 
    #               plotRange=plotRange, sub_plotRange=sub_plotRange, 
    #               leg_loc_num=leg_loc_num, ylim=ylim, ncol=ncol)
    data_color = 'red'
    sig_type_str = KDIF_KDAR_str


    if plotRange[0] is None:
        plotRange[0] = get_min_var(var,df_sig.query('pred >=' + bdt_score),df_data.query('pred >=' + bdt_score), df_mc.query('pred >=' + bdt_score), df_ext.query('pred >=' + bdt_score))
    if plotRange[-1] is None:
        plotRange[-1] = get_max_var(var,df_sig.query('pred >=' + bdt_score),df_data.query('pred >=' + bdt_score), df_mc.query('pred >=' + bdt_score), df_ext.query('pred >=' + bdt_score))
    
    
    #nbins = histedges_equalN(df_mc.query('pred >=' + bdt_score)[var], nbins)
    
    
    #ylim, bins, sig_count = plotter_signal_ax1(df_sig, df_sig_detsys, var=var, loc_leg=leg_loc_num,sig_type_str=sig_type_str, nbins=nbins, ax1=ax1, ax2=ax2, data_color=data_color, plotRange=plotRange, plot_uni=False)
    if bool_full_sample == True:
        print ('No signal')
        ylim = 0
        sig_count = 0
        bins = nbins
        sig_type_str = None
        
    else:
        ylim, bins, sig_count = plotter_signal_ax1(df_sig.query('pred >=' + bdt_score), df_sig_detsys, var=var, loc_leg=leg_loc_num,sig_type_str=sig_type_str, nbins=nbins, f=f, counter_run = counter_run, bdt_score=bdt_score,  ax1=ax1, ax2=ax2, data_color=data_color, plotRange=plotRange, plot_uni=False)


    
        # Return the y max for bkg and compare it with signal and then do ax1.set_yscale
        # Because sometimes signal will be larger than bkg
        # sometimes data will be larger.
        sig_type_str = sig_type_str + ' (' + str(sig_count) + ')'

    
    #ylim=50
    if flag_1bin == True:
        bins=1
        
    output = plotter(df_data.query('pred >=' + bdt_score), df_mc.query('pred >=' + bdt_score), df_ext.query('pred >=' + bdt_score),
            df_ovl_detsys, var=var, f=f, counter_run=counter_run, bdt_score=bdt_score, bnb_flat_scale=True, nbins=bins, ax1=ax1, ax2=ax2, plotRange = plotRange, 
            sub_plotRange=sub_plotRange, leg_loc_num=leg_loc_num, ylim=ylim, ncol=ncol, xlabel=xlabel, sig_type_str=sig_type_str, flag_1bin=flag_1bin, flag_logy=flag_logy)

                
    #fig.savefig('./BDT_score_with_uncertainties/'+KDIF_KDAR_str+'_'+shr+'_'+Run+'_'+mass+'_'+bdt_score+'.pdf')
    if bool_full_sample == True:
        fig.savefig('./BDT_score_with_uncertainties/thesis_plots/'+var+'_'+KDIF_KDAR_str+'_'+shr+'_'+Run+'_'+mass+'_'+bdt_score+'.pdf')
    else:
        fig.savefig('./BDT_score_with_uncertainties/Justin/'+var+'_'+KDIF_KDAR_str+'_'+shr+'_'+Run+'_'+mass+'_'+bdt_score+'.pdf')

    return output

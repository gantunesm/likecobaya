import numpy as np
import matplotlib.pylab as plt
import numpy as np, time, astropy.io.fits, os, sys
#from kxgal import data_utils #my own utilities
#from kxgal import cls_tools #my own utilities
import pymaster as nmt
import sacc
import healpy as hp  

pdfs = []
clgg_all =[]
clkg_all = []
cov_gg_all = []
cov_kg_all = [] 

pz_path = '/home/gabriela/Documents/Pesquisa/kxgal_des/data_output/pz/'
cov_path = '/home/gabriela/Documents/Pesquisa/kxgal_des/data_output/cov_nmt/'
cls_path = '/home/gabriela/Documents/Pesquisa/kxgal_des/data_output/cls/'
 
zs=['z1','z2','z3','z4','z5','z6']

zzs = [] 
for ii in range(len(zs)):
    
    cl_gg_data = np.load(cls_path +'clgg_'+zs[ii]+'_actregion.npy')
    cl_kg_data = np.load(cls_path +'clkg_'+zs[ii]+'_actregion.npy' )
     
    
    #loading covariance:
    cov_nmt_gg = np.load(cov_path+'cov_gg_smoothed_'+zs[ii]+'.npy')
    cov_nmt_kg = np.load(cov_path+'cov_kg_smoothed_'+zs[ii]+'.npy')
    
    zz, pdf = np.loadtxt(pz_path+'pz_act_all_norm_'+zs[ii]+'.txt')

    cov_gg_all.append(cov_nmt_gg)
    cov_kg_all.append(cov_nmt_kg)
    
    clkg_all.append(cl_kg_data)
    clgg_all.append(cl_gg_data)
    pdfs.append(pdf)

    
binner = nmt.NmtBin(1024, nlb=  30)
ell_b = binner.get_effective_ells()
n_ell_large = 3*1024
n_ell = 102 
d_ell = 30
ells_large = np.arange(n_ell_large)

s = sacc.Sacc()
# GC-z1
ells = binner.get_effective_ells()

s.add_tracer('NZ', 'DESgc__0',  # Name
             quantity='galaxy_density',  # Quantity
             spin=0,  # Spin
             z=zzs[0],  # z
             nz= pdfs[0])  # nz
# GC-z1
s.add_tracer('NZ', 'DESgc__1',  # Name
             quantity='galaxy_density',  # Quantity
             spin=0,  # Spin
             z=zzs[1],  # z
             nz= pdfs[1])  # nz

# GC-z1
s.add_tracer('NZ', 'DESgc__2',  # Name
             quantity='galaxy_density',  # Quantity
             spin=0,  # Spin
             z=zzs[2],  # z
             nz= pdfs[2])  # nz

# GC-z1
s.add_tracer('NZ', 'DESgc__3',  # Name
             quantity='galaxy_density',  # Quantity
             spin=0,  # Spin
             z=zzs[3],  # z
             nz= pdfs[3])  # nz

# GC-z1
s.add_tracer('NZ', 'DESgc__4',  # Name
             quantity='galaxy_density',  # Quantity
             spin=0,  # Spin
             z=zzs[4],  # z
             nz= pdfs[4])  # nz

# GC-z1
s.add_tracer('NZ', 'DESgc__5',  # Name
             quantity='galaxy_density',  # Quantity
             spin=0,  # Spin
             z=zzs[5],  # z
             nz= pdfs[5])  # nz

 

# # CMBK
s.add_tracer('misc', 'ACTcv',  # Name
             quantity='cmb_convergence',  # Quantity
             spin=0, ell= np.arange(3*1024)  # Spin
              )  

#Read total mask:
path_maps = '/home/gabriela/Documents/Pesquisa/kxgal_des/data_input/maps/'

mask_1024 = hp.read_map('/home/gabriela/Documents/Pesquisa/kxgal_des/data_input/maps/mask_des_ns1024.fits')
W_c = hp.read_map(path_maps+'mask_ACT_cmblensing_d56_baseline_ns1024.fits')
mask_total = W_c * mask_1024


f0  = nmt.NmtField(mask_total, [mask_total])
wsp = nmt.NmtWorkspace()
wsp.compute_coupling_matrix(f0 , f0 , binner)  


wins = wsp.get_bandpower_windows()
wins_test = (wins[0,:,0])
wins = sacc.BandpowerWindow(ells_large, wins_test.T)


# Create a SACC bandpower window object
# wins = sacc.BandpowerWindow(ells_large, window_single.T)
# wins.nell
# GC-GC
s.add_ell_cl('cl_00',  # Data type
             'DESgc__0',  # 1st tracer's name
             'DESgc__0',  # 2nd tracer's name
             ells,  # Effective multipole
             clgg_all[0],  # Power spectrum values
             window=wins,  # Bandpower windows
            )

s.add_ell_cl('cl_00',  # Data type
             'DESgc__1',  # 1st tracer's name
             'DESgc__1',  # 2nd tracer's name
             ells,  # Effective multipole
             clgg_all[1],  # Power spectrum values
             window=wins,  # Bandpower windows
            )


s.add_ell_cl('cl_00',  # Data type
             'DESgc__2',  # 1st tracer's name
             'DESgc__2',  # 2nd tracer's name
             ells,  # Effective multipole
             clgg_all[2],  # Power spectrum values
             window=wins,  # Bandpower windows
            )

s.add_ell_cl('cl_00',  # Data type
             'DESgc__3',  # 1st tracer's name
             'DESgc__3',  # 2nd tracer's name
             ells,  # Effective multipole
             clgg_all[3],  # Power spectrum values
             window=wins,  # Bandpower windows
            )


s.add_ell_cl('cl_00',  # Data type
             'DESgc__4',  # 1st tracer's name
             'DESgc__4',  # 2nd tracer's name
             ells,  # Effective multipole
             clgg_all[4],  # Power spectrum values
             window=wins,  # Bandpower windows
            )

s.add_ell_cl('cl_00',  # Data type
             'DESgc__5',  # 1st tracer's name
             'DESgc__5',  # 2nd tracer's name
             ells,  # Effective multipole
             clgg_all[5],  # Power spectrum values
             window=wins,  # Bandpower windows
            )
# # GC- CMBK
 
# # GC-CMBK
s.add_ell_cl('cl_00', 'DESgc__0', 'ACTcv', ells, clkg_all[0], window=wins)
s.add_ell_cl('cl_00', 'DESgc__1', 'ACTcv', ells, clkg_all[1], window=wins)
s.add_ell_cl('cl_00', 'DESgc__2', 'ACTcv', ells, clkg_all[2], window=wins)
s.add_ell_cl('cl_00', 'DESgc__3', 'ACTcv', ells, clkg_all[3], window=wins)
s.add_ell_cl('cl_00', 'DESgc__4', 'ACTcv', ells, clkg_all[4], window=wins)
s.add_ell_cl('cl_00', 'DESgc__5', 'ACTcv', ells,  clkg_all[5], window=wins)

from scipy.linalg import block_diag

d = block_diag(cov_gg_all[0],cov_gg_all[1],cov_gg_all[2],cov_gg_all[3],cov_gg_all[4],cov_gg_all[5], cov_kg_all[0], cov_kg_all[1], cov_kg_all[2], cov_kg_all[3], cov_kg_all[4], cov_kg_all[5])
s.add_covariance(d)
s.save_fits("/home/gabriela/Documents/Pesquisa/kxgal_des/data_output/sacc/sacc_all_realdata.fits", overwrite=True)
 

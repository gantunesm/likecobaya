import numpy as np
from scipy.interpolate import interp1d
import pyccl as ccl
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
import sacc



class likecobaya(Likelihood):
    magnification : str #True or False
    pz_errors: str #shift, None or shift_stretch
    input_params_prefix: str 
    input_file : str #name of the sacc file containing the Cls, pdf, covariance

    # List of bin names
    tracers: list = []
    tracer_combinations: dict = {}
    defaults: dict = {}
    
  

    def initialize(self):
        """
        Load the data file, apply the cuts, select the important flags contained in the .yaml file, etc
        """

        # I) Read SACC file
        self._read_tracer_combinations()
        self.sobj = self._load_file(self.input_file)
    
        # Store info in a metadata dictionary. This is useful to avoid computing the bandpower function in each call.
        self.data_metadata = self._extract_tracers_metadata(self.sobj)
        # Load data 
        self.data_array = self.sobj.mean
        # Load covariance matrix
        self.cov = self.sobj.covariance.covmat
        # Store inverse covariance with cuts already applied:
        self.icov = np.linalg.inv(self.cov)
        
        #Some checks useful to see the .out file if everything is making sense:
        print(np.shape(self.data_array) ,' shape data')
        print(np.shape(self.cov), 'shape covariance')


    def _read_tracer_combinations(self):
        # Change the dictionary keys if it is not tuple, remove comma, parenthesis
        k0 =  list(self.tracer_combinations.keys())[0]
        if type(k0) is not tuple:
            # change str -> tuple
            d = {}
            for k, v in self.tracer_combinations.items():
                k1 = k
                k1 = k1.replace('(', '').replace(')', '').replace(',', '')
                # Remove possible ' or "
                k1 = k1.replace('"', '').replace("'", '')
                d[tuple(k1.split())] = v
            self.tracer_combinations = d.copy()
        return


    def _load_file(self, sacc_file):
        print(f'Loading {sacc_file}')
        s = sacc.Sacc.load_fits(sacc_file)
        # Check used tracers are in the sacc file
        tracers_sacc = [trd for trd in s.tracers]
        # Check all tracers are in the sacc file
        for tr in self.tracers:
            if tr not in tracers_sacc:
                raise ValueError('The tracer {} is not present in {}'.format(tr, sacc_file))

        # Remove tracers that are not listed in the .yaml file and corresponding Cls, cov
        for tr in tracers_sacc:
            if tr not in self.tracers:
                # Loop through the tracer combinations to spot those with the
                # unused tracer
                for trs_sacc in s.get_tracer_combinations():
                    if tr in trs_sacc:
                        s.remove_selection(tracers=trs_sacc)
                del s.tracers[tr]


        print('Applying ell cuts')
        for trs in s.get_tracer_combinations():
            trs_input = trs
            # Check if trs is in tracer_combinations
            if trs not in self.tracer_combinations:
                trs_input = trs[::-1]
                #print('here') #trying to debug
            # Check if trs ordered the other way round is in
            # tracer_combinations and remove it if not
            if trs_input not in self.tracer_combinations:
                s.remove_selection(tracers=trs)
                #print('There')#trying to debug
                continue

            trs_val = self.tracer_combinations[trs_input]
            lmin = trs_val.get('lmin', self.defaults['lmin'])
            lmax = trs_val.get('lmax', self.defaults['lmax'])
            self.lmin = lmin
            self.lmax = lmax
            print(trs, lmin, lmax, ' tracer, lmin and lmax')
            s.remove_selection(ell__lt=lmin, tracers=trs)
            s.remove_selection(ell__gt=lmax, tracers=trs)
        return s



    def _get_dtype_for_trs(self, tr1, tr2):
        dt1 = self.sobj.get_tracer(tr1).quantity
        dt2 = self.sobj.get_tracer(tr2).quantity

        dt_to_suffix = {'galaxy_density': '0', 'cmb_convergence': '0'
                        } #Sacc format labels! 

        dtype = 'cl_'#Sacc format labels! 

        dtype += dt_to_suffix[dt1]
        dtype += dt_to_suffix[dt2]
 
        return dtype



    def _extract_tracers_metadata(self, sfile):
        metadata = {}

        for tr1, tr2 in sfile.get_tracer_combinations():
            dtype = self._get_dtype_for_trs(tr1, tr2)
            ell, cl, cov, ind = sfile.get_ell_cl(dtype, tr1, tr2,
                                                      return_cov=True,
                                                      return_ind=True)
            bpw = sfile.get_bandpower_windows(ind)
            metadata[(tr1, tr2)] = {'ell_eff': ell, 'cl': cl,
                                    'cov': cov, 'ind': ind,
                                    'ell_bpw': bpw.values,
                                    'w_bpw': bpw.weight.T}
 
        return metadata



    def dndz(self, trname, sacc_tr, **pars):
        pars_prefix = '_'.join([self.input_params_prefix, trname])

        z = sacc_tr.z
        nz = sacc_tr.nz
        z_m = np.average(z, weights=nz )  
        f = interp1d(z, nz, kind='cubic', fill_value=0.0, bounds_error=False)
        #Apply shift or shift and stretch or none redshift errors
        if (self.pz_errors == 'shift_stretch'):             
            sig_z = pars.get(pars_prefix + '_sz')
            dz = pars.get(pars_prefix +'_dz')
            z_shif_stch = z_m + (z-z_m-dz)*sig_z
            msk = (z_shif_stch > 0) 
            zz_total = np.sort(z_shif_stch[msk])
            nz_new = f(zz_total)
            z_new = z[msk]
            return (z_new, nz_new )

        elif (self.pz_errors == 'shift'):
            dz = pars.get(pars_prefix +'_dz')
            z_shift = (z-dz) 
            msk = (z_shift > 0)  
            zz_total = np.sort(z_shift[msk])
            nz_new = f(zz_total)
            return (z[msk], nz_new )
        else:
            return (sacc_tr.z,sacc_tr.nz)


  
    

    def galaxy_ccl_tracer(self, cosmo, trname, sacc_tr,  **pars):
        pars_prefix = '_'.join([self.input_params_prefix, trname])

        # Shift or stretch the redshift distributions:
        zz, nz = self.dndz(trname, sacc_tr, **pars)
 
        # Galaxy bias
        b = pars.get(pars_prefix + '_b')
        bz = b * np.ones_like(zz)
 
        # Magnification bias
        if self.magnification == 'True':  
            s = pars.get(pars_prefix + '_mag') 
            mag_bias = (zz, s * np.ones_like(zz))
            return ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(zz, nz), bias=(zz, bz), mag_bias=mag_bias)
        else:
            return ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(zz, nz),
                            bias=(zz, bz))


    def kCMB_ccl_tracer(self, cosmo, trname, sacc_tr,  **pars):
        pars_prefix = '_'.join([self.input_params_prefix, trname, 'ACTcv'])
        return ccl.CMBLensingTracer(cosmo, z_source=1100) #kappa cmb lensing tracer

 


    def get_tracers(self, **pars):
        res = self.provider.get_result('CCL')
        cosmo = res['cosmo']
        #print(cosmo ,'-COSMO----')
        ccl_tracers = {}

        # Get Tracers
        for trname, sacc_tr in self.sobj.tracers.items():
            if sacc_tr.quantity == 'galaxy_density':
                tr = self.galaxy_ccl_tracer(cosmo, trname, sacc_tr, **pars)
            elif sacc_tr.quantity == 'cmb_convergence':
                tr = self.kCMB_ccl_tracer(cosmo, trname, sacc_tr, **pars)
            else:
                raise ValueError(f'Tracer type {sacc_tr.quantity} not implemented')
            ccl_tracers[trname] = tr

        return ccl_tracers
 

    def cl_theory(self, **pars):
        res = self.provider.get_result('CCL')
        cosmo = res['cosmo']
        # pk = res['pk']
        ccl_tracers = self.get_tracers(**pars)
        cl_th = np.array([])
        for tr1, tr2 in self.sobj.get_tracer_combinations():
            ell_bpw = self.data_metadata[(tr1, tr2)]['ell_bpw']
            w_bpw = self.data_metadata[(tr1, tr2)]['w_bpw']
             
            cl_trs_unb = ccl.angular_cl(cosmo, ccl_tracers[tr1], ccl_tracers[tr2], ell_bpw) 
            cl_trs = np.dot(w_bpw, cl_trs_unb) #bin theory cls using the coupling matrix 

            cl_th = np.concatenate([cl_th, cl_trs])
        cl_th = np.array(cl_th)
        return cl_th


 
    def get_requirements(self):
        return {'CCL': {'cosmo': None}}
 

    def logp(self, **pars):
        """
        Gaussian likelihood.
        """
        t = self.cl_theory(**pars)
        r = (t - self.data_array)
        chi2 = np.dot(r, self.icov.dot(r))

        return -0.5*chi2



 

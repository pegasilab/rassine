.. table:: Table of the parameters that can be controlled in the
**Rassine_config.py** file.

   ====================
   =============================================================================================================================================================================================================================================================================================================
   ========= =================
   Parameters           Description                                                                                                                                                                                                                                                                                                   Automatic Value
   ====================
   =============================================================================================================================================================================================================================================================================================================
   ========= =================
   spectrum_name        Name of the spectrum file to reduce. The file can be either a pickle, csv, or txt. For pickle and csv the default colunm name must be ’wave’ and ’flux’ but this name can be changed by the column_wave and column_flux parameters. For fits file, use the Rassine_trigger.py file with preprocessing button. No        string
   output_dir           Output directory where RASSINE products are saved. Note that is if the spectrum name is entered in sys mode, the ourput file is by default at the same location than the input file except if an output directory is also specified in sys mode.                                                              Yes       string
   synthetic_spectrum   To allow the reduction of synthetic spectrum.                                                                                                                                                                                                                                                                 No        True/False
   anchor_file          Name of the RASSINE output file that can be used to fix the parameters value. Anchor file will bypass the parameters value entered in sys mode and from the config file.                                                                                                                                      No        string
   column_wave          Name of the column containing the wavelength grid.                                                                                                                                                                                                                                                            No        string
   column_flux          Name of the column containing the flux values.                                                                                                                                                                                                                                                                No        string
   float_precision      Float precision of the wavelength grid.                                                                                                                                                                                                                                                                       No        string
   par_stretching       Shrinking of the flux axis compared to the wavelength axis. The format of the automatic mode is **’auto_x’** with x a 1 decimal positive float number. x = 0.0 means high tension, whereas x = 1.0 mean low tension. You can also enter a float value by yourself (usually between 2 and 30).                 Yes       string or float
   par_vicinity         Size of the window in wavelength indices used to define a local maxima.                                                                                                                                                                                                                                       No        integer
   par_smoothing_box    Size of the window in wavelength indiced used to smooth the spectrum. Put **’auto’** to use the Fourier filtering.                                                                                                                                                                                            Yes       string or integer
   par_smoothing_kernel To use the automatic mode which apply a Fourier filtering use **’erf’ or ’hat_exp’** kernel and ’auto’ in par_smoothing_box. Else, use ’rectangular’, ’gaussian’, ’savgol’. Developers advise the **’savgol’** kernel except if the user is dealing with spectra spanning low and high SNR range.             Yes       string
   par_fwhm             FWHM of the CCF of the spectrum. The user can let **’auto’** to let RASSINE determine this value by itself.                                                                                                                                                                                                   Yes       string or float
   CCF_mask             CCF mask used to determine the FWHM. RASSINE construct its own mask by default. The user can specify its own mask which should be placed in the CCF_MASK directory.                                                                                                                                           Yes       string
   RV_sys               RV systemic of the star in km/s used to shift the CCF mask. Since RASSINE construct directly the mask with the spectrum, the default value is 0.                                                                                                                                                              Yes       float
   mask_telluric        A list of borders region to exclude of the CCF. By default the region where determine for spectrograph in the visible.                                                                                                                                                                                        No        list of list
   par_R                Minimum radius of the alpha shape scaled to the bluest part of the spectrum in :math:`\AA`. Put **’auto’** to let RASSINE fix the value.                                                                                                                                                                      Yes       string or float
   par_Rmax             Maximum radius of the alpha shape scaled to the bluest part of the spectrum in :math:`\AA`. Put **’auto’** to let RASSINE fix the value.                                                                                                                                                                      Yes       string or float
   ====================
   =============================================================================================================================================================================================================================================================================================================
   ========= =================

.. table:: Table of the parameters that can be controlled in the
**Rassine_config.py** file.

   ===========================
   ===========================================================================================================================================================================================
   ========= ==========
   Parameters                  Description                                                                                                                                                                                 Automatic Value
   ===========================
   ===========================================================================================================================================================================================
   ========= ==========
   par_reg_nu                  Penalty law of the alpha shape. Enter **’poly_nu’** with nu a positive 1 decimal float number for the polynomial law, or **’sigmoid_nu_mu’** with nu, mu a positive 1 decimal float number. No        string
   denoising_dist              Window in wavelength indices used to determine the anchor flux value by averaging around the local maximum. Only necessary for low SNR spectra.                                             No        integer
   count_cut_lim               Number of times borders continuum are flatten (2 or 3 give usually good values).                                                                                                            No        integer
   count_out_lim               Number of times outliers rejection algorithm is performed by derivative criterion. One iteration rejected the 0.5% of the anchor points with highest derivative.                            No        integer
   interpol                    ’linear’ or ’cubic’, the shape of the interpolation used in the graphical interface.                                                                                                        No        string
   feedback                    Trigger the interaction with the Sphinx ans the graphical feedback interface.                                                                                                               No        True/False
   only_print_end              Suppress the informations printed except the last line when RASSINE has finished.                                                                                                           No        True/False
   plot_end                    Display the last plot                                                                                                                                                                       No        True/False
   save_last_plot              Save the last plot                                                                                                                                                                          No        True/False
   outputs_interpolation_saved Either ’linear’, ’cubic’ or ’all’ to save some specific continuum.                                                                                                                          No        string
   outputs_denoising_saved     Either ’undenoised’,’denoised’,’all’ to save some specific continuum                                                                                                                        No        string
   light_version               Only save the primary output to produce a lighter output file.                                                                                                                              No        True/False
   ===========================
   ===========================================================================================================================================================================================
   ========= ==========

.. table:: Table of the parameters that can be controlled in the
**Rassine_trigger.py** file.

   =======================
   =====================================================================================================================================================================================================================================================================
   ==========
   Parameters              Description                                                                                                                                                                                                                                                           Value
   =======================
   =====================================================================================================================================================================================================================================================================
   ==========
   instrument              The instrument from which spectra are taken. Either ’HARPS’,’HARPN’,’CORALIE’ or ’ESPRESSO’ for the moment. Will be used during the preprocessing to format the fits file.                                                                                            string
   dir_spec_timeseries     Directory path of the spectra timeseries                                                                                                                                                                                                                              string
   nthreads_preprocess     Number of multiprocessed in preprocessing                                                                                                                                                                                                                             integer
   nthreads_matching       Number of multiprocessed in matching                                                                                                                                                                                                                                  integer
   nthreads_rassine        Number of multiprocessed in rassine                                                                                                                                                                                                                                   integer
   nthreads_intersect      Number of multiprocessed in rassine post reduction                                                                                                                                                                                                                    integer
   rv_timeseries           give the systemic RV of the star                                                                                                                                                                                                                                      float
   dlambda                 Value of the dlambda grid. If all the spectra come from the same instrument and s1d are already on a equidistant grid, RASSINE will determine automatically the dlambda. In the opposite case fix by yourself the dlambda step in :math:`\AA` of the wavelength grid. float
   bin_length_stack        Length in days of the window used to stack spectra (nightly stacking = 1). Spectra are stacked based on their jdb value obtained during the preprocessing.                                                                                                            float
   dbin                    Offset in days for the binning of the stacking (0.5 for daily stack, 0 for nightly stack).                                                                                                                                                                           
   use_master_as_reference To duplicate the master file at the intersect_all_continuum stage                                                                                                                                                                                                     True/False
   full_auto               To completely disable the Sphinx feedback                                                                                                                                                                                                                             True/False
   =======================
   =====================================================================================================================================================================================================================================================================
   ==========

.. table:: Table of the parameters that can be controlled in the
**Rassine_functions.py** file.

   ===============
   =============================================================================================================================================================
   ========= ===============
   Parameters      Description                                                                                                                                                   Automatic Value
   ===============
   =============================================================================================================================================================
   ========= ===============
   protocol_pickle Fix the protocol version of the pickle output file. By default in ’auto’ the protocol is the same than the python version on which you are launching RASSINE. Yes       string of float
   name_voice      Choose the gender voice of RASSINE. Victoria or Daniel are available.                                                                                         No        int
   ===============
   =============================================================================================================================================================
   ========= ===============

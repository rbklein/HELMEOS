# HELMEOS
An entropy stable and kinetic energy conserving solver for the compressible Euler equations using a real gas equations of state based on the Helmholtz free energy.

We implement a generalized Chandrashekar flux:
	f_rho 	= rho_tilde * u_mean 
	f_m 	= f_rho * u_mean + p_tilde
	f_E 	= f_rho * (e_tilde + p_tilde / rho_tilde + (u_mean^2 - 1/2 (u^2)_mean))
	
rho_tilde is a special of average of rho:
	
	rho_tilde = d_rho (p/T) / d_rho (g/T) 

using discrete gradient functions where g is the Gibbs free energy. p_tilde is a calculate using:

	p_tilde = (p/T)_mean / (1/T)_mean

and e_tilde is a special average of the internal energy:
	
	e_tilde = d_(1/T) (g/T) - d_(1/T) (p/T) / rho_tilde

also computed using discrete gradients.


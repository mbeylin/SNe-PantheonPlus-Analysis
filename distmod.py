import sys
import numpy as np
from scipy import integrate,optimize


class flrw:

    def __init__(self, om0, oml, H0=66.7):
        self.om0 = om0
        self.oml = oml
        self.c = 2.9979e5
        self.dH = self.c/H0

    @property
    def omk(self):
        return 1.-self.om0-self.oml

    def _integrand(self, zcmb):
        """
        Computes H0/H(z) where zcmb is redshift measured in cmb frame

        """
        z1 = 1.+zcmb
        return 1/np.sqrt(self.om0*z1**3 + self.omk*z1**2 + self.oml)

    def dL(self, zcmb):
        """
        Computes dL/dH, dH=c/H0 measured in cmb frame

        """
        Ok = self.omk
        Om = self.om0
        Ol = self.oml
        dH = self.dH
        z = zcmb
        z1 = 1.+z
        if Ok < 0: # closed
            I,err = integrate.quad(self._integrand, 0.0, z)
            q = np.sqrt(np.absolute(Ok))
            return z1*np.sin(q*I)/q
        elif Ok > 0: # open
            if Ok == 1: # Milne
                return 0.5*(z1**2-1.) # z1*np.sinh(np.log(z1))
            else:
                I,err = integrate.quad(self._integrand, 0.0, z)
                q = np.sqrt(Ok)
                return z1*np.sinh(q*I)/q
        else: # flat
            if Om == 1: # Einstein-de Sitter
                return 2.*z1*(1.-1./np.sqrt(z1))
            elif Ol == 1: # de Sitter
                return z1*z
            else:
                I,err = integrate.quad(self._integrand, 0.0, z)
                return z1*I

    def mu(self, zcmb):
        return 5*np.log10(self.dL(zcmb)) + 25


class timescape:

    def __init__(self, om0, fv0=0.778, H0=66.7):
        if om0 is not None:
            if 0 < om0 < 1:
                self.om0 = om0
                self.fv0 = 0.5*(np.sqrt(9-8.*self.om0)-1)
            else:
                sys.exit('om0 must be between 0 and 1')
        else:
            self.fv0 = fv0
            self.om0 = 0.5*(1.-self.fv0)*(2.+self.fv0)

        self.c = 2.9979e5
        self.dH = self.c/H0
        self.t0 = (2.+self.fv0)/3 # age of universe

        self._y0 = self.t0**(1./3)
        self._hf = (4.*self.fv0**2+self.fv0+4.)/2/(2.+self.fv0) # H0/barH0
        self._b = 2*(1.-self.fv0)*(2.+self.fv0)/9/self.fv0 # b*barH0

    def _z1(self, t):
        fv = self.fv_t(t)
        return (2.+fv)*fv**(1./3)/3/t/self.fv0**(1./3)

    def tex(self, zcmb):
        """
        Get time t explicitly by inverting func(t,z)=0 for zcmb
        where zcmb is redshift measured in cmb frame
        Note t units: 1/H0

        """
        def f(t,z): return self._z1(t)-(1.+z)
        # Root must be enclosed in [a,b]
        a = 0.01
        b = 1.1 #t=0.93/Hbar0 (for fv0=0.778)
        try:
            root = optimize.brentq(f, a, b, args=(zcmb,), maxiter=400)
        except ValueError:
            sys.exit('z = {0:1.3f}\nf(a) = {1:1.3f}\nf(b) = {2:1.3f}'.format(zcmb, f(a,zcmb), f(b,zcmb)))
        return root

    def _yint(self, Y):
        """
        Compute \mathcal{F}(Y), Y=t^{1/3}

        """
        bb = self._b**(1./3)
        return 2.*Y + (bb/6.)*np.log((Y+bb)**2/(Y**2-bb*Y+bb**2)) \
            + bb/np.sqrt(3)*np.arctan((2.*Y-bb)/(np.sqrt(3)*bb))

    def fv_t(self, t):
        """
        Tracker soln as fn of time

        """
        return 3.*self.fv0*t/(3.*self.fv0*t+(1.-self.fv0)*(2.+self.fv0))

    def fv_z(self, zcmb):
        """
        Tracker soln as fn of redshift

        """
        t = self.tex(zcmb)
        return 3.*self.fv0*t/(3.*self.fv0*t+(1.-self.fv0)*(2.+self.fv0))

    def dA(self, zcmb):
        """
        Angular diameter distance divided by dH=c/H0

        """
        ya = self.tex(zcmb)**(1./3) #t^{1/3}
        return ya**2 * (self._yint(self._y0)-self._yint(ya))

    def H0D(self, zcmb): #H0D/dH
        return self._hf*(1.+zcmb)*self.dA(zcmb)

    def dL(self, zcmb):
        """
        Luminosity distance, units Mpc

        """
        return self.dH*(1.+zcmb)*self.H0D(zcmb)

    def mu(self, zcmb):
        """
        Distance modulus

        """
        return 5*np.log10(self.dL(zcmb)) + 25

class lta:
    """
    Local Time Acceleration (LTA) model class.
    
    This model hypothesizes that dark energy effects are actually caused by 
    a local acceleration of time due to negentropic effects of life.
    """
    
    def __init__(self, om0, sf=0.981, ld=0.0883, qd=-0.0521, H0=66.7):
        """
        Initialize LTA model with full parameter set.
        """
        self.om0 = om0
        self.sf = sf
        self.ld = ld
        self.qd = qd
        self.c = 2.9979e5  # Speed of light in km/s
        self.dH = self.c/H0
        
    def tau(self, t):
        """
        Time acceleration function: τ(t) = SF × (1 + LD × ln(1+t)^QD)
        """
        return self.sf * (1.0 + self.ld * (np.log(1.0 + t))**self.qd)
        
    def _integrand(self, zcmb):
        """
        Computes modified H0/H(z) including LTA effect
        """
        z1 = 1.0 + zcmb
        # Convert redshift to approximate cosmic time (normalized)
        t_approx = 1.0 / np.sqrt(z1)
        
        # Apply time acceleration modification
        tau_mod = self.tau(t_approx)
        
        # Modified expansion rate with LTA effect
        return tau_mod / np.sqrt(self.om0 * z1**3 + (1.0 - self.om0))
    
    def dL(self, zcmb):
        """
        Computes luminosity distance in LTA model.
        """
        I, err = integrate.quad(self._integrand, 0.0, zcmb)
        t_approx = 1.0 / np.sqrt(1.0 + zcmb)
        return (1.0 + zcmb) * I * self.dH * self.tau(t_approx)
    
    def mu(self, zcmb):
        """
        Distance modulus in LTA model.
        """
        return 5.0 * np.log10(self.dL(zcmb)) + 25.0
    
def rundistmod(prefix):
    input = np.loadtxt(prefix+'input.txt')
    zcmb = input[:,0]
    zhel = input[:,-3]
    N = zcmb.size #740

    Oms = np.linspace(0.0, 1.0, 101)
    Ols = np.linspace(0.0, 1.0, 101)

    interp = np.empty((N, 101, 101))
    iarr = np.empty((0,3), dtype=int)

    # compute lcdm dL/dH
    for i, Om in enumerate(Oms):
        for j, Ol in enumerate(Ols):
            for k in range(N):
                fl = flrw(Om,Ol)
                zz = zcmb[k]
                Ok = 1.-Om-Ol
                if np.isfinite(fl._integrand(zz)):
                    interp[k,i,j] = fl.dL(zz)
                else:
                    iarr = np.vstack((iarr, [k,i,j]))
                    interp[k,i,j] = 10000.

    np.save(prefix+'tabledL_lcdm.npy',interp)


    # compute dL timescape normalized to h=61.7
    interp = np.empty((N, 100))
    iarr = np.empty((0,2), dtype=int)
    Oms = np.linspace(0.0, 0.99, 100)
    Oms[0] = 0.001 # avoid singular values
    for i, Om in enumerate(Oms):
        for k in range(N):
            ts = timescape(Om,H0=61.7)
            zz = zcmb[k]
            interp[k,i] = ts.dL(zz)

    np.save(prefix+'tabledL_ts.npy',interp)

    # compute dL for LTA model
    interp = np.empty((N, 100, 100, 100))
    Oms = np.linspace(0.0, 0.99, 100)
    Oms[0] = 0.001  # avoid singular values
    
    # Default values
    default_sf = 0.981
    default_ld = 0.0883
    default_qd = -0.0521
    
    # To reduce computational complexity, create a 3D table varying om0 and the most 
    # sensitive parameters (determine which through preliminary testing)
    SFs = np.linspace(0.979, 0.983, 100)  # SF range
    
    # First create a table with varying om0 and sf (fixing other parameters)
    interp_lta = np.empty((N, 100, 100))
    for i, Om in enumerate(Oms):
        for j, SF in enumerate(SFs):
            for k in range(N):
                lta_model = lta(Om, sf=SF, ld=default_ld, qd=default_qd)
                zz = zcmb[k]
                try:
                    interp_lta[k,i,j] = lta_model.dL(zz)
                except:
                    interp_lta[k,i,j] = 10000.
    
    np.save(prefix+'tabledL_lta.npy', interp_lta)
    
    
if __name__ == '__main__':

    name = sys.argv[1] + '_'
    prefix = 'Pantheon/Build/PP_' + name
    rundistmod(prefix)
        

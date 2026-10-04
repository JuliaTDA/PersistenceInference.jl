"""Independent published-formula checks, using NumPy/SciPy as numeric reference.

Gretton et al. (2008): HSIC = tr(K H L H)/n² (biased V-statistic).
Chazal et al. (2015): centered Gaussian multiplier supremum process.
Phipson & Smyth (2010): Monte Carlo +1 p-values, conservative ties.
Random draws are exchanged as CSV; Julia and NumPy RNGs need not coincide.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.stats import rankdata

root = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/"reference"
read = lambda name: np.loadtxt(root/(name+".csv"), delimiter=",")
errors = []
def check(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    errors.append(np.max(np.abs(np.asarray(actual)-expected)))

curves = read("curves"); n = len(curves)
mean = curves.mean(axis=0); centered = curves-mean
constant = curves.std(axis=0,ddof=1)==0
for kind in ("multiplier", "studentized", "bootstrap"):
    scale = curves.std(axis=0,ddof=1) if kind=="studentized" else (~constant).astype(float)
    normalizer = np.where(scale==0,1,scale)
    if kind=="bootstrap":
        indices = read("bootstrap_indices").astype(int)-1
        process = np.sqrt(n)*(curves[indices].mean(axis=0).T-mean[:,None])
    else:
        process = centered.T @ read("multipliers") / np.sqrt(n)
    maxima = np.max(np.abs(process/normalizer[:,None]),axis=0)
    critical = np.quantile(maxima,0.95,method="linear")
    width = critical*scale/np.sqrt(n)
    check(read(kind+"_maxima"),maxima)
    check(read(kind+"_band"),np.column_stack((mean,mean-width,mean+width)))

K,L = read("K"),read("L"); n = len(K); H = np.eye(n)-np.ones((n,n))/n
hsic = lambda k,l: np.trace(k@H@l@H)/n**2
observed = hsic(K,L)
permutations = read("permutations").astype(int)-1
nulls = np.array([hsic(K,L[np.ix_(p,p)]) for p in permutations.T])
check(read("hsic_nulls"),nulls)
check(read("hsic"),[observed,(1+np.count_nonzero(nulls>=observed))/(1+len(nulls))])
maxima = read("null_maxima"); observed = read("max_test")[0]
# min ranks on negated values count all greater/equal statistics conservatively.
rank = rankdata(-np.r_[observed,maxima],method="max")[0]
threshold = np.sort(maxima)[int(np.ceil(0.95*(len(maxima)+1)))-1]
check(read("max_test"),[observed,rank/(len(maxima)+1),threshold])
print(f"NumPy/SciPy reference passed: max error {max(errors):.3g}; tolerance 1e-10; seed 20261003")

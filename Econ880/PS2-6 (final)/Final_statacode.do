/////////////////////////////////////////////////////////
// Econ880b final take-home exam
// Author: Yuichiro Tsuji
// (For reduced-form heterogeneity test)
/////////////////////////////////////////////////////////
cd "/Users/yuichirotsuji/Documents/Econ880/PS2-6 (final)"
cap log close
log using "Final_statalog.log", replace

// Load data
use Mortgage_simdata_students, clear
format _all %9.0g
xtset id time

/////////////////////////////////////////////////////////
// Test for heterogenious refinancing cost
bysort id: egen duration = count(id)     // generate duration variable
gen r_diff = rate_t- rate			     // generate r-r_0 variable
bysort id: egen r_diff_min = min(r_diff) // calculate minimum r-r_0
*bysort id: egen refi_flag = total(refi)  // generate refinance indicator

duplicates drop id, force // eliminate duplication for estimation

reg duration r_diff_min loan_size,r

/////////////////////////////////////////////////////////
log close

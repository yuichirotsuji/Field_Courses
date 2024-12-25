*************************************************
* Econ 715-2 PS1 (Data cleaning and result confirmation)
* Author: Yuichiro Tsuji
*************************************************
cd "/Users/yuichirotsuji/Documents/Econometrics Data/cps09mar"
use "cps09mar.dta", clear

// create new variables
gen exp = age - 6 - education // experience
set seed 1234
gen u1 = uniform() - 0.5
gen u2 = uniform() - 0.5

gen ued = education + u1		// u-education
gen uexp = exp + u2				// u-experience
gen ln_earnings = log(earnings) // log(earnings) (dependent var)
gen constant = 1				// constant (for regression with Julia)

save "cps09mar_clean.dta", replace
/*
(Note: I'll load this dta file in Julia and do computations.
	   Please see "Bootstrap_functions.jl" and "Bootstrap_scripts.jl")
*/

*******************************************************
/*
(Code for OLS result confirmation 
 (I do the same OLS calculation in Julia))
*/
*******************************************************
* Q1 (OLS)
reg ln_earnings ued uexp if _n <= 100
reg ln_earnings ued uexp if _n <= 100, r

******
* Q3
* (a) OLS (true value)
reg ln_earnings ued uexp

*************************************************


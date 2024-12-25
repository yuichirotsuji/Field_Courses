**************************************************************
* Main Stata code for Econ761 PS5
* Author: Yuichiro Tsuji
**************************************************************
*cd "/Users/yuichirotsuji/Documents/Econ761/PS5/6649_data and programs/data/1997"
*cap log close
*log using "PS5_log.log", replace
*timer clear 1
*timer on 1 // The entrire compilation takes about one sec
************************************************************************
use XMat.dta, clear // dta data created by "PS5_dataclean.do"

***********************************
// Probit (Q1)
***********************************
// KMart problem (Y_i = 1{Kmart entry})
qui probit K_entry W_entry, r
est store K1
qui probit K_entry W_entry ln_pop ln_sales pop_urban, r 
est store K2
qui probit K_entry W_entry ln_pop ln_sales pop_urban midwest southern, r 
est store K3
qui probit K_entry W_entry ln_pop ln_sales pop_urban midwest southern num_small, r 
est store K4
qui probit K_entry W_entry ln_pop ln_sales pop_urban midwest southern num_small K_out W_out, r 
est store K5

estout K1 K2 K3 K4 K5, c(b se(par)) stats(aic bic) // (K3) is best

// WalMart problem (Y_i = 1{Kmart entry})
qui probit W_entry K_entry, r
est store W1
qui probit W_entry K_entry ln_pop ln_sales pop_urban, r 
est store W2
qui probit W_entry K_entry ln_pop ln_sales pop_urban midwest southern, r 
est store W3
qui probit W_entry K_entry ln_pop ln_sales pop_urban midwest southern num_small, r 
est store W4
qui probit W_entry K_entry ln_pop ln_sales pop_urban midwest southern num_small K_out W_out, r 
est store W5

estout W1 W2 W3 W4 W5, c(b se(par)) stats(aic bic) // (W5) is best


***********************************
// IV Probit (Q2)
***********************************
// probit with IV
ivprobit K_entry (W_entry = ln_dBenton) ln_pop ln_sales pop_urban midwest southern, vce(robust)
est store IV
estout IV, c(b se(par))

***********************************
// Ordered Probit (Q3)
***********************************
// create new variables 
cap gen num_large = K_entry + W_entry	//number of large players (i)
cap gen num_all = num_large + num_small //total number of players (ii)

// (i) Y = number of large players
oprobit num_large ln_pop ln_sales pop_urban midwest southern, r

// (ii) Y = number of all players
oprobit num_all ln_pop ln_sales pop_urban midwest southern, r


***********************************
// Two-step estimation of static games (Q4)
***********************************
cap drop _est*
// First step: Estimation of choice probability
logit K_entry W_entry ln_pop ln_sales pop_urban midwest southern, r 
cap predict sigma_K, pr			// predicted entry probability of Kmart
cap gen sigma_Ko = 1 - sigma_K	// predicted non-entry probability of Kmart

logit W_entry K_entry ln_pop ln_sales pop_urban midwest southern, r 
cap predict sigma_W, pr			// predicted entry probability of Wal-mart
cap gen sigma_Wo = 1 - sigma_W	// predicted non-entry probability of Walmart

// Second Step: Inversion
gen Pi_K = log(sigma_K) - log(sigma_Ko)
gen Pi_W = log(sigma_W) - log(sigma_Wo)

// Third step: Structural parameters estimation
reg Pi_K ln_pop ln_sales pop_urban midwest southern sigma_W, r
est store SK

reg Pi_W ln_pop ln_sales pop_urban midwest southern sigma_K, r
est store SW

estout SK SW, c(b)

************************************************************************
*timer off 1
*timer list 1
*cap log close


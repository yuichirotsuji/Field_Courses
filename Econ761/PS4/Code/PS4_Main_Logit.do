*================================================================
// Econ761 Problem set #4 Main stata code
// Author: Yuichiro Tsuji
*================================================================
*cd "/Users/yuichirotsuji/Documents/Econ761/PS4/Data"
cap log close
log using "PS4_Logit_log.log", replace
timer clear 1
timer on 1 // The entrire compilation takes about one min

*================================================================
* 1. Estimating (mixed) logit models
*================================================================
************************************
// Global variables definition
global J = 24 // number of brands in each market
global T = 94 // number of markets
set seed 1234
************************************

************************************
*(1) Estimating logit model
************************************
//import data from Excel file
import excel "cereal_ps3.xls",firstrow clear

// calculate outside share variable
bysort city year quarter: egen share_total = total(share) //get total share of each market
gen share_o = 1 - share_total //share of outside option, S_0 
order share_total share_o, after(share)

// create variables for logit model estimation
gen log_share_ratio = log(share/share_o) //dependent variable: log(S_j/S_0)
egen market = group(year quarter city) //generate market number for iteration
xtset firmbr market

//(i) OLS without brand fixed effect
qui reg log_share_ratio price, noconstant r
*reg log_share_ratio price, r
est store OLS
//(ii) OLS with brand fixed effect
qui xtreg log_share_ratio price, fe r
est store OLS_FE
//(iii)(1) IV without brand fixed effect
qui ivregress 2sls log_share_ratio (price = z1-z20), noconstant r
est store IV
//(iii)(2) IV with brand fixed effect
qui xtivreg log_share_ratio (price = z1-z20), fe vce(r)
est store IV_FE

// Report results with (used in Q1(1))
etable, col(estimates) estimates(OLS OLS_FE IV IV_FE) keep(price)

************************************
*(2) MC recovery
************************************
qui ivreg log_share_ratio (price = z1-z20), noconstant r // I used IV without FE model
predict delta, xb // fitted value (i.e. model-predicted share using estimates)
scalar a = _b[price]

gen share_numer = exp(delta) // e^(delta)
bysort city year quarter: egen ed_total = total(share_numer) // sum of exp(delta) for each market
gen share_hat = share_numer/(1+ed_total) // predicted share

// create Ometa^pre and recover MC
gen eps_own = a*share_hat*(1-share_hat) //own price elasticity is market-brand specific

***** Recover MC (until line.100) *****
// For t=1 (just to create an initial MC vector: Stata can't have an empty vector.)
mkmat eps_own if market == 1, mat(E) // own-price elasticity of market t (J*1 vector)
mkmat price if market == 1, mat(P) // price vector of market t
mkmat share_hat if market == 1, mat(S) // share vector of market t
	
matrix Omega_pre = J($J, $J, 0) // J*J matrix for Omega (in market t)
forvalues j = 1(1)$J {
	mat Omega_pre[`j',`j'] = -E[`j', 1] // (j,j) element is own-price elasticity
	
	// cross price elasticity enters if j and k are produced by the same firm
	forvalues k = 1(1)$J{
		if firm[`j'] == firm[`k'] & `j' != `k'{
			mat Omega_pre[`j',`k'] = a*S[`j',1]*S[`k',1]
		}	
	}
}
// MC of brands in market 1 (we'll add MCs in other markets vertically) 
cap mat drop MC // initialize the MC vector
mat MC = P - (invsym(Omega_pre)*S) //

// same for other markets (t=2,3,...,94)
local J = 24 // number of brands in the market
forvalues t = 2(1)$T{
	// get price and share vectors from data 
	mkmat eps_own if market == `t', mat(E) // own-price elasticity of market t (J*1 vector)
	mkmat price if market == `t', mat(P) // price vector of market t
	mkmat share_hat if market == `t', mat(S) // share vector of market t
	
	matrix Omega_pre = J($J, $J, 0) // J*J matrix for Omega (in market t)
	forvalues j = 1(1)$J{
		mat Omega_pre[`j',`j'] = -E[`j', 1] // (j,j) element is own-price elasticity
	
		// cross price elasticity enters if j and k are produced by the same firm
		forvalues k = 1(1)`J'{
			if firm[`j'] == firm[`k'] & `j' != `k'{
				mat Omega_pre[`j',`k'] = a*S[`j',1]*S[`k',1]
			}	
		}
	}
	mat MC = MC\(P - (invsym(Omega_pre)*S)) //add MC of market t(=2,3,..,94)
}
***** Recover MC ends*****

// calculate markups and PCM, and the statistics
svmat MC, name(MC) // give MC to dta dataframe
gen Markups = price - MC1 // calculate markups
gen PCM = Markups/price // calculate MPC

tabstat Markups PCM MC1, stat(mean median sd) // Mean, median and SD (used in Q1(2))
*bysort firmbr: tabstat Markups PCM MC1, stat(mean median sd)

************************************
*(3) Merger simulation
************************************
// Case I: Post(3)-Nabisco(6) merger
gen firm_PNmerge = firm // new ownership for Omega^post
replace firm_PNmerge = 3 if firm_PNmerge == 6 // Now Post(3) and Nabisco(6) are under the same owner

mkmat delta if market == 1, mat(D) // mean utility of market 1 (J*1 vector)
mkmat eps_own if market == 1, mat(E) // own-price elasticity of market 1 (J*1 vector)
mkmat price if market == 1, mat(P) // price vector of market 1
mkmat share_hat if market == 1, mat(S) // share vector of market 1
mkmat MC1 if market == 1, mat(MC) // MC estimates of market 1 

// Iteration for finding post-merger price
/* (Note: Stata can't handle mata in the while-loop. So instead I did
	20 times loop for each market. I confirmed that the distance between
	P and P_prime are very small after 20 iterations in all of the markets.)*/
mkmat price if market == 1, mat(P_prime) // initial P_prime (=P) in market 1
forvalues l=1(1)20  {
	mat P = P_prime //update next guess of post-merger price
	mat D = a * P //update mean utility

	mata {
		delta = st_matrix("D")
		share_numel = exp(delta)
		share_denom = 1 + colsum(share_numel)
		share = share_numel/share_denom
		st_matrix("Share", share)
	}
	
	mat S = Share //update model-predicted shares
	mat E = (-1)*(a)*hadamard(S,(J($J, 1, 1)-S)) //update own price elasticity
	mat Omega_post = J($J, $J, 0) // Initialize J*J matrix for Omega (in market 1)

	// update Omega^post
	forvalues j = 1(1)$J{
		mat Omega_post[`j',`j'] = E[`j', 1] // (j,j) element is own-price elasticity
		// cross price elasticity enters if j and k are produced by the same firm
		forvalues k = 1(1)$J{
			if firm_PNmerge[`j'] == firm_PNmerge[`k'] & `j' != `k'{
				mat Omega_post[`j',`k'] = a*S[`j',1]*S[`k',1]
				}
			}
		}
	mat P_prime = MC + (invsym(Omega_post)*S)
	
	mata {
		P = st_matrix("P")
		P_prime = st_matrix("P_prime")
		nor = norm(P - P_prime)
		st_numscalar("distance", nor)
	}
	
	local dist = distance
	di `dist'
}

mat P_post_PN = P_prime // Vector for storing post-merger prices
mat Q_post_PN = S // Vector for storing post-merger quantities

// same for other markets (t=2,3,...,94)
forvalues t = 2(1)$T{
	// get price and share vectors from data 
		mkmat delta if market == `t', mat(D) // mean utility of market 1 (J*1 vector)
		mkmat eps_own if market == `t', mat(E) // own-price elasticity of market 1 (J*1 vector)
		mkmat price if market == `t', mat(P) // price vector of market 1
		mkmat share_hat if market == `t', mat(S) // share vector of market 1
		mkmat MC1 if market == `t', mat(MC) // MC estimates of market 1 

		mkmat price if market == `t', mat(P_prime) // initial P_prime (=P) in market 1
		forvalues i=1(1)20  {
		mat P = P_prime //update next guess of post-merger price
		mat D = a * P //update mean utility

		mata {
			delta = st_matrix("D")
			share_numel = exp(delta)
			share_denom = 1 + colsum(share_numel)
			share = share_numel/share_denom
			st_matrix("Share", share)
		}
	
		mat S = Share //update model-predicted shares
		mat E = (-1)*(a)*hadamard(S,(J($J, 1, 1)-S)) //update own price elasticity
		mat Omega_post = J($J, $J, 0) // Initialize J*J matrix for Omega (in market 1)

		// update Omega^post
		forvalues j = 1(1)$J{
			mat Omega_post[`j',`j'] = E[`j', 1] // (j,j) element is own-price elasticity
			// cross price elasticity enters if j and k are produced by the same firm
			forvalues k = 1(1)$J{
				if firm_PNmerge[`j'] == firm_PNmerge[`k'] & `j' != `k'{
					mat Omega_post[`j',`k'] = a*S[`j',1]*S[`k',1]
				}
			}
		}
		mat P_prime = MC + (invsym(Omega_post)*S)
	
		mata {
			P = st_matrix("P")
			P_prime = st_matrix("P_prime")
			nor = norm(P - P_prime)
			st_numscalar("distance", nor)
		}
	
	local dist = distance
	*di `dist'
	}
	mat P_post_PN = P_post_PN \ P_prime
	mat Q_post_PN = Q_post_PN \ S
}

// give back P^post and Q^post to dta dataframe
svmat P_post_PN, name(price_PN)
svmat Q_post_PN, name(share_PN)

// get statistics by brands
*tabstat price price_PN1, stat(mean median SD)
*bysort firmbr: tabstat price price_PN1, stat(mean median SD)
gen diff_p_PN = price_PN1 - price
bysort firmbr: tabstat diff_p_PN, stat(mean median SD)

gen Q_diff_PN = share_PN1 - share
bysort firmbr: tabstat Q_diff_PN, stat(mean median SD)

************************************************************************
************************************************************************
// Case II: GM(2)-Quaker(4) merger
gen firm_GQmerge = firm // new ownership for Omega^post
replace firm_GQmerge = 2 if firm_GQmerge == 4 // Now GM(2) and Quaker(4) are under the same owner

mkmat delta if market == 1, mat(D) // mean utility of market 1 (J*1 vector)
mkmat eps_own if market == 1, mat(E) // own-price elasticity of market 1 (J*1 vector)
mkmat price if market == 1, mat(P) // price vector of market 1
mkmat share_hat if market == 1, mat(S) // share vector of market 1
mkmat MC1 if market == 1, mat(MC) // MC estimates of market 1 

mkmat price if market == 1, mat(P_prime) // initial P_prime (=P) in market 1
forvalues i=1(1)20  {
	mat P = P_prime //update next guess of post-merger price
	mat D = a * P //update mean utility

	mata {
		delta = st_matrix("D")
		share_numel = exp(delta)
		share_denom = 1 + colsum(share_numel)
		share = share_numel/share_denom
		st_matrix("Share", share)
	}
	
	mat S = Share //update model-predicted shares
	mat E = (-1)*(a)*hadamard(S,(J($J, 1, 1)-S)) //update own price elasticity
	mat Omega_post = J($J, $J, 0) // Initialize J*J matrix for Omega (in market 1)

	// update Omega^post
	forvalues j = 1(1)$J{
		mat Omega_post[`j',`j'] = E[`j', 1] // (j,j) element is own-price elasticity
		// cross price elasticity enters if j and k are produced by the same firm
		forvalues k = 1(1)$J{
			if firm_GQmerge[`j'] == firm_GQmerge[`k'] & `j' != `k'{
				mat Omega_post[`j',`k'] = a*S[`j',1]*S[`k',1]
				}
			}
		}
	mat P_prime = MC + (invsym(Omega_post)*S)
	
	mata {
		P = st_matrix("P")
		P_prime = st_matrix("P_prime")
		nor = norm(P - P_prime)
		st_numscalar("distance", nor)
	}
	
	local dist = distance
	di `dist'
}

mat P_post_GQ = P_prime // Vector for storing post-merger prices
mat Q_post_GQ = S // Vector for storing post-merger quantities

// same for other markets (t=2,3,...,94)
forvalues t = 2(1)$T{
	// get price and share vectors from data 
	mkmat delta if market == `t', mat(D) // mean utility of market 1 (J*1 vector)
	mkmat eps_own if market == `t', mat(E) // own-price elasticity of market 1 (J*1 vector)
	mkmat price if market == `t', mat(P) // price vector of market 1
	mkmat share_hat if market == `t', mat(S) // share vector of market 1
	mkmat MC1 if market == `t', mat(MC) // MC estimates of market 1 

	mkmat price if market == `t', mat(P_prime) // initial P_prime (=P) in market 1
	forvalues i=1(1)20  {
	mat P = P_prime //update next guess of post-merger price
	mat D = a * P //update mean utility

	mata {
		delta = st_matrix("D")
		share_numel = exp(delta)
		share_denom = 1 + colsum(share_numel)
		share = share_numel/share_denom
		st_matrix("Share", share)
	}
	
	mat S = Share //update model-predicted shares
	mat E = (-1)*(a)*hadamard(S,(J($J, 1, 1)-S)) //update own price elasticity
	mat Omega_post = J($J, $J, 0) // Initialize J*J matrix for Omega (in market 1)

	// update Omega^post
	forvalues j = 1(1)$J{
		mat Omega_post[`j',`j'] = E[`j', 1] // (j,j) element is own-price elasticity
		// cross price elasticity enters if j and k are produced by the same firm
		forvalues k = 1(1)$J{
			if firm_GQmerge[`j'] == firm_GQmerge[`k'] & `j' != `k'{
				mat Omega_post[`j',`k'] = a*S[`j',1]*S[`k',1]
			}
		}
	}
	mat P_prime = MC + (invsym(Omega_post)*S)
	
	mata {
		P = st_matrix("P")
		P_prime = st_matrix("P_prime")
		nor = norm(P - P_prime)
		st_numscalar("distance", nor)
	}
	
	local dist = distance
	*di `dist'
	}
	mat P_post_GQ = P_post_GQ \ P_prime
	mat Q_post_GQ = Q_post_GQ \ S
}

// give back P^post and Q^post to dta dataframe
svmat P_post_GQ, name(price_GQ) 
svmat Q_post_GQ, name(share_GQ) 

*tabstat price price_GQ1, stat(mean median SD)
*bysort firmbr: tabstat price price_PN1, stat(mean median SD)
gen diff_p_GQ = price_GQ1 - price
bysort firmbr: tabstat diff_p_GQ, stat(mean median SD)

gen Q_diff_GQ = share_GQ1 - share
bysort firmbr: tabstat Q_diff_GQ, stat(mean median SD)

	
************************************************************************
timer off 1
timer list 1
cap log close

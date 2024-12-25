*================================================================
// Econ761 Problem set #4 Main stata code for simulation with BLP results
// Author: Yuichiro Tsuji
*================================================================
//cd "/Users/yuichirotsuji/Documents/Econ761/PS4/Data"
cap log close
log using "PS4_PyBLP_log.log", replace
timer clear 1
timer on 1 // The entrire compilation takes about *** 15 min ***

************************************************************************
************************************
// Global variables definition
global J = 24 // number of brands in each market
global T = 94 // number of markets
set seed 1234
************************************************************************
*(2) MC recovery with mixed logit demand & PyBLP results
************************************
// import data again and generate basic variables
import excel "cereal_ps3.xls",firstrow clear
bysort year quarter city: egen share_total = total(share) //get total share of each market
gen share_o = 1 - share_total //share of outside option, S_0 
order share_total share_o, after(share)

// merge with demographic and v's (created by PS4_datacleaning.do)
merge m:1 year quarter city using demogr_and_v // all observations are matched
drop cyq _merge

************************************
// calculate choice probability and market shares
egen market = group(year quarter city) //generate market number for iteration
order market, after(quarter)
gen constant = 1, before(price) // constant var (just for calculation)

// use coeffcient estimates obtained by PyBLP
mat theta_1 = (-1.5869 \ -62.729 \ 0.1441 \ 0.7725)
mat theta_2 = (0.5581, 3.3124, 0, 1.1859, 0 \ /*
			 */ 3.3125, 588.325, -30.192, 0, 11.055\ /*
			 */ 0.0058, -0.3850, 0, 0.0522, 0\ /*
			 */ 0.0934, 0.748, 0, -1.3534, 0) 

mkmat constant price sugar mushy , mat(x_jt) // get "X2" variables from data
mat delta_jt = x_jt *  theta_1 // market-brand specific mean utility
*mat list delta_jt // (2256*1) vector
svmat delta_jt, name(delta) // give delta to dta dataframe

***** Compute \mu_ijt for 20 people in each market (until line.213)*****
// create \mu_1jt (first person in each market)
mkmat inc1 inc_sq1 age1 child1, mat(D) // demographic variables
mkmat vo_1 vp_1 vs_1 vm_1, mat(V) // demographic random draws
mat mu_ijt = hadamard(x_jt, V)*theta_2[1..4,1] /*
			*/+ hadamard(x_jt, D*(theta_2[1..4,2..5])')*J(4,1,1)
mat alpha_i = /* // price sensitive parameter (need for calculating ds/dp)
		*/ J(2256,1,theta_1[2,1]) + V[1..2256,2]*theta_2[2,1] + D*(theta_2[2,2..5])'

// repeat for 20 consumers in the market
forvalues i = 2(1)20{
	mkmat inc`i' inc_sq`i' age`i' child`i', mat(D) // demographic variables
	mkmat vo_`i' vp_`i' vs_`i' vm_`i', mat(V) // demographic random draws
	mat mu_ijt = mu_ijt, hadamard(x_jt, V)*theta_2[1..4,1] /*
			*/+ hadamard(x_jt, D*(theta_2[1..4,2..5])')*J(4,1,1)
	mat alpha_i = /* // price sensitive parameter (need for calculating ds/dp)
		*/ alpha_i, J(2256,1,theta_1[2,1]) + V[1..2256,2]*theta_2[2,1] + D*(theta_2[2,2..5])'
}
***** \mu_ijt cumputation complete *****
svmat mu_ijt, name(mu) // give \mu_ijt to dta dataframe
svmat alpha_i, name(alph) // give \alpha_i to dta dataframe

// create individual choice prob (as new vars) for 20 people in each market
forvalues i = 1(1)20{
	gen s_numel_`i' = exp(delta1 + mu`i')
	bysort market: egen s_denom_`i' = total(s_numel_`i')
	gen s_jt_`i' = s_numel_`i'/(1+s_denom_`i')
	drop s_numel_`i'
	drop s_denom_`i'
}

// derive market shares (Equation(11) of Nevo(2000a))
egen s_jt = rowmean(s_jt_1 - s_jt_20) 


************************************
// create Ometa^pre and recover MC
forvalues i = 1(1)20{
	gen eps_own_`i' = alph`i' * s_jt_`i'*(1-s_jt_`i')
} 
egen eps_own = rowmean(eps_own_1 - eps_own_20) // average of 20 individual \epsilon

***** Recover MC (until line.xxx) *****
// For t=1 (just to create an initial MC vector: Stata can't have an empty vector.)
mkmat eps_own if market == 1, mat(E) // own-price elasticity of market 1 (J*1 vector)
mkmat price if market == 1, mat(P) // price vector of market 1
mkmat s_jt if market == 1, mat(S) // share vector of market 1
mkmat s_jt_1 - s_jt_20 if market == 1, mat(Si) // choice prob matrix of 20 people in market 1
mkmat alph1 - alph20 if market == 1, mat(Ai) // price sensitivity parameter of 20 people in market 1
	
matrix Omega_pre = J($J, $J, 0) // J*J matrix for Omega (in market 1)
forvalues j = 1(1)$J{
	mat Omega_pre[`j',`j'] = -E[`j', 1] // (j,j) element is own-price elasticity
	
	// cross price elasticity enters if j and k are produced by the same firm
	forvalues k = 1(1)$J{
	if firm[`j'] == firm[`k'] & `j' != `k'{
		local s = 0
		forvalues i =1(1)20{
			local s = `s' + (Ai[1, `i']*Si[`j',`i']*Si[`k',`i'])
		}
		mat Omega_pre[`j',`k'] = `s'/20 // average of 20 people's individual elasticities
		}
	}
}

// MC of brands in market 1 (we'll add MCs in other markets vertically) 
cap mat drop MC // initialize the MC vector
mat MC = P - (invsym(Omega_pre)*S) //

// same for other markets (t=2,3,...,94)
forvalues t = 2(1)94{
	// get price and share vectors from data 
	mkmat eps_own if market == `t', mat(E) // own-price elasticity of market t (J*1 vector)
	mkmat price if market == `t', mat(P) // price vector of market t
	mkmat s_jt if market == `t', mat(S) // share vector of market t
	mkmat s_jt_1 - s_jt_20 if market == `t', mat(Si) // choice prob matrix of 20 people in market 1
	mkmat alph1 - alph20 if market == `t', mat(Ai) // price sensitivity parameter of 20 people in market 1
	
	matrix Omega_pre = J($J, $J, 0) // J*J matrix for Omega (in market t)
	forvalues j = 1(1)$J{
		mat Omega_pre[`j',`j'] = -E[`j', 1] // (j,j) element is own-price elasticity
	
		// cross price elasticity enters if j and k are produced by the same firm
		forvalues k = 1(1)$J{
			if firm[`j'] == firm[`k'] & `j' != `k'{
				local s = 0
				forvalues i =1(1)20{
					local s = `s' + (Ai[1, `i']*Si[`j',`i']*Si[`k',`i'])
				}
				mat Omega_pre[`j',`k'] = `s'/20 // average of 20 people's individual elasticities
			}
		}
	}	
	mat MC = MC\(P - (invsym(Omega_pre)*S)) //add MC of market t(=2,3,..,94)
}
***** Recover MC ends*****
svmat MC, name(MC) // give MC to dta dataframe
gen Markups = price - MC1 // calculate markups
gen PCM = Markups/price // calculate MPC

tabstat Markups PCM MC1, stat(mean median sd) // Mean, median and SD (used in Q1(6))

************************************************************************
************************************
* Merger simulation with PyBLP results
************************************
************************************
timer clear 2
timer on 2
// Case I: Post(3)-Nabisco(6) merger
cap gen firm_PNmerge = firm // new ownership for Omega^post
replace firm_PNmerge = 3 if firm_PNmerge == 6 // Now Post(3) and Nabisco(6) are under the same owner

mat C = J($J, 1, 1) //constants

mkmat price if market == 1, mat(P) // price vector of market 1
mkmat sugar mushy if market == 1, mat(X) // exogenous variables of market 1
mkmat delta1 if market == 1, mat(Delta) // mean utility of market 1
mkmat s_jt if market == 1, mat(S) // share vector of market 1

// individual parts (make individual-specific vector for calculation)
forvalues i = 1(1)20{
	mkmat eps_own_`i' if market == 1, mat(E`i') // individual own-price elasticity of market 1
	mkmat s_jt_`i' if market == 1, mat(S`i') // choice prob matrix of 20 people in market 1
	mkmat alph`i' if market == 1, mat(A`i') // price sensitivity parameter of 20 people in market 1
}

mkmat eps_own if market == 1, mat(E) // own-price elasticity of market 1 (J*1 vector)
mkmat s_jt_1 - s_jt_20 if market == 1, mat(Si) // choice prob matrix of 20 people in market 1
mkmat alph1 - alph20 if market == 1, mat(Ai) // price sensitivity parameter of 20 people in market 1
mkmat MC1 if market == 1, mat(MC) // MC estimates of market 1 

matrix Omega_post = J($J, $J, 0) // J*J matrix for Omega (in market 1)
forvalues j = 1(1)$J{
	mat Omega_post[`j',`j'] = -E[`j', 1] // (j,j) element is own-price elasticity
	
	// cross price elasticity enters if j and k are produced by the same firm
	forvalues k = 1(1)$J{
	if firm_PNmerge[`j'] == firm_PNmerge[`k'] & `j' != `k'{
		local s = 0
		forvalues i =1(1)20{
			local s = `s' + (Ai[1, `i']*Si[`j',`i']*Si[`k',`i'])
		}
		mat Omega_post[`j',`k'] = `s'/20 // average of 20 people's individual elasticities
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
//di `dist'


// get individual-specific demographic vars and v-shock values
forvalues i = 1(1)20{
	mkmat inc`i' inc_sq`i' age`i' child`i' if market == 1, mat(D`i') // demographic variables
	mkmat vo_`i' vp_`i' vs_`i' vm_`i' if market == 1, mat(V`i') // demographic random draws
}

forvalues l = 1(1)20 {
	mat P = P_prime // update prices
	mat Delta = (C,P,X)*theta_1 // update mean utility
	mat S = J($J, 1, 0)

	forvalues i = 1(1)20{
		mat M`i' = hadamard((C,P,X), V`i')*theta_2[1..4,1] /*
			*/+ hadamard((C,P,X), D`i'*(theta_2[1..4,2..5])')*J(4,1,1) //update individual shock
		mat A`i' = /* // update price sensitive parameter
		*/ J(24, 1, theta_1[2,1]) + V`i'[1..24,2]*theta_2[2,1] + D`i'*(theta_2[2,2..5])'	
	
		mata {
			delta = st_matrix("Delta")
			mu = st_matrix("M`i'")
			share_numel = exp(delta+mu)
			share_denom = 1 + colsum(share_numel)
			share = share_numel/share_denom
			st_matrix("Share`i'", share)
			}
		mat S`i' = Share`i'
		mat S = S + S`i'
	}
	mat S = S/20

	mat Si = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, /*
		*/ S11, S12, S13, S14, S15, S16, S17, S18, S19, S20]
	mat Ai = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, /*
		*/ A11, A12, A13, A14, A15, A16, A17, A18, A19, A20]
		
	mat Omega_post = J($J, $J, 0) // J*J matrix for Omega (in market 1)		
	forvalues j = 1(1)$J {
		// cross price elasticity enters if j and k are produced by the same firm
		forvalues k = 1(1)$J {
			if `j' == `k'{
				local eps = 0
				forvalues i =1(1)20{
					local eps = `eps' + (Ai[1, `i']*Si[`j',`i']*(1-Si[`k',`i']))
				}
				mat Omega_post[`j',`k'] = (-1)*`eps'/20 // average of 20 people's individual elasticities
			}		
			else if firm_PNmerge[`j'] == firm_PNmerge[`k'] & `j' != `k'{
				local s = 0
				forvalues i =1(1)20{
					local s = `s' + (Ai[1, `i']*Si[`j',`i']*Si[`k',`i'])					
				}
				mat Omega_post[`j',`k'] = `s'/20 // average of 20 people's individual elasticities
			}
		}
	}
	mat P_prime = MC + (invsym(Omega_post)*S) // new guess of post-merger price
	
	mata {
		P = st_matrix("P")
		P_prime = st_matrix("P_prime")
		nor = norm(P - P_prime)
		st_numscalar("distance", nor)
	}
	
	local dist = distance
	//di `dist'
}

mat P_post_PN = P_prime // Vector for storing post-merger prices
mat Q_post_PN = S // Vector for storing post-merger quantities

*****
// same for other markets (t=2,3,...,94)
forvalues t = 2(1)$T {
	mkmat price if market == `t', mat(P) // price vector of market 1
	mkmat sugar mushy if market == `t', mat(X) // exogenous variables of market 1
	mkmat delta1 if market == `t', mat(Delta) // mean utility of market 1
	mkmat s_jt if market == `t', mat(S) // share vector of market 1

	// individual parts (make individual-specific vector for calculation)
	forvalues i = 1(1)20{
		mkmat eps_own_`i' if market == `t', mat(E`i') // individual own-price elasticity of market 1
		mkmat s_jt_`i' if market == `t', mat(S`i') // choice prob matrix of 20 people in market 1
		mkmat alph`i' if market == `t', mat(A`i') // price sensitivity parameter of 20 people in market 1
	}

	mkmat eps_own if market == `t', mat(E) // own-price elasticity of market 1 (J*1 vector)
	mkmat s_jt_1 - s_jt_20 if market == `t', mat(Si) // choice prob matrix of 20 people in market 1
	mkmat alph1 - alph20 if market == `t', mat(Ai) // price sensitivity parameter of 20 people in market 1
	mkmat MC1 if market == `t', mat(MC) // MC estimates of market 1 

	matrix Omega_post = J($J, $J, 0) // J*J matrix for Omega (in market 1)
	forvalues j = 1(1)$J{
		mat Omega_post[`j',`j'] = -E[`j', 1] // (j,j) element is own-price elasticity
	
		// cross price elasticity enters if j and k are produced by the same firm
		forvalues k = 1(1)$J{
			if firm_PNmerge[`j'] == firm_PNmerge[`k'] & `j' != `k'{
				local s = 0
				forvalues i =1(1)20{
					local s = `s' + (Ai[1, `i']*Si[`j',`i']*Si[`k',`i'])
				}
				mat Omega_post[`j',`k'] = `s'/20 // average of 20 people's individual elasticities
			}
		}
	}

	mata {
		P = st_matrix("P")
		P_prime = st_matrix("P_prime")
		nor = norm(P - P_prime)
		st_numscalar("distance", nor)
		}
	local dist = distance
	//di `dist'

	// get individual-specific demographic vars and v-shock values
	forvalues i = 1(1)20{
		mkmat inc`i' inc_sq`i' age`i' child`i' if market == 1, mat(D`i') // demographic variables
		mkmat vo_`i' vp_`i' vs_`i' vm_`i' if market == 1, mat(V`i') // demographic random draws
	}

	forvalues l = 1(1)20 {
		mat P = P_prime // update prices
		mat Delta = (C,P,X)*theta_1 // update mean utility
		mat S = J($J, 1, 0)
	
		forvalues i = 1(1)20{
			mat M`i' = hadamard((C,P,X), V`i')*theta_2[1..4,1] /*
				*/+ hadamard((C,P,X), D`i'*(theta_2[1..4,2..5])')*J(4,1,1) //update individual shock
			mat A`i' = /* // update price sensitive parameter
				*/ J(24, 1, theta_1[2,1]) + V`i'[1..24,2]*theta_2[2,1] + D`i'*(theta_2[2,2..5])'	
	
			mata {
				delta = st_matrix("Delta")
				mu = st_matrix("M`i'")
				share_numel = exp(delta+mu)
				share_denom = 1 + colsum(share_numel)
				share = share_numel/share_denom
				st_matrix("Share`i'", share)
			}
			mat S`i' = Share`i'
			mat S = S + S`i'
		}
		
		mat S = S/20
		mat Si = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, /*
		*/ S11, S12, S13, S14, S15, S16, S17, S18, S19, S20]
		mat Ai = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, /*
		*/ A11, A12, A13, A14, A15, A16, A17, A18, A19, A20]
		
		mat Omega_post = J($J, $J, 0) // J*J matrix for Omega (in market 1)		
		forvalues j = 1(1)$J {
			// cross price elasticity enters if j and k are produced by the same firm
			forvalues k = 1(1)$J {
				if `j' == `k'{
					local eps = 0
					forvalues i =1(1)20{
					local eps = `eps' + (Ai[1, `i']*Si[`j',`i']*(1-Si[`k',`i']))
					}
					mat Omega_post[`j',`k'] = (-1)*`eps'/20 // average of 20 people's individual elasticities
				}		
				else if firm_PNmerge[`j'] == firm_PNmerge[`k'] & `j' != `k'{
					local s = 0
					forvalues i =1(1)20{
						local s = `s' + (Ai[1, `i']*Si[`j',`i']*Si[`k',`i'])					
					}
					mat Omega_post[`j',`k'] = `s'/20 // average of 20 people's individual elasticities
				}
			}
		}
		mat P_prime = MC + (invsym(Omega_post)*S) // new guess of post-merger price
	
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

svmat P_post_PN, name(price_PN)
svmat Q_post_PN, name(share_PN)

*tabstat price price_PN1, stat(mean median SD)
*bysort firmbr: tabstat price price_PN1, stat(mean median SD)
gen diff_p_PN = price_PN1 - price
bysort firmbr: tabstat diff_p_PN, stat(mean median SD)

gen Q_diff_PN = share_PN1 - share
bysort firmbr: tabstat Q_diff_PN, stat(mean median SD)

timer off 2
timer list 2 
	
************************************************************************
************************************************************************
timer clear 3
timer on 3
// Case II: GM(2)-Quaker(4) merger
gen firm_GQmerge = firm // new ownership for Omega^post
replace firm_GQmerge = 2 if firm_GQmerge == 4 // Now GM(2) and Quaker(4) are under the same owner

mat C = J($J, 1, 1) //constants

mkmat price if market == 1, mat(P) // price vector of market 1
mkmat sugar mushy if market == 1, mat(X) // exogenous variables of market 1
mkmat delta1 if market == 1, mat(Delta) // mean utility of market 1
mkmat s_jt if market == 1, mat(S) // share vector of market 1

// individual parts (make individual-specific vector for calculation)
forvalues i = 1(1)20{
	mkmat eps_own_`i' if market == 1, mat(E`i') // individual own-price elasticity of market 1
	mkmat s_jt_`i' if market == 1, mat(S`i') // choice prob matrix of 20 people in market 1
	mkmat alph`i' if market == 1, mat(A`i') // price sensitivity parameter of 20 people in market 1
}

mkmat eps_own if market == 1, mat(E) // own-price elasticity of market 1 (J*1 vector)
mkmat s_jt_1 - s_jt_20 if market == 1, mat(Si) // choice prob matrix of 20 people in market 1
mkmat alph1 - alph20 if market == 1, mat(Ai) // price sensitivity parameter of 20 people in market 1
mkmat MC1 if market == 1, mat(MC) // MC estimates of market 1 

matrix Omega_post = J($J, $J, 0) // J*J matrix for Omega (in market 1)
forvalues j = 1(1)$J{
	mat Omega_post[`j',`j'] = -E[`j', 1] // (j,j) element is own-price elasticity
	
	// cross price elasticity enters if j and k are produced by the same firm
	forvalues k = 1(1)$J{
	if firm_GQmerge[`j'] == firm_GQmerge[`k'] & `j' != `k'{
		local s = 0
		forvalues i =1(1)20{
			local s = `s' + (Ai[1, `i']*Si[`j',`i']*Si[`k',`i'])
		}
		mat Omega_post[`j',`k'] = `s'/20 // average of 20 people's individual elasticities
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
//di `dist'
	
// get individual-specific demographic vars and v-shock values
forvalues i = 1(1)20{
	mkmat inc`i' inc_sq`i' age`i' child`i' if market == 1, mat(D`i') // demographic variables
	mkmat vo_`i' vp_`i' vs_`i' vm_`i' if market == 1, mat(V`i') // demographic random draws
}

forvalues l = 1(1)20 {
	mat P = P_prime // update prices
	mat Delta = (C,P,X)*theta_1 // update mean utility
	mat S = J($J, 1, 0)

	forvalues i = 1(1)20{
		mat M`i' = hadamard((C,P,X), V`i')*theta_2[1..4,1] /*
			*/+ hadamard((C,P,X), D`i'*(theta_2[1..4,2..5])')*J(4,1,1) //update individual shock
		mat A`i' = /* // update price sensitive parameter
		*/ J(24, 1, theta_1[2,1]) + V`i'[1..24,2]*theta_2[2,1] + D`i'*(theta_2[2,2..5])'	
	
		mata {
			delta = st_matrix("Delta")
			mu = st_matrix("M`i'")
			share_numel = exp(delta+mu)
			share_denom = 1 + colsum(share_numel)
			share = share_numel/share_denom
			st_matrix("Share`i'", share)
			}
		mat S`i' = Share`i'
		mat S = S + S`i'
	}
	mat S = S/20

	mat Si = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, /*
		*/ S11, S12, S13, S14, S15, S16, S17, S18, S19, S20]
	mat Ai = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, /*
		*/ A11, A12, A13, A14, A15, A16, A17, A18, A19, A20]
		
	mat Omega_post = J($J, $J, 0) // J*J matrix for Omega (in market 1)		
	forvalues j = 1(1)$J {
		// cross price elasticity enters if j and k are produced by the same firm
		forvalues k = 1(1)$J {
			if `j' == `k'{
				local eps = 0
				forvalues i =1(1)20{
					local eps = `eps' + (Ai[1, `i']*Si[`j',`i']*(1-Si[`k',`i']))
				}
				mat Omega_post[`j',`k'] = (-1)*`eps'/20 // average of 20 people's individual elasticities
			}		
			else if firm_GQmerge[`j'] == firm_GQmerge[`k'] & `j' != `k'{
				local s = 0
				forvalues i =1(1)20{
					local s = `s' + (Ai[1, `i']*Si[`j',`i']*Si[`k',`i'])					
				}
				mat Omega_post[`j',`k'] = `s'/20 // average of 20 people's individual elasticities
			}
		}
	}
	mat P_prime = MC + (invsym(Omega_post)*S) // new guess of post-merger price
	
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

*****
// same for other markets (t=2,3,...,94)
forvalues t = 2(1)$T {
	mkmat price if market == `t', mat(P) // price vector of market 1
	mkmat sugar mushy if market == `t', mat(X) // exogenous variables of market 1
	mkmat delta1 if market == `t', mat(Delta) // mean utility of market 1
	mkmat s_jt if market == `t', mat(S) // share vector of market 1

	// individual parts (make individual-specific vector for calculation)
	forvalues i = 1(1)20{
		mkmat eps_own_`i' if market == `t', mat(E`i') // individual own-price elasticity of market 1
		mkmat s_jt_`i' if market == `t', mat(S`i') // choice prob matrix of 20 people in market 1
		mkmat alph`i' if market == `t', mat(A`i') // price sensitivity parameter of 20 people in market 1
	}

	mkmat eps_own if market == `t', mat(E) // own-price elasticity of market 1 (J*1 vector)
	mkmat s_jt_1 - s_jt_20 if market == `t', mat(Si) // choice prob matrix of 20 people in market 1
	mkmat alph1 - alph20 if market == `t', mat(Ai) // price sensitivity parameter of 20 people in market 1
	mkmat MC1 if market == `t', mat(MC) // MC estimates of market 1 

	matrix Omega_post = J($J, $J, 0) // J*J matrix for Omega (in market 1)
	forvalues j = 1(1)$J{
		mat Omega_post[`j',`j'] = -E[`j', 1] // (j,j) element is own-price elasticity
	
		// cross price elasticity enters if j and k are produced by the same firm
		forvalues k = 1(1)$J{
			if firm_GQmerge[`j'] == firm_GQmerge[`k'] & `j' != `k'{
				local s = 0
				forvalues i =1(1)20{
					local s = `s' + (Ai[1, `i']*Si[`j',`i']*Si[`k',`i'])
				}
				mat Omega_post[`j',`k'] = `s'/20 // average of 20 people's individual elasticities
			}
		}
	}

	mata {
		P = st_matrix("P")
		P_prime = st_matrix("P_prime")
		nor = norm(P - P_prime)
		st_numscalar("distance", nor)
		}
	local dist = distance
	//di `dist'

	// get individual-specific demographic vars and v-shock values
	forvalues i = 1(1)20{
		mkmat inc`i' inc_sq`i' age`i' child`i' if market == 1, mat(D`i') // demographic variables
		mkmat vo_`i' vp_`i' vs_`i' vm_`i' if market == 1, mat(V`i') // demographic random draws
	}

	forvalues l = 1(1)20 {
		mat P = P_prime // update prices
		mat Delta = (C,P,X)*theta_1 // update mean utility
		mat S = J($J, 1, 0)
	
		forvalues i = 1(1)20{
			mat M`i' = hadamard((C,P,X), V`i')*theta_2[1..4,1] /*
				*/+ hadamard((C,P,X), D`i'*(theta_2[1..4,2..5])')*J(4,1,1) //update individual shock
			mat A`i' = /* // update price sensitive parameter
				*/ J(24, 1, theta_1[2,1]) + V`i'[1..24,2]*theta_2[2,1] + D`i'*(theta_2[2,2..5])'	
	
			mata {
				delta = st_matrix("Delta")
				mu = st_matrix("M`i'")
				share_numel = exp(delta+mu)
				share_denom = 1 + colsum(share_numel)
				share = share_numel/share_denom
				st_matrix("Share`i'", share)
			}
			mat S`i' = Share`i'
			mat S = S + S`i'
		}
		
		mat S = S/20
		mat Si = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, /*
		*/ S11, S12, S13, S14, S15, S16, S17, S18, S19, S20]
		mat Ai = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, /*
		*/ A11, A12, A13, A14, A15, A16, A17, A18, A19, A20]
		
		mat Omega_post = J($J, $J, 0) // J*J matrix for Omega (in market 1)		
		forvalues j = 1(1)$J {
			// cross price elasticity enters if j and k are produced by the same firm
			forvalues k = 1(1)$J {
				if `j' == `k'{
					local eps = 0
					forvalues i =1(1)20{
					local eps = `eps' + (Ai[1, `i']*Si[`j',`i']*(1-Si[`k',`i']))
					}
					mat Omega_post[`j',`k'] = (-1)*`eps'/20 // average of 20 people's individual elasticities
				}		
				else if firm_GQmerge[`j'] == firm_GQmerge[`k'] & `j' != `k'{
					local s = 0
					forvalues i =1(1)20{
						local s = `s' + (Ai[1, `i']*Si[`j',`i']*Si[`k',`i'])					
					}
					mat Omega_post[`j',`k'] = `s'/20 // average of 20 people's individual elasticities
				}
			}
		}
		mat P_prime = MC + (invsym(Omega_post)*S) // new guess of post-merger price
	
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

svmat P_post_GQ, name(price_GQ)
svmat Q_post_GQ, name(share_GQ)

*tabstat price price_GQ1, stat(mean median SD)
*bysort firmbr: tabstat price price_GQ1, stat(mean median SD)
gen diff_p_GQ = price_GQ1 - price
bysort firmbr: tabstat diff_p_GQ, stat(mean median SD)

gen Q_diff_GQ = share_GQ1 - share
bysort firmbr: tabstat Q_diff_GQ, stat(mean median SD)

timer off 3
timer list 3
	
************************************************************************

timer off 1
timer list 1
cap log close

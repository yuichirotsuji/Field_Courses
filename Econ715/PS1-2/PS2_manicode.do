**************************************************************
* Main Stata code for Econ715 PS2
* Author: Yuichiro Tsuji
**************************************************************
*cd "/Users/yuichirotsuji/Documents/Econ715/PS1-2/Dupas_data&dofile_MS9508"
*cap log close
*log using "PS2_log.log", replace
*timer clear 1
*timer on 1 // about 1 sec for the entire compilation

**************************************************************
* (0. Data cleaning)
**************************************************************
// use data from Dupas(2014)
use main_dataset, clear

// clean data (as directed in "Data" section in PS2)
keep if price == 0| price == 50 | price == 100 | price == 200
gen usednet = 0
replace usednet = 1 if fol1_inuse == 1

keep price purchasednet usednet		// keep necessary variables
order usednet purchasednet price	// re-order the variables (Y,D,Z)
sort price							// sort price (ascending order)

**************************************************************
* 1. Propensity score
**************************************************************
// sample propensity score 
cap bysort price: egen P = mean(purchasednet)

// display each propensity score
di as text "p(0)=" as result (P[1])		//p(0)=0.97435898
di as text "p(50)=" as result (P[118])	//p(50)=0.625
di as text "p(100)=" as result (P[234])	//p(100)=0.35964912
di as text "p(200)=" as result (P[344])	//p(200)=0.14772727

**************************************************************
* 2. LATE estimation
**************************************************************
// estimate LATE(0,50)
qui ivregress 2sls usednet (purchasednet = price) if price == 0 | price == 50,r
est store L1

// estimate LATE(50,100)
qui ivregress 2sls usednet (purchasednet = price) if price == 50 | price == 100,r 
est store L2

// estimate LATE(100,200)
qui ivregress 2sls usednet (purchasednet = price) if price == 100 | price == 200 ,r
est store L3

estout L1 L2 L3, c(b se) keep(purchasednet) // display the result table

/* // check the IV estimates
cap bysort price: egen Y = mean(usednet)
di as text "LATE(0,50)=" as result (Y[1]-Y[118])/(P[1]-P[118])        // LATE(0,50)=0.7147226
di as text "LATE(50,100)=" as result (Y[118]-Y[230])/(P[118]-P[230])  // LATE(50,100)=0.7508855
di as text "LATE(100,200)=" as result (Y[230]-Y[344])/(P[230]-P[344]) // LATE(100,200)=0.476952
*/

**************************************************************
* 3. MTE estimation
**************************************************************
// generate variables for MTE estimation
cap gen p1 = P/2
cap gen p2 = P^2/3
cap gen p3 = P^3/4

// linear regression
qui reg usednet p1 if purchasednet == 1, r		 // linear specification
est store MTE_linear

qui reg usednet p1 p2 p3 if purchasednet == 1, r // cubic specification
est store MTE_cubic

estout MTE_linear MTE_cubic, keep(_cons p1 p2 p3) // display the result table

// plot MTE function (Note: I use r(coefs) in estout results)
scalar ao = r(coefs)[4,1]	// store \alpha_0
scalar a = r(coefs)[1,1]	// store \alpha_1
scalar bo = r(coefs)[4,2]	// store \beta_0
scalar b = r(coefs)[1,2]	// store \beta_1
scalar bb = r(coefs)[2,2]	// store \beta_2
scalar bbb = r(coefs)[3,2]	// store \beta_3

cap gen MTE_lin = ao + a*(_n/100) if _n <= 100 // predicted liner MTE
cap gen MTE_cub = bo + b*(_n/100)  + bb*(_n/100)^2 + bbb*(_n/100)^3 /*
				*/if _n <= 100 // predicted cubic MTE
cap gen u = _n/100 if _n <= 100 // a variable for u in (0,1) to make plot

graph twoway (line MTE_lin u) (line MTE_cub u), title("Estimated MTE functions") ytitle("MTE(u)")
graph export "MTE.png", replace

**************************************************************
* 4. LATE via MTE
**************************************************************
// Linear specification
di as text "LATE(0,50) in linear specification is " as result ao+((a/2)*(P[1]+P[118]))
di as text "LATE(50,100) in linear specification is " as result ao+((a/2)*(P[118]+P[230]))
di as text "LATE(100,200) in linear specification is " as result ao+((a/2)*(P[230]+P[344]))

// Cubic specification
di as text "LATE(0,50) in cubic specification is " /*
	*/ as result bo+((b/2)*(P[1]+P[118]))+((bb/3)*(P[1]^3 - P[118]^3)+(bbb/4)*(P[1]^4 - P[118]^4))/(P[1]-P[118]) 
di as text "LATE(50,100) in cubic specification is " /*
	*/ as result bo+((b/2)*(P[118]+P[230]))+((bb/3)*(P[118]^3 - P[230]^3)+(bbb/4)*(P[118]^4 - P[230]^4))/(P[118]-P[230]) 
di as text "LATE(100,200) in cubic specification is " /*
	*/ as result bo+((b/2)*(P[230]+P[344]))+((bb/3)*(P[230]^3 - P[344]^3)+(bbb/4)*(P[230]^4 - P[344]^4))/(P[230]-P[344]) 

**************************************************************
* 5. Policy evaluation
**************************************************************
*(b) logit estimation
logit purchasednet price, r

**********
*(c) B(z) and C(z) plot
// get estimates (from postestimation command of logit())
scalar go = e(b)[1,2]
scalar g = e(b)[1,1]

// generate B(z) and C(z) for z in [0,150]
cap gen z = _n if _n <= 150 // a variable for z in [0,150] to make plot
cap gen p_star = exp(go+(g*z))/(1+exp(go+(g*z))) if _n <= 150 	  // predicted choice probabilities with logit
cap gen p_bar = exp(go+(g*150))/(1+exp(go+(g*150)))  if _n <= 150 // predicted choice probabilities at z=150

cap gen C = (150 - _n)*p_star if _n <= 150 // generate C(z)
replace C = C/150	// normalize for graph

/*
cap gen Y_new_lin = ao + (a/2)*(exp(go+(g*z))/(1+exp(go+(g*z))))+(exp(go+(g*150))/(1+exp(go+(g*150)))) if _n <= 150
cap gen Y_150_lin = ao + (a/2)*(exp(go+(g*150))/(1+exp(go+(g*150)))) /*
			*/ if _n <= 150
cap gen B_lin = Y_new_lin - Y_150_lin if _n <= 150
*/

cap gen B_lin = ao*(p_star - p_bar) + (a/2)*(p_star^2 - p_bar^2) if _n <= 150

/*
gen Y_new_cub = bo+((b/2)*((exp(go+(g*z))/(1+exp(go+(g*z)))))) /*
		*/ +((bb/3)*((exp(go+(g*z))/(1+exp(go+(g*z)))))^3) +((bbb/4)*((exp(go+(g*z))/(1+exp(go+(g*z)))))^4) /*
		*/ if _n <= 150
gen Y_150_cub = bo+((b/2)*((exp(go+(g*150))/(1+exp(go+(g*150)))))) /*
		*/ +((bb/150)*((exp(go+(g*150))/(1+exp(go+(g*150)))))^3) +((bbb/4)*((exp(go+(g*150))/(1+exp(go+(g*150)))))^4) /*
		*/ if _n <= 150
cap gen B_cub = Y_new_cub - Y_150_cub
*/

cap gen B_cub = bo*(p_star - p_bar) + (b/2)*(p_star^2 - p_bar^2) /*
			*/ + (bb/3)*(p_star^3 - p_bar^3) + (bbb/4)*(p_star^4 - p_bar^4) /*
			*/ if _n <= 150

graph twoway (line B_lin z) (line B_cub z) (line C z), title("B(z) and C(z) plots") ytitle(B(z) and C(Z)/150)
graph export "B_and_C.png", replace

************************************************************************
*cap timer off 1
*cap timer list 1
*cap log close

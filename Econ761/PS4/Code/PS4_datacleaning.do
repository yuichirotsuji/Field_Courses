*================================================================
// Econ761 Problem set #4 Data cleaning code
// Author: Yuichiro Tsuji
*================================================================
*****************************************************************
/* (Note: This code is basically for generating "demogr_and_v.dta",
		  which will be used in "PS4_Main_BLP" for simulation.
		  Text data are from MATLAB data.)　*/
*****************************************************************

*cd "/Users/yuichirotsuji/Documents/Econ761/PS4/Data"
*****************************************************************
* data cleaning
*****************************************************************
// make dta file of v (from demog.ps3)
import excel "demog_ps3.xls",firstrow clear
// rename demographic random draws(v_i)
forvalues i = 1(1)20{
	rename v`i' vo_`i'	
}
forvalues i = 21(1)40{
	local j = `i' - 20
	rename v`i' vp_`j'	
}
forvalues i = 41(1)60{
	local j = `i' - 40
	rename v`i' vs_`j'	
}
forvalues i = 61(1)80{
	local j = `i' - 60
	rename v`i' vm_`j'
}
save v, replace

// make demographic variable dataset using matlab files (!!NO Excel FILE!!)
// (Note: I used writematlix() function to generate text data from matlab)
import delim "demogr.txt", clear //demographic variables from matlab data
save demogr, replace

import delim "id_demo.txt", clear //demographic ids from matlab data
rename v1 cyq
merge 1:1 _n using demogr //merge with demographic variables
drop _merge

**
// rename demographic variables
forvalues i = 1(1)20{
	rename v`i' inc`i'	
}
forvalues i = 21(1)40{
	local j = `i' - 20
	rename v`i' inc_sq`j'	
}
forvalues i = 41(1)60{
	local j = `i' - 40
	rename v`i' age`j'	
}
forvalues i = 61(1)80{
	local j = `i' - 60
	rename v`i' child`j'
}

**
// generate city-year-quarter variables for merging with v
tostring cyq, replace
gen city = substr(cyq,1,2)
replace city = substr(cyq,1,1) if strlen(cyq) == 4
gen year = substr(cyq,-3,2)
gen quarter = substr(cyq,-1,1)
order city year quarter
destring city year quarter, replace

// merge demographic variables and random draw
merge 1:1 city year quarter using v
drop _merge

cap erase demogr.dta
cap erase v.dta
save demogr_and_v, replace // I use this in main code(PS4_Main.do)

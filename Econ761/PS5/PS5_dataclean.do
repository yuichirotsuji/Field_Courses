************************************************************************
* Econ761 PS5 Initial Data cleaning code (from data files of Jia (2008))
* Author: Yuichiro Tsuji
************************************************************************
cd "/Users/yuichirotsuji/Documents/Econ761/PS5/6649_data and programs/data/1997"

************************************************************************
// Clean XMat.out
import delimited "XMat.out", clear

rename v1 county	 // county id
rename v2 ln_pop	 // log of population
rename v3 ln_sales	 // log of sales per capita
rename v4 pop_urban	 // percentage of urban population
rename v5 midwest	 // dummy for the midwest region
rename v6 ln_dBenton // log of distance to Benton county
rename v7 southern	 // dummy for the southern region
rename v8 K_entry 	 // dummy for Kmart entry
rename v9 W_entry 	 // dummy for Wal-Mart entry
rename v10 num_small // number of small firms
rename v11 K_out	 // distance-weighted number of Kmart outside
rename v12 W_out	 // distance-weighted number of Wal-Mart outside
cap drop v16 		 // drop missing var

save "XMat.dta", replace // save dta file (used in "PS5.main.dta")

************************************************************************
// clean data78.out
*import delimited "data78.out", clear

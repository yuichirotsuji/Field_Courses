clear
set mem 200m
#delimit;
set more off;


*********************;
*FIGURES;
*********************;

**********************;
*** FIGURE 1;
*********************;
	#delimit;
	use main_dataset, clear;
	bys price: egen nobs=count(purchasednet);
	bys price: egen mean=mean(purchasednet);
	graph twoway scatter mean price [w=nobs], msymbol(Oh)
			||  qfit purchasednet price, xtitle("Phase 1 Price (in Ksh)", margin(small))
	legend(off)  nodraw ytitle("Share Purchased")
	title(Panel A) subtitle(Purchased Olyset net in Phase 1) name(demandcurve1a, replace);
	
	#delimit;
	use main_dataset, clear;
	bys price: egen nobs2=count(fol2_using_net);
	bys price: egen mean2=mean(fol2_using_net);
	bys price: egen nobs1=count(fol1_inuse);
	bys price: egen mean1=mean(fol1_inuse);
	graph twoway scatter mean2 price [w=nobs2], msymbol(O) mcolor(gs8) 
			|| lfit fol1_inuse  price, xtitle("Price of Phase 1 LLIN (in Ksh)", margin(small)) lpat(dash) lcolor(gs10)
			|| scatter mean1 price [w=nobs1], msymbol(Oh) mcolor(navy) 
			||  lfit fol2_using_net price, xtitle("Phase 1 price(in Ksh)", margin(small)) lcolor(gs8)
	legend(row(2) order(1 "Using net at 1-yr follow-up" 3 "Using net at 2-month follow-up" ))  nodraw ytitle("Share Using")
	
	title(Panel B) subtitle(Usage of Phase 1 Olyset net (if purchased)) name(demandcurve1b, replace);
	
	#delimit;
	use main_dataset, clear;
	gen adopted_1yr=purchasednet;
		replace adopted_1yr=0 if fol2_using==0;
		replace adopted_1yr=1 if adopted==1;
	bys price: egen nobs=count(adopted_1yr);
	bys price: egen mean=mean(adopted_1yr);
	label var nobs "Num of Obs Per Price";
	graph twoway scatter mean price [w=nobs], msymbol(Oh)
			||  qfit adopted_1yr price, xtitle("Phase 1 price (in Ksh)", margin(small))
	legend(off)  nodraw ytitle("Share Adopted")
	title(Panel C) subtitle(Adopted (=purchased & used) Olyset net in Phase 1) name(demandcurve1c, replace);
	
	graph combine demandcurve1a demandcurve1b demandcurve1c, ycommon col(3) saving(demand_fig1, replace) xsize(12) ysize(5);

**********************;
***FIGURE A1: TIME TO REDEEM BY PRICE GROUP;
*********************;
#delimit;

	gen str12 pricerange2="1- FREE" if price==0;
	replace pricerange2="2- 40-50" if price>0&price<60;
	replace pricerange2="3- 60-90" if price>50&price<100;
	replace pricerange2="4- 100-120" if price>90&price<130;
	replace pricerange2="5- 130-150" if price>120&price<160;
	replace pricerange2="6- 190-250" if price>150&price<310;

	gen free=0;
		replace free=1 if price==0;
		gen notfree=1-free;

for any coef upper lower : gen X=.;
gen n=_n;

	xi: reg timetoredeem1 free i.pricerange2, nocons;
		mat beta=e(b);
		mat var=e(V);
		for num 1/6:
		 replace coef=beta[1,X] if _n==X \
		 replace upper=coef+1.96*var[X, X]^.5 if _n==X \
		 replace lower=coef-1.96*var[X, X]^.5 if _n==X;	 
	graph twoway (scatter coef n if _n<=6, connect(l) xlabel(1 "FREE" 2 "40-50" 3 "60-90" 4 "100-120" 5 "130-150" 6 "190-250"
	 ))
	 (rcap lower upper n if _n<=6, msymbol(i)), title("") 
	 legend(order(1 "Average" 2 "95% CI") rows(1)) note(" ") xtitle("Price of Olyset net in Phase 1 (in Ksh)", margin(small))
	 saving(learningfiga1, replace);


*****************;
** FIGURE 2;
*****************;
	#delimit;
	use main_dataset, clear;
	drop if purchasednet2==.;
	gen got2=purchasednet*purchasednet2;
	
	for any purchasednet purchasednet2 got2 \ num 1/3: global varY="X";
	for num 1/3 \ any "Panel A" "Panel B" "Panel C" \ any "Purchased Olyset net in Phase 1" "Purchased Olyset net at 150 Ksh in Phase 2" "Purchased both Olysets" : 
		global titleX="Y" \
		global subtitleX="Z";
	
	local i 1;
	while `i'<=3 {;
		bys price: egen mean=mean(${var`i'});
		sum mean;
		bys price: egen nobs=count(${var`i'});
		graph twoway scatter mean price [w=nobs], msymbol(Oh)
				||  qfit mean price
				, xtitle("Price of Phase 1 LLIN (in Ksh)", margin(small)) title("${title`i'}") subtitle("${subtitle`i'}", size(medium)) 
				legend(off)  name(f_`i', replace) nodraw;
		drop mean nobs* ;
	local i = `i' + 1;
	};
	
	graph combine f_1 f_2 f_3, col(3) saving(demand_fig2, replace) xsize(12) ysize(5);



*********************;
*********************;
* TABLES;
*********************;
*********************;

global controls0="treatM treatBoth burdenpitch costspitch  ";
global controls1="bg_hhsize bg_distfromshop bg_age bg_children bg_female_head_primarycomplete wealthus_100 bg_bank_account  bg_ITNlastnight  bg_a30_free_bednet distfromshopmissing agemissing treatM treatBoth burdenpitch costspitch ";

*************************************;
* TABLE 1. BASELINE CHARACTERISTICS;
*************************************;

	#delimit;
	use main_dataset, clear;

	*************************************************************;
	* CREATE A BUNCH OF VARIABLES THAT WILL BE USEFUL THROUGHOUT;
	*************************************************************;

	gen usprice=price/65;
		gen usprice2=usprice*usprice;
	
	gen free=0;
		replace free=1 if price==0;
		gen notfree=1-free;

	gen highsubsidy=free;
		replace highsubsidy=1 if price<=50;
	
	gen purchasednet_sub=purchasednet if purchasednet2!=.;
	egen atleastone=rmax(purchasednet_sub purchasednet2);
	gen got2=purchasednet*purchasednet2;
	
	gen experienced=0 if purchasednet==0;
		replace experienced=0 if purchasednet==1&(fol1_hanging==0&fol2_net_hanging==0);
		replace experienced=0 if purchasednet==0|(purchasednet==1&(fol1_hanging==0&fol2_net_hanging==.));
		replace experienced=0 if purchasednet==0|(purchasednet==1&(fol1_hanging==.&fol2_net_hanging==0));
		replace experienced=1 if fol2_net_hanging==1|fol1_hanging==1;
		replace experienced=0 if purchasednet==1 & experienced==.;
	
	gen wealthus=bg_wealth/65;
		gen wealthus_100=wealthus/100;


	gen var="";
	for any sample_mean sample_sd coeff0 se0 coeff1 se1 coeff2 se2 N jointf: gen X=.;
	local vars=15;
	for any
		bg_hhsize  bg_age   bg_children   bg_female_head_primarycomplete 	bg_active wealthus 
		bg_a17_elec bg_bank_account bg_a29_net_number  bg_ITNlastnight fol1_preownLLIN 
		bg_a30_free_bednet bg_know_shop bg_WTP bg_distfromshop 
	\ num 1/`vars':
	 replace var="X" if _n==Y \
	xi: reg X free highsubsidy  i.cfw_id \
	 mat beta0=e(b) \
	 replace coeff0=beta0[1,1] if _n==Y \
	 mat var=e(V) \
	 replace se0=sqrt(var[1,1]) if _n==Y \
	xi: reg X free usprice usprice2  i.cfw_id \
	 replace N=e(N) if _n==Y \
	 mat beta=e(b) \
	 replace coeff1=beta[1,1] if _n==Y \
	 replace coeff2=beta[1,2] if _n==Y \
	 mat var=e(V) \
	 replace se1=sqrt(var[1,1]) if _n==Y \
	 replace se2=sqrt(var[2,2]) if _n==Y \
	 test free usprice usprice2  \	
	 replace jointf=r(p) if _n==Y \
	 sum X \
	 replace sample_mean=r(mean) if _n==Y \
	 replace sample_sd=r(sd) if _n==Y ;
	for var sample_mean-se2: replace X=round(X, 0.001);
	outsheet var sample_mean sample_sd coeff0 se0  coeff1 se1 coeff2 se2 jointf N if _n<=`vars' using table1.xls,replace;
	

foreach var in distfromshop age {;
	gen `var'missing=0;
	replace `var'missing=1 if `var'==.;
	sum `var';
	replace `var'=r(mean) if `var'missing==1;
}; 


foreach i in 500 250  {;
	foreach def in free cheap inuse 1week 2week 3week 1mo own experienced {;
		gen dens_`def'`i'=n_`def'`i'/n_total`i';
		replace dens_`def'`i'=0 if n_total`i'==0;
	};
};


sort hhid;
save modified_dataset, replace;


********************************;
****TABLE 2;
********************************;
#delimit;
use modified_dataset, clear;

drop if Long_home==. | Lat_home==.;

global controls1="bg_hhsize bg_distfromshop bg_age bg_children bg_female_head_primarycomplete wealthus_100 bg_bank_account  bg_ITNlastnight  bg_a30_free_bednet treatM treatBoth burdenpitch costspitch ";

for any dens_cheap cut1 cut2 : gen X=.;
gen cons=1;

foreach cont in 0 1 {;
	*make sure there are no obs with missing regressors for Conley covariance matrix;				
		if "`cont'"=="1" {;
			foreach var of varlist ${controls`cont'} {;
					drop if `var'==.;
					};
				}; 
	foreach spec in reg  {;
		foreach indepvar in "dens_cheap" {;
			foreach radius in 500 {;
				*cutoff points for kernel weights in Conley covariance matrix;
					replace cut1= `radius'*(2/1000)*(180/(3963*1.609344*_pi));
					replace cut2= `radius'*(2/1000)*(180/(3963*1.609344*_pi));
				foreach var in dens_cheap {;
					replace `var'=`var'`radius';
					};
					
					foreach outcome of varlist purchasednet experienced purchasednet2 got2 purchasedwg {;
						local append_replace="append";
							 if "`indepvar'"=="dens_cheap" & "`outcome'"=="purchasednet" {;		
							 local append_replace="replace";	
							};
						preserve;
						drop if `outcome'==.;
	
						*first get num of regressors;
						xi: `spec' `outcome' highsubsidy `indepvar' ${controls`cont'}  i.cfw_id;
							local num=e(df_m)+1;
						*then get conley se;
							xi: x_ols Lat_home Long_home cut1 cut2 `outcome' `indepvar' highsubsidy  ${controls`cont'}  i.cfw_id cons, coord(2) xreg(`num');
							
							gen Conley_se_sub=sqrt(cov_dep[2,2]);
							gen Conley_se_dens=sqrt(cov_dep[1,1]);
														
						sum `outcome';
							local meanvarall=r(mean);	
						sum `outcome' if highsubsidy==0;
							local meanvar=r(mean);	
						xi: `spec' `outcome' `indepvar' highsubsidy  ${controls`cont'} i.cfw_id;
							outreg highsubsidy `indepvar' using rev_spill_`spec'`cont'.xls, nor2   bdec(3) adec(3) 
							nonotes se sigsymb(***,**,*)  addstat("Mean no sub", `meanvar', "controls", `cont',"radius", `radius',"Conley Sub", Conley_se_sub,  "Conley Dens", Conley_se_dens, "Mean all", `meanvarall') `append_replace';
						drop Conley*;
						restore;
						di in blue "`indepvar', `radius': `outcome' done";
					};
				};
			};
		};		
	};



*set trace on
set more off;
********************************************;
********************************************;
********************************************;
* TABLE 3: STRUCTURAL ESTIMATION;
********************************************;
********************************************;
********************************************;
#delimit cr
* First, drop the program "MyEstimator", in case it's already in memory.
	capture program drop MyEstimator

*************************
* Now define the program.
*************************
	program MyEstimator
	args lfn mu   l1 l2 a refdep la fe1 fe8 fe24
	tempname 	tempL11 tempL1p1  tempL01 tempL10 tempL1p0 tempL00
	tempvar 	P11 P01 P1 P1p P01p refprice1  fe 
	quietly {


		gen `fe'                =`fe1'*cfw1 + `fe8'*cfw8 + `fe24'*cfw24
		gen double `refprice1' = price1

		gen double `P1'  	= 1 / (1 + exp(-`mu'-`fe'+`a'*price1))

		gen double `P1p'  	= [((1-p1)^c)*`P1'+(1-(1-p1)^c)*1/ (1 + exp(-`mu'-`l1'-`fe'+`a'*price1))]

		gen double `P01'	= 1 / (1 + exp(-privret*gamma*(`mu')-`fe'+`a'*price2+`refdep'*`la'*(price2-`refprice1'))) if (price2>=`refprice1')
		replace  `P01'		= 1 / (1 + exp(-privret*gamma*(`mu')-`fe'+`a'*price2+`refdep'*(price2-`refprice1'))) if (price2<`refprice1')

		gen double `P01p'	=((1-p2)^c)*`P01'+(1-(1-p2)^c)*1/ (1 + exp(-privret*gamma*(`mu'+`l1')-`fe'+`a'*price2+`refdep'*`la'*(price2-`refprice1'))) if (price2>=`refprice1')
		replace `P01p'		=((1-p2)^c)*`P01'+(1-(1-p2)^c)*1/ (1 + exp(-privret*gamma*(`mu'+`l1')-`fe'+`a'*price2+`refdep'*(price2-`refprice1'))) if (price2<`refprice1')

		gen double `P11'	= 1 / (1 + exp(-privret*gamma*(`mu'+`l1'+`l2')-`fe'+`a'*price2+`refdep'*`la'*(price2-`refprice1'))) if (price2>=`refprice1')
		replace `P11'		= 1 / (1 + exp(-privret*gamma*(`mu'+`l1'+`l2')-`fe'+`a'*price2+`refdep'*(price2-`refprice1'))) if (price2<`refprice1')
		
		
		*those who purchased in period 2 and in period 1
			gen double `tempL11' 	= .
			replace `tempL11' 	= purchasednet2*purchasednet*ln(`P11'*`P1p')

		*those who did not purchase in period 2 but purchased in period 1
			gen double `tempL10' 	= .
			replace `tempL10' 	= (1-purchasednet2)*purchasednet*ln((1-`P11')*`P1p')

		*those who purchased in period 2 but not in period 1
			gen double `tempL01' 	= .
			replace `tempL01' 	= purchasednet2*(1-purchasednet)*ln(`P01p'*(1-`P1p'))

		*those who did not purchase in either period
			gen double `tempL00' 	= .
			replace `tempL00' 	= (1-purchasednet2)*(1-purchasednet)*ln((1-`P01p')*(1-`P1p'))

		replace `lfn'		= `tempL11' +  `tempL10'+ `tempL01' + `tempL00'
	
	}
end


*************************************************
* Now prepare the data
*************************************************

use modified_dataset, clear
	gen price1=price
	gen price2=150

* Deal with missing obs
	keep if purchasednet!=. & purchasednet2!=. & price1!=.
	drop if Long_home==. | Lat_home==.
	
* village dummies
foreach num of numlist 1 8 24 25 {
	gen cfw`num'=cfw_id==`num'
	}

*number of social contacts
	gen c=min(4,n_total250)

* proba to learn from contact who got product
	* within phase 1 -- can only learn from those who buy within first month
	gen p1=n_1mo250/n_total250
		replace p1=0 if n_total250==0
	* by the time phase 2 comes around, had time to learn from anyone who acquired product in phase 1
	gen p2=n_own250/n_total250
		replace p2=0 if n_total250==0


save temp, replace

******************************************************
* And now, the estimation...
******************************************************

global if ""

local title ""
	**************************
	** MAIN SPEC: ASSUMPTIONS
	**************************
	
	* health spillover
		local threshold 0.6
		gen gamma=1 if dens_own500<`threshold' 
			replace gamma=0.8 if dens_own500>=`threshold'
	
	* diminishing returns
		gen privret=1 if purchasednet==0
			replace privret=0.85 if purchasednet==1


		
	******************************************************
	* VARY THE ASSUMPTION ON THE LOSS AVERSION PARAMETER
	******************************************************
		* matrix in which the results will be saved
		mat EST=J(11,13,0)
		
		local i 1
		foreach num of numlist 1 1.2 1.4 1.6 1.8 2 2.2 2.4 2.6  {
				constraint 1 [la]_cons =`num'
				ml model lf	MyEstimator /mu /l1 /l2  /a /refdep  /la /fe1 /fe8 /fe24  ${if}, tech(nr dfp bfgs bhhh) constraints(1) robust
					ml init /refdep =0.02
					ml search
					ml max
			mat C=e(b)
			mat V=e(V)
			local j=`i'+1	
			mat EST[`i', 1]=`num'
			mat EST[`i', 2]=C[1,1]
			mat EST[`i', 3]=C[1,2]	
			mat EST[`i', 4]=C[1,3]	
			mat EST[`i', 5]=C[1,4]		
			mat EST[`i', 6]=C[1,5]		
			
			mat EST[`i', 7]=sqrt(V[1,1])
			mat EST[`i', 8]=sqrt(V[2,2])
			mat EST[`i', 9]=sqrt(V[3,3])	
			mat EST[`i', 10]=sqrt(V[4,4])	
			mat EST[`i', 11]=sqrt(V[5,5])	
			
			lincom [l1]_cons + [l2]_cons
			mat EST[`i', 12]=r(estimate)
			mat EST[`i', 13]=r(se)
			local i=`i'+1	
			}
		

		svmat EST
		keep EST*
		drop if _n>9
		for any movingvar mu l1 l2 a refdep se_mu se_l1 se_l2 se_a se_refdep l se_l \ num 1/13: rename ESTY X
		foreach var of  varlist mu l1 l2 a refdep {
			gen up_`var'=`var'+se_`var'*1.64
			gen l_`var'=`var'-se_`var'*1.64
		}
		
		local xtitle1 ""Loss aversion""
		local xtitle2 ""parameter (lambda)""
		local mov "la"
		
		*mu	
		local var "mu"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs4) yscale(range(0 4)) ylab(0(1)4, nogrid) xlab(1 1.6 2 2.6)  lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle1' `xtitle2', margin(medium))  saving(`var'_`mov', replace)  graphregion(color(white) fcolor(white))
		
		local xtitle ""lambda""

		*l1	
		local var "l1"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-.5 1)) ylab(-.5(.5)1, nogrid) xlab(1 1.6 2 2.6)  lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace)  yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))
			
		*l2	
		local var "l2"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(0 1.5)) ylab(0(.5)1.5, nogrid) xlab(1 1.6 2 2.6)  lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace)  yline(0, lwi(medthick) lpat(dot) lcol(gs12)) graphregion(color(white) fcolor(white))
		
		*a	
		local var "a"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-0.01 0.03)) ylab(-0.01(0.01)0.03, nogrid) xlab(1 1.6 2 2.6)  lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace) yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))
			
		*refdep	
		local var "refdep"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-0.002 0.004)) ylab(-0.002(0.002)0.004, nogrid) xlab(1 1.6 2 2.6)  lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace) yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))
		
		outsheet using la_sensitivity.out, replace

	
	******************************************************
	* VARY THE ASSUMPTION ON DIMINISHING RETURNS
	******************************************************
		use temp, replace
		
		* health spillover
			local threshold 0.6
			gen gamma=1 if dens_own500<`threshold'
				replace gamma=0.8 if dens_own500>=`threshold'
		
		* diminishing returns
			gen privret=1 if purchasednet==0
			
			
		* matrix in which the results will be saved
		mat EST=J(11,13,0)
		
		local i 1
		foreach num of numlist 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85  0.9 0.95 1 {
				constraint 1 [la]_cons =1.6
				replace privret=`num' if purchasednet==1
				ml model lf	MyEstimator /mu /l1 /l2 /a /refdep  /la /fe1 /fe8 /fe24  ${if}, tech(nr dfp bfgs bhhh) constraints(1) robust
					ml init /refdep =0.02
					ml search
					ml max
			mat C=e(b)
			mat V=e(V)
			local j=`i'+1	
			mat EST[`i', 1]=`num'
			mat EST[`i', 2]=C[1,1]
			mat EST[`i', 3]=C[1,2]	
			mat EST[`i', 4]=C[1,3]	
			mat EST[`i', 5]=C[1,4]		
			mat EST[`i', 6]=C[1,5]		
			
			mat EST[`i', 7]=sqrt(V[1,1])
			mat EST[`i', 8]=sqrt(V[2,2])
			mat EST[`i', 9]=sqrt(V[3,3])	
			mat EST[`i', 10]=sqrt(V[4,4])	
			mat EST[`i', 11]=sqrt(V[5,5])	
			
			lincom [l1]_cons + [l2]_cons
			mat EST[`i', 12]=r(estimate)
			mat EST[`i', 13]=r(se)
			local i=`i'+1	
			}
		
		svmat EST
		keep EST*
		drop if _n>11
		for any movingvar mu l1 l2 a refdep se_mu se_l1 se_l2 se_a se_refdep l se_l \ num 1/13: rename ESTY X
		foreach var of  varlist mu l1 l2 a refdep {
			gen up_`var'=`var'+se_`var'*1.64
			gen l_`var'=`var'-se_`var'*1.64
		}
		
		local xtitle1 ""Relative return""
		local xtitle2 ""of 2nd LLIN (v2)""
		local mov "priv"
		
		*mu	
		local var "mu"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs4) yscale(range(0 4)) ylab(0(1)4, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle1' `xtitle2', margin(medium))  saving(`var'_`mov', replace)  graphregion(color(white) fcolor(white))
		
		local xtitle ""v2""
		*l1	
		local var "l1"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-.5 1)) ylab(-.5(.5)1, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace)  yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))
			
		*l2	
		local var "l2"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(0 4)) ylab(0(1)4, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace)  yline(0, lwi(medthick) lpat(dot) lcol(gs12)) graphregion(color(white) fcolor(white))
		
		*a	
		local var "a"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-0.01 0.03)) ylab(-0.01(0.01)0.03, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace) yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))
			
		*refdep	
		local var "refdep"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-0.002 0.004)) ylab(-0.002(0.002)0.004, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace) yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))
		
		
		
	******************************************************
	* VARY THE ASSUMPTION ON HEALTH SPILLOVERS
	******************************************************
		use temp, replace
		
		* health spillover
			local threshold 0.6
			gen gamma=1 if dens_own500<`threshold'
		
		* diminishing returns
			gen privret=1 if purchasednet==0
				replace privret=0.85 if purchasednet==1
			
		* matrix in which the results will be saved
		mat EST=J(11,13,0)
		
		local i 1
		foreach num of numlist 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1 {
				constraint 1 [la]_cons =1.6
				replace gamma=`num' if dens_own500>=`threshold'
				ml model lf	MyEstimator /mu /l1 /l2 /a /refdep  /la /fe1 /fe8 /fe24  ${if}, tech(nr dfp bfgs bhhh) constraints(1) robust
					ml init /refdep =0.02
					ml search
					ml max
			mat C=e(b)
			mat V=e(V)
			local j=`i'+1	
			mat EST[`i', 1]=`num'
			mat EST[`i', 2]=C[1,1]
			mat EST[`i', 3]=C[1,2]	
			mat EST[`i', 4]=C[1,3]	
			mat EST[`i', 5]=C[1,4]		
			mat EST[`i', 6]=C[1,5]		
			
			mat EST[`i', 7]=sqrt(V[1,1])
			mat EST[`i', 8]=sqrt(V[2,2])
			mat EST[`i', 9]=sqrt(V[3,3])	
			mat EST[`i', 10]=sqrt(V[4,4])	
			mat EST[`i', 11]=sqrt(V[5,5])	
			
			lincom [l1]_cons + [l2]_cons
			mat EST[`i', 12]=r(estimate)
			mat EST[`i', 13]=r(se)
			local i=`i'+1	
			}
		
		svmat EST
		keep EST*
		drop if _n>11
		for any movingvar mu l1 l2 a refdep se_mu se_l1 se_l2 se_a se_refdep l se_l \ num 1/13: rename ESTY X
		foreach var of  varlist mu l1 l2 a refdep {
			gen up_`var'=`var'+se_`var'*1.64
			gen l_`var'=`var'-se_`var'*1.64
		}
		
		local xtitle1 ""Relative disease risk""
		local xtitle2 ""with spillover (alpha)""
		local mov "spill"
		
		*mu	
		local var "mu"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs4) yscale(range(0 4)) ylab(0(1)4, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle1' `xtitle2', margin(medium))  saving(`var'_`mov', replace)  graphregion(color(white) fcolor(white))
		
		local xtitle ""alpha""
		*l1	
		local var "l1"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-.5 1)) ylab(-.5(.5)1, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace)  yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))

		*l2	
		local var "l2"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(0 1.5)) ylab(0(.5)1.5, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace)  yline(0, lwi(medthick) lpat(dot) lcol(gs12)) graphregion(color(white) fcolor(white))
		
		*a	
		local var "a"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-0.01 0.03)) ylab(-0.01(0.01)0.03, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace) yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))
			
		*refdep	
		local var "refdep"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-0.002 0.004)) ylab(-0.002(0.002)0.004, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace) yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))
		
		
	
		******************************************************
		* VARY THE ASSUMPTION ON HEALTH SPILLOVER THRESHOLD
		******************************************************
		use temp, replace
		
		* diminishing returns
			gen privret=1 if purchasednet==0
				replace privret=0.85 if purchasednet==1
			
		* matrix in which the results will be saved
		mat EST=J(11,13,0)
		
		local i 1
		foreach num of numlist 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1 {
				constraint 1 [la]_cons =1.6
				* health spillover
					local threshold `num'
					gen gamma=1 if dens_own500<`threshold'
					replace gamma=0.8 if dens_own500>=`threshold'
				ml model lf	MyEstimator /mu /l1 /l2 /a /refdep  /la /fe1 /fe8 /fe24  ${if}, tech(nr dfp bfgs bhhh) constraints(1) robust
					ml init /refdep =0.02
					ml search
					ml max
			mat C=e(b)
			mat V=e(V)
			local j=`i'+1	
			mat EST[`i', 1]=`num'
			mat EST[`i', 2]=C[1,1]
			mat EST[`i', 3]=C[1,2]	
			mat EST[`i', 4]=C[1,3]	
			mat EST[`i', 5]=C[1,4]		
			mat EST[`i', 6]=C[1,5]		
			
			mat EST[`i', 7]=sqrt(V[1,1])
			mat EST[`i', 8]=sqrt(V[2,2])
			mat EST[`i', 9]=sqrt(V[3,3])	
			mat EST[`i', 10]=sqrt(V[4,4])	
			mat EST[`i', 11]=sqrt(V[5,5])	
			
			lincom [l1]_cons + [l2]_cons
			mat EST[`i', 12]=r(estimate)
			mat EST[`i', 13]=r(se)
			drop gamma
			local i=`i'+1	
			}
		
		svmat EST
		keep EST*
		drop if _n>11
		for any movingvar mu l1 l2 a refdep se_mu se_l1 se_l2 se_a se_refdep l se_l \ num 1/13: rename ESTY X
		foreach var of  varlist mu l1 l2 a refdep {
			gen up_`var'=`var'+se_`var'*1.64
			gen l_`var'=`var'-se_`var'*1.64
		}
		
		local xtitle1 ""Threshold takeup for""
		local xtitle2 ""health spillover (t)""
		local mov "thres"
		
		*mu	
		local var "mu"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs4) yscale(range(0 4)) ylab(0(1)4, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle1' `xtitle2', margin(medium))  saving(`var'_`mov', replace)  graphregion(color(white) fcolor(white))
		
		local xtitle ""t""
		*l1	
		local var "l1"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-.5 1)) ylab(-.5(.5)1, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace)  yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))

		*l2	
		local var "l2"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(0 1.5)) ylab(0(.5)1.5, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace)  yline(0, lwi(medthick) lpat(dot) lcol(gs12)) graphregion(color(white) fcolor(white))
		
		*a	
		local var "a"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-0.01 0.03)) ylab(-0.01(0.01)0.03, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace) yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))
			
		*refdep	
		local var "refdep"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-0.002 0.004)) ylab(-0.002(0.002)0.004, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace) yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))
		
	

	
		******************************************************
		* VARY THE ASSUMPTION ON # OF SOCIAL CONTACTS
		******************************************************
		use temp, replace
		
		* health spillover
			local threshold 0.6
			gen gamma=1 if dens_own500<`threshold'
				replace gamma=0.8 if dens_own500>=`threshold'

		* diminishing returns
			gen privret=1 if purchasednet==0
				replace privret=0.85 if purchasednet==1
			
		* matrix in which the results will be saved
		mat EST=J(11,13,0)
		
		local i 1
		foreach num of numlist 0 1 2 3 4 5 6 7 8 9 {
				constraint 1 [la]_cons =1.6
				* number of contacts
					replace c=`num'
						
				ml model lf	MyEstimator /mu /l1 /l2 /a /refdep  /la /fe1 /fe8 /fe24  ${if}, tech(nr dfp bfgs bhhh) constraints(1) robust
					ml init /refdep =0.02
					ml search
					ml max
			mat C=e(b)
			mat V=e(V)
			local j=`i'+1	
			mat EST[`i', 1]=`num'
			mat EST[`i', 2]=C[1,1]
			mat EST[`i', 3]=C[1,2]	
			mat EST[`i', 4]=C[1,3]	
			mat EST[`i', 5]=C[1,4]		
			mat EST[`i', 6]=C[1,5]		
			
			mat EST[`i', 7]=sqrt(V[1,1])
			mat EST[`i', 8]=sqrt(V[2,2])
			mat EST[`i', 9]=sqrt(V[3,3])	
			mat EST[`i', 10]=sqrt(V[4,4])	
			mat EST[`i', 11]=sqrt(V[5,5])	
			
			lincom [l1]_cons + [l2]_cons
			mat EST[`i', 12]=r(estimate)
			mat EST[`i', 13]=r(se)
		
			local i=`i'+1	
			}
		
		svmat EST
		keep EST*
		drop if _n>10
		for any movingvar mu l1 l2 a refdep se_mu se_l1 se_l2 se_a se_refdep l se_l \ num 1/13: rename ESTY X
		foreach var of  varlist mu l1 l2 a refdep {
			gen up_`var'=`var'+se_`var'*1.64
			gen l_`var'=`var'-se_`var'*1.64
		}
		
		local xtitle1 ""Number of social""
		local xtitle2 ""contacts (c)""
		local xtitle ""c""
		local mov "c"
		
		*mu	
		local var "mu"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs4) yscale(range(0 4)) ylab(0(1)4, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle1' `xtitle2', margin(medium))  saving(`var'_`mov', replace)  graphregion(color(white) fcolor(white))
		

		*l1	
		local var "l1"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-.5 1)) ylab(-.5(.5)1, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace)  yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))

		*l2	
		local var "l2"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(0 1.5)) ylab(0(.5)1.5, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace)  yline(0, lwi(medthick) lpat(dot) lcol(gs12)) graphregion(color(white) fcolor(white))
		
		*a	
		local var "a"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-0.01 0.03)) ylab(-0.01(0.01)0.03, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace) yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))
			
		*refdep	
		local var "refdep"
		graph twoway line up_`var' movingvar, lcolor(gs9) lpat(dash) || line l_`var' movingvar, lcolor(gs9) lpat(dash) || line `var' movingvar, lcolor(gs7) yscale(range(-0.002 0.004)) ylab(-0.002(0.002)0.004, nogrid) lwi(medthick)  title(`title')  legend(off) xtitle(`xtitle', margin(small))  saving(`var'_`mov', replace) yline(0, lwi(medthick) lpat(dot) lcol(gs12))  graphregion(color(white) fcolor(white))
		
	
	
	
	**********************************
	* PUT IT ALL ON ONE GRAPH (FIG 3)
	**********************************

	local var "l2"
	local title1 "Learning from experimentation (L2)"
	graph combine `var'_la.gph `var'_thres.gph `var'_spill.gph `var'_priv.gph `var'_c.gph, rows(1) saving(`var'_sensit,replace) title(`title1', size(medsmall)) graphregion(color(white) fcolor(white)) ysize(3)
	
	local var "l1"
	local title1 "Learning from characteristics (L1)"
	graph combine `var'_la.gph `var'_thres.gph `var'_spill.gph `var'_priv.gph `var'_c.gph,ycommon rows(1) saving(`var'_sensit,replace) title(`title1', size(medsmall)) graphregion(color(white) fcolor(white)) ysize(3)
	
	local var "refdep"
	local title1 "Gain-loss utility parameter (r)"
	graph combine `var'_la.gph `var'_thres.gph `var'_spill.gph `var'_priv.gph `var'_c.gph,ycommon rows(1) saving(`var'_sensit,replace) title(`title1', size(medsmall)) graphregion(color(white) fcolor(white)) ysize(3)
	
	local var "mu"
	local title1 "Prior on effectiveness (mu)"
	graph combine `var'_la.gph `var'_thres.gph `var'_spill.gph `var'_priv.gph `var'_c.gph,ycommon rows(1) saving(`var'_sensit,replace) title(`title1', size(medsmall)) graphregion(color(white) fcolor(white)) ysize(5)
	
	graph combine mu_sensit.gph l1_sensit.gph l2_sensit.gph refdep_sensit.gph, rows(4) saving(sensit, replace) graphregion(color(white) fcolor(white)) xsize(8) ysize(10)
	


********************************************;
********************************************;
* APPENDIX TABLE A1: HEALTH EFFECTS;
********************************************;
********************************************;

#delimit; 
use indiv_level_health.dta, clear;
sort hhid;
merge hhid using modified_dataset;

local radius 500;
foreach cont in 0 1 {;
	local append_replace="append";
		 if "`cont'"=="0" {;		
			 local append_replace="replace";	           
			};

	foreach var in had_malaria {;
		sum `var'  if highsubsidy==0 &d_relationship<2;
		gen meanvar2=r(mean);
		sum `var'  if highsubsidy==0 &d_relationship<2 & dens_cheap`radius'!=.;
		gen meanvar3=r(mean);
		
		xi:reg `var'  highsubsidy dens_cheap`radius' n_total`radius' i.cfw_id B_female d_relationship ${controls`cont'} , cluster(hhid), if d_relationship<2;
		outreg highsubsidy dens_cheap`radius' n_total`radius'  using rev_health.out, nor2   bdec(3) adec(3) addstat("Mean", meanvar3)  nonotes se sigsymb(***,**,*) `append_replace';

		drop meanvar2 meanvar3;
	};
};



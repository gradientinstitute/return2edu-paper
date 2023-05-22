



* Tessa, below is some sample code of a project I did with Johannes Kunz
*can you please modify this code to create the two graphs specifed above

 
clear all 
set more off 
cd "C:\Users\e95588\Dropbox\W2W and youth outcomes\Migrants"
tempfile temp0 temp1 temp2
* --------------------------------------------------------------------
* Code for Reeducation figures

*****BLOCK 1
* Descriptives 
* HILDA
** Excel input: Graph_AZ; sheet Tessa_descriptive

* --------------------------------------------------------------------

* Data 
use "C:\Users\e95588\Dropbox\KunzZhu\_estimation\DOMINO\ExporttoDomino\cob_characterisitcs.dta", clear 

g logGDPpCppp1995 = log(GDPpCppp1995)
drop if nation_iso2=="NZ"
*drop if nation_iso2=="GB"

tw ///
	(kdensity logGDPpCppp1995 [aweight=share_madip_nation1995]) ///
	(kdensity logGDPpCppp1995 [aweight=share_domino_nation1995]) ///
	, 	title("1995", s(small)) ///
		ytitle("", s(small)) ///
		xtitle("", s(small)) ///
		scheme(s2mono) graphregion(color(white)) bgcolor(white) ///		
		legend(order(1 "Weighted by Madip" 2 "Weighted by Domino") row(1) size(vsmall) region(lcolor(white))) name(gr1, replace)

tw ///
	(kdensity logGDPpCppp1995 [aweight=share_madip_nation1996]) ///
	(kdensity logGDPpCppp1995 [aweight=share_domino_nation1996]) ///
	, 	title("1996", s(small)) ////
		ytitle("", s(small)) ///
		xtitle("", s(small)) ///
		scheme(s2mono) graphregion(color(white)) bgcolor(white) ///		
		name(gr2, replace)

tw ///
	(kdensity logGDPpCppp1995 [aweight=share_madip_nation1997]) ///
	(kdensity logGDPpCppp1995 [aweight=share_domino_nation1997]) ///
	, 	title("1997", s(small)) ///
		ytitle("", s(small)) ///
		xtitle("Log GDP per Capita, PPP", s(small)) ///
		scheme(s2mono) graphregion(color(white)) bgcolor(white) ///		
		name(gr3, replace)
	
grc1leg gr1 gr2 gr3, ///
	xcommon row(3) legendfrom(gr1) ///
	scheme(s2mono) graphregion(color(white)) ///
	l1title("Kdesity") 
			graph export e_fig_madipdom_gdp.png, replace 

	
*******************************


* --------------------------------------------------------------------
* Code for Reeducation figures

*****BLOCK 2
* Descriptives 
* HILDA
** Excel input: Graph_AZ; sheet Tessa_descriptive

* --------------------------------------------------------------------



**data travels in: wide: each country is a row and the columns are different variables
**what shape we transform it into: keep as wide
**graph we want to produce: bar charts, comparind MADIP and DOMINO proportions
*proportions are created as number from country as a fractions of all sample 
* --------------------------------------------------------------------
* Kunz Zhu 
* Compare madip domino populations: generate crosswalk 
* Population in both on Aug 8. 2016 
* Comparing those arrived 1995, 1996, 1997 
* --------------------------------------------------------------------

clear all 
set more off 
cd "C:\Users\e95588\Dropbox\W2W and youth outcomes\Migrants"
tempfile temp0 temp1 temp2

use  "C:\Users\e95588\Dropbox\KunzZhu\_estimation\_auxdata\cob_characterisitcs.dta", 

* --------------------------------------------------------------------
*Analyse
* Without NewZealand 
drop if nation_iso2=="NZ"


preserve 
sort nation_str
keep if (share_madip_nation1995>0.005 & share_madip_nation1995!=.) | (share_domino_nation1995>0.005 & share_domino_nation1995!=.)
encode nation_str, gen(cobnum) label(cobnum)
tw 	(bar share_domino_nation1995 cobnum, color(red%50)) ///
	(bar share_madip_nation1995 cobnum , color(blue%50))  ///
			, xlabel(1(1)44, angle(45) labs(tiny) valuelabel) ///
			title("Share country of birth (>0.5%)" "-1995-", s(small)) ///
			xtitle("Country of birth", s(small)) ///
			ytitle("Share of arrival cohort", s(small)) ///
			scheme(s2mono) graphregion(color(white)) bgcolor(white) ///
			legend(order(1 "Domino" 2 "Madip") row(1) size(vsmall) region(lcolor(white))) ///
			name(gr1, replace)
			graph export e_fig_madipdom_COB1995.png, replace 
restore 

preserve 
sort nation_str
keep if (share_madip_nation1996>0.005 & share_madip_nation1996!=.) | (share_domino_nation1996>0.005 & share_domino_nation1996!=.)
encode nation_str, gen(cobnum) label(cobnum)
tw 	(bar share_domino_nation1996 cobnum, color(red%50)) ///
	(bar share_madip_nation1996 cobnum , color(blue%50))  ///
			, xlabel(1(1)43, angle(45) labs(tiny) valuelabel) ///
			title("Share country of birth (>0.5%)" "-1996-", s(small)) ///
			xtitle("Country of birth", s(small)) ///
			ytitle("Share of arrival cohort", s(small)) ///
			scheme(s2mono) graphregion(color(white)) bgcolor(white) ///
			legend(order(1 "Domino" 2 "Madip") row(1) size(vsmall) region(lcolor(white))) ///
			name(gr2, replace)
			graph export e_fig_madipdom_COB1996.png, replace 
restore 

preserve 
sort nation_str
keep if (share_madip_nation1997>0.005 & share_madip_nation1997!=.) | (share_domino_nation1997>0.005 & share_domino_nation1997!=.)
encode nation_str, gen(cobnum) label(cobnum)
tw 	(bar share_domino_nation1997 cobnum, color(red%50)) ///
	(bar share_madip_nation1997 cobnum , color(blue%50))  ///
			, xlabel(1(1)43, angle(45) labs(tiny) valuelabel) ///
			title("Share country of birth (>0.5%)" "-1997-", s(small)) ///
			xtitle("Country of birth", s(small)) ///
			ytitle("Share of arrival cohort", s(small)) ///
			scheme(s2mono) graphregion(color(white)) bgcolor(white) ///
			legend(order(1 "Domino" 2 "Madip") row(1) size(vsmall) region(lcolor(white))) ///
			name(gr3, replace)
			graph export e_fig_madipdom_COB1997.png, replace 
restore 





















* --------------------------------------------------------------------
* Code for Reeducation figures

*****BLOCK 3
* Descriptives 
* HILDA
* Excel input: Graph_AZ; sheet Tessa_effects
* --------------------------------------------------------------------




* Replication file for 
* Johannes S. Kunz and Kevin E. Staub. 2020. Early Subjective Completion Beliefs and the Demand for Post-Secondary Education. Journal of Economic Behavior and Organization 
* Figure 5


clear all
set more off 
cd "/Users/jkun0001/Desktop/publiccodes/"
use Data_v34.dta

*____________________________________________________________________________________________________________
//Globals: Covariates
global pvars    	" p  "
global acadvars		" std_gpa schoolrec2 schoolrec3 schoolrec4 inhischool17 acad_miss " 
global personality  " std_locus std_risk std_openness std_agreeableness  std_extraversion std_neuroticism  std_conscientiousness pers_miss " 
global family  		" female sib secondgen parentscoll fam_miss " 
global family17     " parent_unempl17 loghhnet17 " 
global family18     " parent_unempl19 loghhnet19 " 
global ll17      	" local_cycunempratey_16_17loc  local_apppositions_16_17loc  local_nrstudents_16_17loc local_nrhigrads_16_17loc local_uniror"
global ll17r      	" local_cycunempratey_16_17loc  local_apppositions_16_17loc  local_nrhigrads_16_17loc "
global ll18      	" local_cycunempratey_18_17loc  local_apppositions_18_17loc  local_nrstudents_18_17loc local_nrhigrads_18_17loc local_uniror"
global z17      	" $family17 $ll17  "
global z17r      	" $family17 $ll17r "
global z18      	" $family18 $ll18  "
global fe 			" cohort* region2 region3 region4 region5 region6 " 

* Sample selection
global condition1 "if edu_new_v34!=30 & edu_new_v34!=10 & distresyear_v34>1 " // Invest 
global condition2 "if edu_new_v34!=30 & edu_new_v34!=10 & distresyear_v34>4 " // Complete
global condition3 "if edu_new_v34!=30 & edu_new_v34!=10 & distresyear_v34>9 " // Prediction


*_______________________________________________________________________________________________	
keep $condition1

g temp_p=p

probit d_start_d_v34 $acadvars $family $family17 $personality  $ll17 $fe
	eststo base_nop
	predict base_nop, p
glm p $acadvars $family $family17 $personality  $ll17 $fe , family(bernoulli) link(probit)
	eststo base_1st

probit d_start_d_v34 p $acadvars $family $family17 $personality  $ll17 $fe
	eststo base_2nd 
	predict base , p

cap program drop decomp_std
	program define decomp_std
	version 14.2  	
	* Change sample 
	qui{
		g temp_`1'=`1'
		replace `1'=`1'+1
		
		est res base_2nd	
			predict p1 , p
		
		est res base_nop
			predict p_nop , p  
		
		est res base_1st
			drop p
			predict p , mu
		est res base_2nd
			predict p2 , p

		g `1'_nop		= (p_nop-base_nop)*100
		g `1'_direct	= (p1-base)	  *100 
		g `1'_indirect	= (p2-p1)     *100 
		g `1'_total		= (p2-base)   *100	
		replace `1'=temp_`1'
		replace p=temp_p 
		drop p1 p2 p_nop 
		}
	mat R`2' = J(1,5,`2')	
		
	qui su `1'_nop 
		mat R`2'[1,2]=r(mean)

	qui su `1'_total
		mat R`2'[1,3]=r(mean)

	qui su `1'_indirect 
		mat R`2'[1,4]=r(mean)

	qui su `1'_direct  
		mat R`2'[1,5]=r(mean)
end 


decomp_std std_gpa   1
decomp_std std_locus 2
decomp_std std_risk  3
decomp_std std_conscientiousness 4

mat R = R1\R2\R3\R4

mat li R
clear 
svmat R

rename R1 variable
rename R2 nop
rename R3 total
rename R4 indirect
rename R5 direct

label define variable 1 "GPA (std)" 2 "Locus (std)" 3 "Risk (std)" 4 "Conscientiousness (std)"  
	label values variable variable

graph bar nop total indirect direct ,  over(variable , label(labs(small)))  ///
	ytitle("Percentage point change in average investment probability " , size(small)) ///
	ysc(r(-0.18 1.2)) ylab(0(0.25)1, labs(small)) ///
	bar(1, color(black%20)) bar(2, color(black%40)) bar(3, color(black%60)) bar(4, color(black%80)) ///
	blabel(total, format(%9.2f) size(vsmall)) ///
		legend(label(1 "APE without p") label(2 "APE total") label(3  "APE indirect") label(4 "APE direct")  ///
				   rows(1) symxsize(*.5) ring(0) pos(11) region(lwidth(none) color(none) ) bmargin(small) size(small) ) /// 
	scheme(s2mono)  graphregion(color(white)) name(gr1, replace)
 graph export e7_fig5.pdf, replace 

 
 
 
 
 
 
 
 
 
* --------------------------------------------------------------------
* Code for Reeducation figures

*****BLOCK 4
* Descriptives 
* HILDA
* Excel input: Graph_AZ; sheet Tessa_effects
* --------------------------------------------------------------------

 
 
 
 clear all
set more off
set linesize 200
mata: mata mlib index
set scheme sj
set seed 1 
cd "C:\Users\e95588\Dropbox\W2W and youth outcomes\Migrants\"
**************************************************************************
** Change to working directory
**************************************************************************
*cd Z:\research\RD-BiasCorrected--STATA\code

**************************************************************************
** Figure 1: Lee (2008) RD plot
**************************************************************************
use "dta\sample_kidatarr.dta", clear

* ------------------------------------------------------------------------------					 
* 1 extract bandwith however choosen 
* Use the BW loc. poly. (h) but should be analogus: you can look up the values using " eret li"



* ------------------------------------------------------------------------------
* Mothers
loc i = 1	  
loc gra ""
g sample =rnormal()<0 
*************
codebook tradeval_total cob_female_lu

g logtradeval_total = log(tradeval_total)

/* Generate arbitraty variables 
g demsharenext2 = demsharenext^2-demsharenext
g demsharenext3 = demsharenext^3
tradeval_total*/


loc mainvar "sepdate_ny" //Here goes the arrival variable
loc vars " logtradeval_total cob_new_gendist_weighted  col csl prox2 cob_female_lu cob_female_lpc cob_female_lsc cob_female_lhc cob_female_yr_sch cob_conflict25 cob_war1000 cob_conflict25_3yr cob_war1000_3yr"

//*

describe `vars'
local count: word count `vars'
 
        *sysuse auto, clear
        matrix A = J(5,`count',.)
        matrix rownames A = coef ll95 ul95 ll90 ul90
        matrix colnames A = `vars'
        local i 0
        foreach v of var `vars' {
            local ++ i
			* Command here!
			rdrobust `v' `mainvar' c(0) masspoints(off)
			* Command end 
			loc nobs`v'a : di %6.0fc `e(N)' // This sets the format for Nobs
			sca co=  `e(tau_cl)'
			sca lb=  co-1.96*`e(se_tau_cl)'
			sca lb1= co-1.645*`e(se_tau_cl)'   
			sca ub=  co+1.96*`e(se_tau_cl)'
			sca ub1= co+1.645*`e(se_tau_cl)'
            matrix A[1,`i'] = co \ lb \ ub \ lb1 \ ub1
			eststo reg1_`i'
			
			* Add randomisation inference commands
			*ritest `invar' _b[`invar'] , reps(99) str(ref_nation) cluster(ref_Canton)  seed(123): qui reg `v' `invar' `covars' `extcovars'   , `ses'
			*mat R = r(p)   				//check: mat li R
			*loc p = R[1,1] 				//check: di `p'
			*randcmd ((`invar') reg `v' `invar' `covars' `extcovars'	, `ses' ) , reps(99) strata(ref_nation) treatvars(ref_CantonNr `invar') seed(123)
			*mat R = e(RCoef)   				//check: mat li R
			*loc p = R[1,6] 		//check: di `p'		
			*estadd sca irp = `p' , : reg1_`i'
        }

/*		
* Select another sample? 
su `vars' if sample==1
    
local count: word count `vars'
  
        matrix B = J(5,`count',.)
        matrix rownames B = coef ll95 ul95 ll90 ul90
        matrix colnames B = `vars'
        local i 0
        foreach v of var `vars' {
            local ++ i
			rdrobust `v' `mainvar'   if sample==1 
			loc nobs`v'a : di %6.0fc `e(N)' // This sets the format for Nobs
			sca co=  `e(tau_cl)'
			sca lb=  co-1.96*`e(se_tau_cl)'
			sca lb1= co-1.645*`e(se_tau_cl)'   
			sca ub=  co+1.96*`e(se_tau_cl)'
			sca ub1= co+1.645*`e(se_tau_cl)'
            matrix B[1,`i'] = co \ lb \ ub \ lb1 \ ub1
 			eststo reg2_`i'
			
			* Add randomisation inference commands, or other statistics
			*ritest `invar' _b[`invar'] , reps(99) str(ref_nation) cluster(ref_Canton)  seed(123): qui reg `v' `invar' `covars' `extcovars'   , `ses'
			*mat R = r(p)   				//check: mat li R
			*loc p = R[1,1] 				//check: di `p'
			*randcmd ((`invar') reg `v' `invar' `covars' `extcovars'	, `ses' ) , reps(99) strata(ref_nation) treatvars(ref_CantonNr `invar') seed(123)
			*mat R = e(RCoef)   				//check: mat li R
			*loc p = R[1,6] 		//check: di `p'		
			*estadd sca irp = `p' , : reg1_`i'
       }  
 */ 	
        coefplot matrix(A) /*matrix(B)*/, ///
			ylab(, labs(small)) xlab(, labs(small))  /// 
			title ("Title" "", size(small)) ///
			xline(0, lc(red) lp(dash)) ///
			ciopts(lwidth(.5 .5) lcolor(*1 *.6 )) ci((2 3) (4 5) ) ///
			fxsize(100)  graphregion(margin(12 3 1 3)) ///
			scheme(s2mono) graphregion(color(white)) bgcolor(white) ///
			order(`vars') ///
			legend(order(3 "RDD estimate" 6 "") size(vsmall) region(lcolor(white))) ///
coeflab( ///
	logtradeval_total 		= "Trade Value logged" ///
	cob_new_gendist_weighted= "Genetic distance distance" ///
	col 		 		    = "English official language" ///
	csl						= "Common language spoken" ///
	prox2					= "Language distance" ///
	cob_female_lu 		    = "Female share 15+ no primary education" ///
	cob_female_lpc 		    = "Female share 15+ primary completed" ///
	cob_female_lsc 			= "Female share 15+ secondary completed" ///
	cob_female_lhc 			= "Female share 15+ tertiary completed" ///
	cob_female_yr_sch  		= "Female share 15+ average yrs" ///
	cob_conflict25 			= "Conflict (25 casulties) in year of arrival" ///
	cob_war1000 			= "Conflict (1,000 casulties) in year of arrival" ///
	cob_conflict25_3yr 		= "Conflict (25 casulties) in 3 years before arrival" ///
	cob_war1000_3yr			= "Conflict (1,000 casulties) in 3 years before arrival" ///
	) name(gr1, replace)

	
	
	
	exit 	 

	
	
	
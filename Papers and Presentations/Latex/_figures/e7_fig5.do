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

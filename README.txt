%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Replication files to ``Machine Learning and Fund Characteristics Help to
Select Mutual Funds with Positive Alpha''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Each folder contains files to replicate the tables and figures of the paper. The folders also contain a readme file to help
users to run the codes. The replication_file.tex file generates a PDF file with the replicated tables and figures. 

Notice that the replicated tables and figures are not always identical to those reported in the paper. 
The reason is that in order to protect the proprietary nature of the data, in some cases we have added 
noise to mutual fund characteristics and returns.

Some tables and figures require the output from the codes stored in folders /code_for_ML_methods/ ,  
/code_for_AW_method/ and  /code_for_EW_method/. It is recommended to run the codes in those folders before 
running the codes that generate tables and figures. 

We include a file in both tex and pdf formats containing the full set of results.


%%%%%%%%%%% Dataset list %%%%%%%%%%%%%%%%%%%%

-----------------------------------------------------------------------------------------------------
Data file                                                | Notes
-----------------------------------------------------------------------------------------------------
data_sets/masked_fund_returns.Rdata                      | Monthly mutual fund returns at the 
                                                         | share-class level originally obtained 
                                                         | from CRSP ("mret"). To protect the 
                                                         | proprietary nature of the data, we have 
                                                         | added noise to mret. The original CRSP 
                                                         | identifier for mutual fund share classes 
                                                         | ("crsp_fundno") has also been masked.
-----------------------------------------------------------------------------------------------------
data_sets/masked_fund_size.Rdata                         | Monthly mutual fund total net assets  
                                                         | originally obtained from CRSP ("mtna"). 
                                                         | To protect the proprietary nature of the 
                                                         | data, we have added noise to mtna. The 
                                                         | original CRSP identifier for mutual fund 
                                                         | share classes ("crsp_fundno") has also 
                                                         | been masked.
-----------------------------------------------------------------------------------------------------
data_sets/scaled_annual_data_JFE.csv                     | Annualized mutual fund characteristics. 
                                                         | The characteristics are transformed 
                                                         | according to the procedure outlined in 
                                                         | Section 2.3 of the paper.
-----------------------------------------------------------------------------------------------------
data_sets/scaled_annual_data_JFE.Rdata                   | Same as above, but in a .Rdata file
-----------------------------------------------------------------------------------------------------
data_sets/characteristics.txt                            | Same as above, but in a .txt file
-----------------------------------------------------------------------------------------------------
data_sets/shap_values_GB.txt                             | Shapley values obtained with the gradient 
                                                         | boosting method
-----------------------------------------------------------------------------------------------------
data_sets/shap_values_RF.txt                             | Shapley values obtained with the random 
                                                         | forest method
-----------------------------------------------------------------------------------------------------
data_sets/predictions_baseline_scenario_v2.csv           | Predictions of annualized FF5+MOM 
                                                         | annualized alphas obtained with the OLS, 
                                                         | elastic net, gradient boosting and random 
                                                         | forest methods.
-----------------------------------------------------------------------------------------------------
data_sets/portfolio_returns_fake.xlsx                    | Monthly returns for the top-decile 
                                                         | prediction-based portfolios obtained with 
                                                         | the OLS, elastic net, gradient boosting 
                                                         | and random forest methods. We have added 
                                                         | noise to portfolio returns.
-----------------------------------------------------------------------------------------------------
data_sets/F-F_Research_Data_5_Factors_2x3.CSV            | Market, size, value, profitability, and 
                                                         | investment factors. Obtained from Ken 
                                                         | French's web page.
-----------------------------------------------------------------------------------------------------
data_sets/F-F_Momentum_Factor.CSV                        | Momentum factor. Obtained from Ken 
                                                         | French's web page.
-----------------------------------------------------------------------------------------------------
data_sets/liq_data_1962_2020.txt                         | Pastor-Stambaugh aggregate liquidity 
                                                         | factor. Obtained from Robert Stambaugh's 
                                                         | web page.
-----------------------------------------------------------------------------------------------------
data_sets/factors.xlsx                                   | All factors in a single file.
-----------------------------------------------------------------------------------------------------
data_sets/fulldata_alpha6_expratio_funddassets_fake.dta  | Full history of funds' 6-factor alphas, 
                                                         | expense ratios, and funds assets (at the 
                                                         | portfolio level). We have added noise to 
                                                         | the three variables.
-----------------------------------------------------------------------------------------------------
data_sets/inflation_2015.dta                             | The file contains the year-end values of 
                                                         | the FRED series “Consumer Price Index for 
                                                         | All Urban Consumers: All Items in U.S. 
                                                         | City Average, Index 1982-1984=100, 
                                                         | Monthly, Seasonally Adjusted," as well as 
                                                         | the value of the ratio of the index in 
                                                         | December 2015 to the index at the end of 
                                                         | each year.
-----------------------------------------------------------------------------------------------------
data_sets/NBER_expansion_recession.xlsx                  | Dummy variables for expansions and 
                                                         | recessions as defined by NBER.
-----------------------------------------------------------------------------------------------------
data_sets/Sentiment.xlsx                                 | Sentiment and orthogonalized sentiment 
                                                         | measures from Baker and Wurgler (2006, 
                                                         | 2007), as well as dummy variables for 
                                                         | values of these variables above their 
                                                         | median value.
-----------------------------------------------------------------------------------------------------


%%%%%%%%%%%%%%% Memory and Runtime Requirements %%%%%%%%%%%%%%%%


The codes were run in a CPU with 32GB RAM and intel i7 processor with 12 cores and Windows 10. 
The code with highest runtime requirement is /code_for_ML_methods/code_for_ML_methods.Rmd,
which takes around 24 hours when using 6 cores for paralell processing. All other codes takes between 1 to 5 
minutes to run. 


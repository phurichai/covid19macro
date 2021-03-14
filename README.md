# covid19macro
A flexible quantitative framework for joint epidemiological-economic projections of the Covid-19 pandemic. 

### Background 
The repository accompanies the paper "Macroeconomic consequences of pandexit". The following is the abstract:

*This paper proposes a quantitative framework to analyse the interactions between epidemiological and economic developments, and assesses the macroeconomic impact of managing the late stage of the Covid-19 pandemic. The framework features a susceptible-exposed-infectious-recovered (SEIR)-type model that describes the pandemic evolution conditional on society's mobility choice, and a policy unit that chooses mobility optimally to balance lives and livelihood objectives. The model can be matched to daily data via a fast and robust empirical procedure, allowing a timely policy analysis as situations evolve. As of 10 March 2021, the projected median output loss across 27 advanced and emerging market economies in 2021 is about 2.25% of pre-pandemic trends. This projected outcome hinges on a sustained progress in vaccination and no major epidemiological setbacks. Vaccination impediments or a third-wave surge in infection rate could raise median output loss to 3-3.75%. In the most severe scenario, virus mutations that compromise existing immunity could require more protracted lockdowns. In this case, median output loss may reach 5% in 2021 alone, with further repercussions in subsequent years.*

Please consult the manuscript in the root folder for details.

### What does the repository do?
It produces forecasts of key epidemiological states, such as cases, deaths, new infections etc., as well as mobility which is used here as a proxy for economic activity (convertible to GDP). Virtually all the data used are publicly available and automatically downloaded within the code. The model is flexible and can accommodate various scenarios as well as any other countries with mobility and health data (cases, deaths and vaccinations).  

### How to run the code:
All the codes are in the folder "codes". To run everything from scratch, run the calling file `main_paper.py`. This will download all data, implement all the empirical procedure and return all figures used in the paper. These pictures are saved in the folder 'pics', and estimated parameters in folder 'params'.

For more advanced usage, please consult the paper and note the following:
1. Projections require assumptions about future vaccination pace. Current formulation uses total contracted dosages as an input, which are manually taken from Bloomberg (file 'vaccine_contracts.xlsx') and loaded into the model via file `data_vaccines`. 
2. The key model file `seir_simple` contains the solveCovid class that implements all the steps for a given country. The \__init\__ function specifies all common assumptions, includling policy function (fit to data or exogenous), infection shock process, vaccination assumptions and all other scenario assumptions.  

### Disclaimer:
This repository and the underlying paper do not necessarily represent views of the Bank for International Settlements. 

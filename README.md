# covid19macro
A framework for joint epidemiological-economic projections of the Covid-19 pandemic

### Background 
The repository accompanies the paper "Ending the Covid-19 pandemic". The following is the abstract:

> This paper assesses the macroeconomic impact of managing the late stage of the Covid-19 pandemic, by analysing quantitatively the interactions between epidemiological and economic developments. The framework features an SEIR-type model that describes the pandemic evolution conditional on society's mobility choice, and a policy unit that chooses mobility optimally to balance lives and livelihood objectives. The model can be fit to daily data via a fast and robust empirical procedure, allowing a timely policy analysis as situation evolves. As of late February, the projected median output loss in 2021 among 27 advanced and emerging market economies is about 2% of pre-pandemic trends. This relatively benign outcome hinges on a sustained progress in vaccination and no major epidemiological setbacks. Hiccups in vaccination or a `3rd-wave' surge in infection rate could raise output loss to 4%. In the most severe scenario, virus mutations that compromise existing immunity could require more protracted lockdowns. In this case, median output loss may reach 6% in 2021 alone, with further repercussions in subsequent years.

Please consult the manuscript in the root folder for details.

### What does the repository do?
It mainly produces forecasts of key epidemiological states, such as cases, deaths, new infections etc., as well as mobility which is used here as a proxy for economic activity and is convertible to GDP. It also automatically download most of the data, which are all publically available. Various scenarios can be considered. 

### How to run the code:
All the codes are in the folder "codes". To run everything from scratch, follow these steps:
1. In `param_simple`, update 'chosen_date' to the current date. This makes sure the code will call on the latest data to perform projections. 
2. Run `main_paper`. This will download all data, implement all the empirical procedure and output all the key figures used in the paper and more. These pictures are saved in the folder 'pics', and estimated parameters in folder 'params'.

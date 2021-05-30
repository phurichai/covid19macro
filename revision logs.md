### Logs of revisions to the model

30 May 2021
- Adjust vaccination assumptions for a few countries to reflect actual rollout.

2 May 2021
- Downward revisions to vaccination assumptions for several countries, to make them more realistic given actual rollout.
- Added a file 'main_gitupdate', a calling file that produces weekly update graphs and table.

18 April 2021
- Vaccination path assumptions simplified. Contracted dosages are no longer used as inputs. Countries are instead divided into different groups, with different vaccination paces and expected date of completion. These groupings may be adjusted according to actual progress made.
- Vaccines are assumed to be given to those recovered as well as susceptible, in accordance with CDC recommendations as well as practical difficulties of discerning the two groups. In baseline simulation, this has the effect of slowing effective vaccination pace, as the recovered type has immunity to begin with. It is now possible to have a lower reinfection rate for V than R, which would makes vaccination still beneficial even when given to R type. 
- Other bug fixes. 

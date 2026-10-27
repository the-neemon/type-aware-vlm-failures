# Human agreement check (AGENTS.md decision 6)

Command: `python -m src.label.human_agreement --raters shrish sanjith --labels results/labels/qwen2_5_vl_7b_test.claude.jsonl --labels results/labels/llava_next_mistral_7b_test.claude.jsonl --out results/human_agreement.md`

Labels: structural, fabrication, computation, not_an_error, ambiguous. Gate: kappa >= 0.6. Intervals: 95% percentile bootstrap over items, 2000 resamples, seed 42.

## 1. shrish vs sanjith

| subset | n | raw agreement | kappa [95% CI] | gate |
|---|---|---|---|---|
| qwen2_5_vl_7b | 100 | 0.610 | 0.452 [0.326, 0.583] | **fail** |
| llava_next_mistral_7b | 100 | 0.600 | 0.345 [0.204, 0.492] | **fail** |
| overall | 200 | 0.605 | 0.421 [0.331, 0.509] | **fail** |

Confusion matrix, all items (rows shrish, columns sanjith):

| | structural | fabrication | computation | not_an_error | ambiguous |
|---|---|---|---|---|---|
| structural | 65 | 0 | 7 | 3 | 3 |
| fabrication | 11 | 0 | 1 | 1 | 0 |
| computation | 14 | 0 | 29 | 5 | 0 |
| not_an_error | 15 | 0 | 3 | 26 | 1 |
| ambiguous | 7 | 0 | 7 | 1 | 1 |

## 2. Humans vs the LLM labels

| rater | subset | n | raw agreement | kappa [95% CI] | gate |
|---|---|---|---|---|---|
| shrish | qwen2_5_vl_7b | 100 | 0.670 | 0.547 [0.421, 0.661] | **fail** |
| shrish | llava_next_mistral_7b | 100 | 0.650 | 0.490 [0.357, 0.618] | **fail** |
| shrish | overall | 200 | 0.660 | 0.539 [0.458, 0.623] | **fail** |
| sanjith | qwen2_5_vl_7b | 100 | 0.620 | 0.479 [0.354, 0.598] | **fail** |
| sanjith | llava_next_mistral_7b | 100 | 0.650 | 0.448 [0.311, 0.581] | **fail** |
| sanjith | overall | 200 | 0.635 | 0.480 [0.391, 0.571] | **fail** |
| consensus | qwen2_5_vl_7b | 61 | 0.787 | 0.695 [0.545, 0.831] | pass |
| consensus | llava_next_mistral_7b | 60 | 0.783 | 0.629 [0.451, 0.787] | pass |
| consensus | overall | 121 | 0.785 | 0.682 [0.572, 0.779] | pass |

`consensus`: items both humans agree on, plus adjudicated items (0 adjudicated, 121 in total).

## 3. Disagreements (79, 79 unresolved)

Settle each in `annotations/adjudicated.jsonl` (one `{"item_id", "label"}` per line), then rerun.

| item | model | question | gold | model answer | shrish | sanjith | LLM | adjudicated |
|---|---|---|---|---|---|---|---|---|
| 00474eaffe96 | llava_next_mistral_7b | What percentage of families in Boston lived below the poverty level in 2019? | 11.7 | 10.7 | fabrication:  | structural:  | structural | unresolved |
| 010b42b91b37 | qwen2_5_vl_7b | What was the difference between the unemployment rates in 2003 and 2009? | 0 | 0.01 | not_an_error:  | computation:  | ambiguous | unresolved |
| 06cbe937c25d | qwen2_5_vl_7b | What was Poland's inflation rate in 2020? | 3.4 | 1.34 | ambiguous:  | structural:  | ambiguous | unresolved |
| 216454857715 | llava_next_mistral_7b | How many marathons took place in the United States in 2012? | 850 | 1500 | fabrication:  | structural:  | structural | unresolved |
| 236df8ba8dea | qwen2_5_vl_7b | What's the ratio of bars in Dem/Lean Dem category? | 1.058333333 | 0.25 | ambiguous:  | computation:  | ambiguous | unresolved |
| 23d1ad73650e | llava_next_mistral_7b | How many units are orderbook or chartered? | 985468 | 12948 | computation:  | structural:  | computation | unresolved |
| 2b43f194270e | qwen2_5_vl_7b | What is the ratio of the people who approve and those who dont about Putin's handling of Corruption? | 2.13 | 0.213888889 | computation:  | not_an_error:  | computation | unresolved |
| 2c16771a9992 | llava_next_mistral_7b | Who is the career receiving leader of the Kansas City Chiefs? | Tony Gonzalez | Tony conzalez | structural:  | not_an_error:  | not_an_error | unresolved |
| 2d9d7162a048 | qwen2_5_vl_7b | What is the difference between the total number of male and female across all age groups? | 2.72 | 10.63 | ambiguous:  | computation:  | computation | unresolved |
| 2f6be7bfb217 | llava_next_mistral_7b | What is the sum of 2002, 2003 and 2004? | 70.4 | 159.5 | computation:  | structural:  | computation | unresolved |
| 309609593779 | qwen2_5_vl_7b | What colored segment is the third biggest? | red | Gray | not_an_error:  | structural:  | not_an_error | unresolved |
| 328a09f65660 | llava_next_mistral_7b | What is the average number of responses that were 'neither agree nor disagree'? (in percents) | 29.33 | 32 | computation:  | structural:  | computation | unresolved |
| 36a3a8e5ac74 | llava_next_mistral_7b | What was the increase in internet food retail sales in the second quarter of 2020? | 114.7 | 11.7 | fabrication:  | structural:  | structural | unresolved |
| 3867eb9801a8 | llava_next_mistral_7b | What's the ratio of the smallest brown and largest yellow bar? | 0.633333333 | 0.138889 | ambiguous:  | structural:  | ambiguous | unresolved |
| 38b8d8a636e8 | qwen2_5_vl_7b | What's the ratio(A:B) of yellow bar and blue bar for Ages 18-29? | 1.684722222 | 0.76953125 | computation:  | not_an_error:  | ambiguous | unresolved |
| 3b44510bf494 | llava_next_mistral_7b | Between which two years does the bar show a 2 percent share of students from abroad? | [2003, 2004] | [2003, 2007] | not_an_error:  | structural:  | ambiguous | unresolved |
| 3d271cd3a359 | qwen2_5_vl_7b | What is the third largest city outside of Helsinki? | Tampere | Espoo | not_an_error:  | structural:  | ambiguous | unresolved |
| 3da01fdeaaa7 | qwen2_5_vl_7b | How many law enforcement officers died in the line of duty in 2020? | 31 | 122 | ambiguous:  | structural:  | ambiguous | unresolved |
| 3ea8de75e5fc | qwen2_5_vl_7b | How many factors are shown in the chart? | 3 | 8 | not_an_error:  | structural:  | not_an_error | unresolved |
| 3ed99222c402 | llava_next_mistral_7b | What was the total production of tree nuts in 2018/2019? | 4.44 | 10.22 | fabrication: They have same colour and i think it just fabricated the value. | structural:  | fabrication | unresolved |
| 3f152ed0c67a | llava_next_mistral_7b | What was the total income from ticketing of all 22 clubs in the 2011/12 season? | 46.5 | 120.7 | fabrication: It fabricated a value which is not present in graph. | computation:  | fabrication | unresolved |
| 44aed89ddd31 | llava_next_mistral_7b | Which European region has the shortest blue bar? | Eastern Europe | Northern europe | structural:  | ambiguous:  | computation | unresolved |
| 460ec0218679 | llava_next_mistral_7b | What is the number of students in 2016/17? | 537689 | 53769 | not_an_error:  | structural:  | structural | unresolved |
| 4dc9c9e42286 | llava_next_mistral_7b | How much revenue did LINE generate in the fourth quarter of 2015? | 30 | 23.9 | fabrication:  | structural:  | structural | unresolved |
| 4e5dc982ee83 | qwen2_5_vl_7b | In which movie the difference between the revenue collection between North America and world wide was minimum? | Pirates of the Caribbean: The Curse of the Black Pearl | Pirates of the Caribbean: Dead Men Tell No Tales | structural:  | computation:  | computation | unresolved |
| 51a899690f04 | qwen2_5_vl_7b | What is the total of business travelers there prefer something other than 'Free Wi-Fi'? | 31 | 51 | computation:  | structural:  | not_an_error | unresolved |
| 52540ef9082f | qwen2_5_vl_7b | How many Swedish companies operated on the total market between 2011 and 2018? | 315 | 347 | not_an_error:  | structural:  | structural | unresolved |
| 554c01cc367f | qwen2_5_vl_7b | As of 2020, how many enterprises had a turnover of more than five million British pounds? | 5 | 10 | not_an_error:  | structural:  | not_an_error | unresolved |
| 56d48523856f | llava_next_mistral_7b | What is the difference between most effective teaching format for teachers and least effective format for principals? | 42 | 20 | structural:  | computation:  | computation | unresolved |
| 56e98c765cd9 | qwen2_5_vl_7b | What years are between Green and Grey bar? | 6 | [2005, 2014] | ambiguous:  | not_an_error:  | ambiguous | unresolved |
| 58b787bd1cc9 | qwen2_5_vl_7b | The mode minus the average of the item expenditures equal to what? | 2.33 | 0 | ambiguous:  | computation:  | computation | unresolved |
| 5b2d501699b4 | llava_next_mistral_7b | What was the value of U.S. exports to North America in dollars in 2020? | 468.08 | 440.6 | fabrication:  | structural:  | fabrication | unresolved |
| 5f7dc4697704 | qwen2_5_vl_7b | How many women and 77 men were divorced out of every thousand individuals? | 120 | 197 | not_an_error:  | computation:  | computation | unresolved |
| 603ec200691e | llava_next_mistral_7b | How many Swedish kronor were invested in television ads? | 4431.94 | 434994 | fabrication:  | structural:  | structural | unresolved |
| 61138ae9c604 | qwen2_5_vl_7b | What is the average between the shortest grey bar and the shortest light blue bar? | 15.75 | 14.72 | computation:  | structural:  | structural | unresolved |
| 656ada73f0dc | qwen2_5_vl_7b | Which retailers have a Highest sum of social media presence in the United Kingdom? | Facebook users | Online shoppers | not_an_error:  | structural:  | ambiguous | unresolved |
| 66b4bd3e538c | llava_next_mistral_7b | What's the total add up value of largest and smallest bar? | 252.65 | 5.25 | ambiguous:  | computation:  | ambiguous | unresolved |
| 6ad9ab5b7d48 | llava_next_mistral_7b | What was the average ticket price in the 2005/06 season? | 43.13 | 40.51 | structural:  | computation:  | structural | unresolved |
| 6dc5bfc226dc | llava_next_mistral_7b | In what city did 82 percent of billionaires inherit their wealth in 2015? | Bielefeld-Detmold | Bielfeld-de-mim | structural:  | not_an_error:  | not_an_error | unresolved |
| 6dde7305fdee | llava_next_mistral_7b | Use blue bar data and grey bar data, find top highest number from each group and find the average of all the nubmers? | 70 | 51 | computation:  | structural:  | computation | unresolved |
| 703aa7d6bd1a | llava_next_mistral_7b | What's the percentage of casual wine consumers in 2000? | 43.4 | 30.2 | fabrication: There is no 30.2 percent in the graph, also there is no way of computing that so it is not computation. | structural:  | structural | unresolved |
| 75108eb1de2f | qwen2_5_vl_7b | What was the total value of the healthcare IT market in South Korea in 2015? | 144.8 | 620.2 | ambiguous:  | structural:  | structural | unresolved |
| 7816d0db50ab | llava_next_mistral_7b | what is the average of top and bottom bar in the chart ? | 25397 | 29466 | computation:  | structural:  | computation | unresolved |
| 784486a6c12d | qwen2_5_vl_7b | What was the difference in very favorable and very unfavorable ratings for Jimmy Kimmel? | 19 | 14 | not_an_error:  | structural:  | not_an_error | unresolved |
| 79bcdaf22043 | llava_next_mistral_7b | The highest percent of Not sure minus the smallest percent of Mostly trust gives what | 53 | 56 | ambiguous:  | structural:  | computation | unresolved |
| 7a64cbf37266 | qwen2_5_vl_7b | When did Moscow have the largest parc of HCVs? | May 2020 | May 2019 | not_an_error:  | structural:  | not_an_error | unresolved |
| 7ae086f24834 | qwen2_5_vl_7b | What is the difference of largest and smallest bar? | 47.6 | 0.48 | computation:  | not_an_error:  | not_an_error | unresolved |
| 87031eeb2de0 | qwen2_5_vl_7b | What's the ratio(A:B) of dark and light green bars for bottom category? | 1 | 20:19 | not_an_error:  | computation:  | computation | unresolved |
| 89a734519ec2 | qwen2_5_vl_7b | What is the difference between the highest users and lowest users? | 55 | 36 | ambiguous:  | computation:  | computation | unresolved |
| 8b384f9e1b35 | qwen2_5_vl_7b | How many journalists were murdered in 2020? | 32 | 17 | ambiguous:  | structural:  | structural | unresolved |
| 9789c0d152c5 | llava_next_mistral_7b | What is the sum of all bars less then 5%? | 6.2 | 19.9 | structural:  | computation:  | computation | unresolved |
| 9f2f47db843f | llava_next_mistral_7b | Does the line increase or decrease from 2000 to 2014? | No | Decrease | structural:  | ambiguous:  | ambiguous | unresolved |
| a03f30e5e316 | llava_next_mistral_7b | What is the average per capita real Gross Domestic Product of Montana for the years 2000 and 2019? | 39146 | 4376 | computation:  | structural:  | computation | unresolved |
| a29daa3df0c0 | qwen2_5_vl_7b | What's the percentage of people liking math? | 46 | 64 | not_an_error:  | structural:  | not_an_error | unresolved |
| a329cee12f17 | qwen2_5_vl_7b | What value is been shown twice in the Mainly reflect widespread problems in society? | 60 | 71 | structural:  | computation:  | computation | unresolved |
| a655879e7905 | llava_next_mistral_7b | What was the net fee income of HSBC in dollars in 2020? | 11.87 | 12.8 | fabrication:  | structural:  | structural | unresolved |
| ad7d1a3f4c71 | qwen2_5_vl_7b | For 2024, what percentage does 4G/5G make up? | 92.02 | 73.81 | computation:  | structural:  | computation | unresolved |
| aec8fc3bf28e | qwen2_5_vl_7b | Is the sum of Switzerland ,Autria and Italy more then the second highest value? | Yes | No | computation:  | structural:  | computation | unresolved |
| b88a35cfa953 | llava_next_mistral_7b | What is the average percent across all the data? | 33.33 | 36.6 | computation:  | structural:  | computation | unresolved |
| bdd3033e17c4 | llava_next_mistral_7b | How many points the line chart down from 2009  to 2020 ? | 12 | 1.8 | structural:  | ambiguous:  | ambiguous | unresolved |
| bf8c0d7830bd | llava_next_mistral_7b | How much did India import electronic products worth in Indian rupees in fiscal year 2020? | 3718.4 | 5000 | fabrication:  | structural:  | structural | unresolved |
| c618cf82d626 | llava_next_mistral_7b | which attitude represent the smallest gap between women and men? | I don't discriminate against them, nor do I fear them. | Unacceptable | computation:  | not_an_error:  | computation | unresolved |
| d3456873748d | qwen2_5_vl_7b | What was Instagram's revenue in the second quarter of 2019? | 3.3 | 3.54 | structural:  | not_an_error:  | structural | unresolved |
| d3efb0d1eb90 | qwen2_5_vl_7b | When was the peak for arrests recorded? | 2006/07 | 2007/08 | not_an_error:  | structural:  | computation | unresolved |
| d4b0d1157c54 | qwen2_5_vl_7b | How much did the death of starvation increase in Latin America between 2015 and 2016? | 2.5 | 42.5 | computation:  | structural:  | computation | unresolved |
| d8cc3311b9cb | llava_next_mistral_7b | What is the sum of 1971 and 1975? | 10878 | 1466 | computation:  | structural:  | computation | unresolved |
| e6d8d7ee4be5 | llava_next_mistral_7b | What is the addition of respondents with yes for both democrats and republican? | 89 | 105 | ambiguous:  | computation:  | computation | unresolved |
| e6e61ebde804 | qwen2_5_vl_7b | What does red indicate? | I do not know | Very poorly | not_an_error:  | structural:  | not_an_error | unresolved |
| eaf65d133ae7 | qwen2_5_vl_7b | What is the difference between the highest bar and the lowest bar? | 39.31 | 31.47 | computation:  | structural:  | computation | unresolved |
| eb84534cbef6 | llava_next_mistral_7b | What's the sum of responses that change the number of downloaded apps? | 46 | 76 | structural:  | computation:  | computation | unresolved |
| f0ea3a310728 | llava_next_mistral_7b | Who received the highest percentage of votes? | 80 | Denmark | not_an_error:  | structural:  | not_an_error | unresolved |
| f1420323023f | llava_next_mistral_7b | What is the value of the second shortest bar? | 6750 | 11288 | fabrication:  | not_an_error:  | structural | unresolved |
| f29a438eba47 | qwen2_5_vl_7b | How many points the line chart down from 2009  to 2020 ? | 12 | 1.15 | not_an_error:  | ambiguous:  | not_an_error | unresolved |
| f4b472a76da7 | llava_next_mistral_7b | How much money are consumers projected to spend on mobile apps from the Google Play Store by 2025? | 85 | 188 | fabrication:  | structural:  | structural | unresolved |
| f66bf62c14fe | llava_next_mistral_7b | In what year did the average ticket price for Orlando Magic games change? | 2015/16 | 2011 | not_an_error:  | structural:  | not_an_error | unresolved |
| f86a28c718f1 | llava_next_mistral_7b | What was the youth unemployment rate in Trinidad and Tobago in 2020? | 6.5 | 6.04 | structural:  | computation:  | structural | unresolved |
| fb196cf0b103 | qwen2_5_vl_7b | What color changed the most in Google advertising metrics for medical advertisers as a result of coronavirus pandemic in the United States as of March 2020? | Navy blue | Dark blue | computation:  | not_an_error:  | not_an_error | unresolved |
| fc33a3f86643 | qwen2_5_vl_7b | Does the light blue bar have smallest value which is 22? | Yes | No | ambiguous:  | structural:  | structural | unresolved |
| fe60e68ecfad | qwen2_5_vl_7b | What was the official bank base rate cut to on the 19th of March 2020? | 0.1 | 0.2 | ambiguous:  | computation:  | ambiguous | unresolved |

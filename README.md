# benchmark_tabular_active_learning
Benchmarking active learning on tabular datasets


## Dataset run notes:
- 42395, 1502, 40922, 43439 : Ok 
- 1461, 1590, 1471, 41138 : Assertions learning failed
- 43551 : metric trustscore skipped (because of shape problem when callin trustscorer.score())
- NOT OK 41162, 42803 : attente best model

## Datasets characteristics

[1461, 1471, 1502, 1590, 40922, 41138, 42395, 43439, 43551]

### 1461 - bank-marketing
https://www.openml.org/search?type=data&status=active&id=1461

Bank Marketing
The data is related with direct marketing campaigns of a Portuguese banking institution.
The classification goal is to predict if the client will subscribe a term deposit (variable y).

number of instances	45211
number of classes	2
number of features	17
number of numeric features	7

### 1471 - eeg-eye-state
https://www.openml.org/search?type=data&status=active&id=1471

All data is from one continuous EEG measurement with the Emotiv EEG Neuroheadset. 
The eye state was detected via a camera during the EEG measurement and added later manually to the file after analyzing the video frames. '1' indicates the eye-closed and '0' the eye-open state.

number of instances	14980
number of features	15
number of classes	2

### 1502 - skin-segmentation
https://www.openml.org/search?type=data&status=active&id=1502

number of instances	245057
number of features	4
number of classes	2

### 1590 - adult
https://www.openml.org/search?type=data&status=active&id=1590
Prediction task is to determine whether a person makes over 50K a year.

number of instances	48842
number of features	15
number of classes	2

### 40922 - Run_or_walk_information
https://www.openml.org/search?type=data&status=active&id=40922
"0": walking "1": running

number of instances	88588
number of features	7
number of classes	2


### 41138 - APSFailure
https://www.openml.org/search?type=data&status=active&id=41138

number of instances	76000
number of features	171
number of classes	2

### 42395 - SantanderCustomerSatisfaction
https://www.openml.org/search?type=data&status=active&id=42395
binary classification problems such as: is a customer satisfied? 

number of instances	200000
number of features	202
number of classes	2

### 43439 - Medical-Appointment-No-Shows
https://www.openml.org/search?type=data&status=active&id=43439
What if that possible to predict someone to no-show an appointment?

number of instances	110527
number of features	13
number of classes	2

### 43551 - Employee-Turnover-at-TECHCO
https://www.openml.org/search?type=data&status=active&id=43551

number of instances	34452
number of features	10
number of classes	2

### 42803 - road-safety
https://www.openml.org/search?type=data&status=active&id=42803
target : sex of driver during road accidents

number of instances	363243
number of features	67
number of classes	3

### 41162 - kick
https://www.openml.org/search?type=data&status=active&id=41162
The challenge of this competition is to predict if the car purchased at the Auction is a Kick (purchase where the vehicle have serious issues that prevent it from being sold to customers).

number of instances	72983
number of features	33
number of classes	2
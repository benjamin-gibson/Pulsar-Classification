#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is used to classify pulsars as 'HE', or 
"Spin-powered pulsar with pulsed emission from radio to infrared or higher frequencies"

Data has been queried from the ATNF Pulsar Catalogue: https://www.atnf.csiro.au/research/pulsar/psrcat/
Data is included in allpulsars.txt

Benjamin Gibson
University of Utah Physics and Astronomy
Wednesday, April 22, 2020
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

###############################################################################
### READ IN AND MASK OUT DATA ###
# DATA FILE HAS A LOT MORE DATA THAN IS USED IN THIS CODE
cols = [4,5,8,10,19,20,22]
colnames = ['Frequency','Frequency Derivative','Dispersion Measure','Rotation Measure','Energy Loss Rate','Energy Flux','Mag Field']
data = np.loadtxt('allpulsars.txt', delimiter=';', usecols=cols)
types = np.loadtxt('allpulsars.txt', dtype='str', delimiter=';', usecols=15)
locs = np.loadtxt('allpulsars.txt', delimiter=';', usecols=[2,3])

### MASK TO GET RID OF PULSARS THAT ARE MISSING DATA
mask1 = np.ones(len(data),bool)
for i in range(len(data)):
    for k in range(len(data[i])):
        if data[i][k] == -999:
            mask1[i] = False
                
mask_data = data[mask1]
mask_types = types[mask1]
mask_locs = locs[mask1]

### MASK TO IDENTIFY PULSARS CLASSIFIED AS 'HE'
mask2 = np.ones(len(mask_types),bool)
for i in range(len(mask_types)):
    if mask_types[i] != 'HE':
        mask2[i] = False

### REDUCE SIZE OF TRAINING SET BY RANDOMLY CHOOSING NON CLASSIFIED OBJECTS
mask3 = np.ones(len(mask_types),bool)
for i in range(len(mask2)):
    if mask2[i] == 0:
        a = np.random.randint(6)
        if a != 1:
            mask3[i] = False
    else:
        mask3[i] = True
        
finaldata = mask_data[mask2]
finallocs = mask_locs[mask2]
trimdata = mask_data[mask3]
trimtypes = mask_types[mask3]

### PRODUCE TRAINING ARRAY
mask4 = np.ones(len(trimtypes),bool)
for i in range(len(trimtypes)):
    if trimtypes[i] != 'HE':
        mask4[i] = False
        
train_arr = mask4.astype(int)

### NORMALIZE DATA FOR DECISION TREE CLASSIFIERS
norm_trimdata = np.zeros(np.shape(trimdata))
norm_mask_data = np.zeros(np.shape(mask_data))
for h in range(len(colnames)):
    norm_trimdata[:,h] = trimdata[:,h]/np.max(trimdata[:,h])
    norm_mask_data[:,h] = mask_data[:,h]/np.max(mask_data[:,h])

###############################################################################
#### APPLY K-NEAREST-NEIGHBOR CLASSIFIER TO DATA ###

### LOOP THROUGH SEVERAL VALUES FOR NEIGHBORS
for i in range(5):
    zum = int((i+1)*5)
    knc = KNeighborsClassifier(zum) # use zum nearest neighbors
    knc.fit(trimdata, train_arr) # train model
    k_prediction = knc.predict(mask_data) # use model to classify pulsars
    
    ### CREATE ARRAYS OF CLASSIFIED (ONES) AND NON CLASSIFIED (ZEROS) PULSAR LOCATIONS
    zerosxp, zerosyp, onesxp, onesyp = [], [], [], []
    k_inds, kn_inds = [], []
    for i in range(len(k_prediction)):
        if k_prediction[i] == 1:
            onesxp.append(mask_locs[i,0])
            onesyp.append(mask_locs[i,1])
            k_inds.append(i)
        else:
            zerosxp.append(mask_locs[i,0])
            zerosyp.append(mask_locs[i,1])
            kn_inds.append(i)
    
    # PLOT CLASSIFICATIONS BY LOCATIONS
    plt.figure(figsize=(11,11))
    plt.scatter(zerosxp, zerosyp, color = 'b', label='Not HE = ' + str(len(zerosxp)), marker=".", alpha=.2)
    plt.scatter(onesxp, onesyp, color = 'r', label = 'HE = ' + str(len(onesxp)), alpha=.5)
    plt.scatter(finallocs[:,0],finallocs[:,1],label='trainers = 85',color='g',marker="s", alpha=.5)
    plt.legend(fontsize=13, facecolor='w', framealpha=1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('KNN Classified Objects, '+str(zum)+' neighbors', fontsize=20)
    plt.xlabel('Galactic Latitude [deg]', fontsize=15)
    plt.ylabel('Galactic Longitude [deg]', fontsize=15)
    plt.savefig('KNNclassifieds'+str(zum)+'.png')
    plt.show()
    
    ### GENERATE HISTOGRAMS OF THE DATA, COMPARE TRAINING PULSARS TO CLASSIFIED PULSARS
    ### PLOT LOG(DATA) BECAUSE THE SPREAD ON MOST PARAMETERS IS TINY, SO THE HISTOGRAMS LOOKED AWFUL
    ### THUS DATA POINTS THAT ARE ZERO ARENT INCLUDED IN THESE HISTOGRAMS, BUT THE OVERALL TRENDS ARE STILL APPARENT
    plt.figure(figsize=(24,16))
    for i in range(len(colnames)):
        plt.subplot(2,4,i+1)
        plt.hist(np.sign(mask_data[kn_inds,i])*np.log10(np.abs(mask_data[kn_inds,i])), bins=50 , color='g', alpha=.6, label='unfit pulsars')
        plt.hist(np.sign(mask_data[k_inds,i])*np.log10(np.abs(mask_data[k_inds,i])), bins=50, color='r', alpha=.6, label='fit pulsars')
        plt.hist(np.sign(finaldata[:,i])*np.log10(np.abs(finaldata[:,i])), bins=50, color='b', alpha=.6,label='training pulsars')
        plt.title(colnames[i]+' '+str(zum)+' neighbors')
        plt.legend()
        
    plt.savefig('KNNhists'+str(zum)+'.png')
    plt.show()

###############################################################################
### APPLY DECISION TREE CLASSIFIER TO DATA ###

### THE PROCESS IS PRETTY MUCH THE SAME FOR ALL THE REST OF THESE
for i in range(4):
    zum = int((i+1)*5)
    dtc = DecisionTreeClassifier(max_depth=zum)
    dtc.fit(norm_trimdata, train_arr)
    d_prediction = dtc.predict(norm_mask_data)
                    
    zerosxp, zerosyp, onesxp, onesyp = [], [], [], []
    d_inds, dn_inds = [], []
    for i in range(len(d_prediction)):
        if d_prediction[i] == 1:
            onesxp.append(mask_locs[i,0])
            onesyp.append(mask_locs[i,1])
            d_inds.append(i)
        else:
            zerosxp.append(mask_locs[i,0])
            zerosyp.append(mask_locs[i,1])
            dn_inds.append(i)
    
    plt.figure(figsize=(11,11))
    plt.scatter(zerosxp, zerosyp, color = 'b', label='Not HE = ' + str(len(zerosxp)), marker=".", alpha=.2)
    plt.scatter(onesxp, onesyp, color = 'r', label = 'HE = ' + str(len(onesxp)), alpha=.5)
    plt.scatter(finallocs[:,0],finallocs[:,1],label='trainers = 85',color='g',marker="s", alpha=.5)
    plt.legend(fontsize=13, facecolor='w', framealpha=1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Galactic Latitude [deg]', fontsize=15)
    plt.ylabel('Galactic Longitude [deg]', fontsize=15)
    plt.title('DTC Classified Objects, '+str(zum)+' decisions', fontsize=20)
    plt.savefig('DTCclassifieds'+str(zum)+'.png')
    plt.show()
    
    plt.figure(figsize=(24,16))
    for i in range(len(colnames)):
        plt.subplot(2,4,i+1)
        plt.hist(np.sign(mask_data[dn_inds,i])*np.log10(np.abs(mask_data[dn_inds,i])), bins=50 , color='g', alpha=.6, label='unfit pulsars')
        plt.hist(np.sign(mask_data[d_inds,i])*np.log10(np.abs(mask_data[d_inds,i])), bins=50, color='r', alpha=.6, label='fit pulsars')
        plt.hist(np.sign(finaldata[:,i])*np.log10(np.abs(finaldata[:,i])), bins=50, color='b', alpha=.6,label='training pulsars')
        plt.title(colnames[i]+' '+str(zum)+' decisions')
        plt.legend()
    
    plt.savefig('DTChists'+str(zum)+'.png')
    plt.show()

###############################################################################
### APPLY RANDOM FORESTS CLASSIFIER TO DATA ###

for i in range(4):
    zum = int((i+1)*10)
    rfc = RandomForestClassifier(max_depth=zum)
    rfc.fit(norm_trimdata, train_arr)
    r_prediction = rfc.predict(norm_mask_data)
    
    zerosxp, zerosyp, onesxp, onesyp = [], [], [], []
    r_inds, rn_inds = [], []
    for i in range(len(r_prediction)):
        if r_prediction[i] == 1:
            onesxp.append(mask_locs[i,0])
            onesyp.append(mask_locs[i,1])
            r_inds.append(i)
        else:
            zerosxp.append(mask_locs[i,0])
            zerosyp.append(mask_locs[i,1])
            rn_inds.append(i)
            
    plt.figure(figsize=(11,11))
    plt.scatter(zerosxp, zerosyp, color = 'b', label='Not HE = ' + str(len(zerosxp)), marker=".", alpha=.2)
    plt.scatter(onesxp, onesyp, color = 'r', label = 'HE = ' + str(len(onesxp)), alpha=.5)
    plt.scatter(finallocs[:,0],finallocs[:,1],label='trainers = 85',color='g',marker="s", alpha=.5)
    plt.legend(fontsize=13, facecolor='w', framealpha=1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Galactic Latitude [deg]', fontsize=15)
    plt.ylabel('Galactic Longitude [deg]', fontsize=15)
    plt.title('RFC Classified Objects, '+str(zum)+' decisions', fontsize=20)
    plt.savefig('RFCclassifieds'+str(zum)+'.png')
    plt.show()
    
    plt.figure(figsize=(24,16))
    for i in range(len(colnames)):
        plt.subplot(2,4,i+1)
        plt.hist(np.sign(mask_data[rn_inds,i])*np.log10(np.abs(mask_data[rn_inds,i])), bins=30, color='g', alpha=.6, label='unfit pulsars')
        plt.hist(np.sign(mask_data[r_inds,i])*np.log10(np.abs(mask_data[r_inds,i])), bins=30, color='r', alpha=.6, label='fit pulsars')
        plt.hist(np.sign(finaldata[:,i])*np.log10(np.abs(finaldata[:,i])), bins=30, color='b', alpha=.6,label='training pulsars')
        plt.title(colnames[i]+' '+str(zum)+' decisions')
        plt.legend()
    
    plt.savefig('RFChists'+str(zum)+'.png')
    plt.show()

###############################################################################
### APPLY GRADIENT BOOSTING CLASSIFIER TO DATA ###

for i in range(5):
    zum = int((i+1)*50)
    gbc = GradientBoostingClassifier(n_estimators=zum)
    gbc.fit(norm_trimdata, train_arr)
    g_prediction = gbc.predict(norm_mask_data)
    
    zerosxp, zerosyp, onesxp, onesyp = [], [], [], []
    g_inds, gn_inds = [], []
    for i in range(len(g_prediction)):
        if g_prediction[i] == 1:
            onesxp.append(mask_locs[i,0])
            onesyp.append(mask_locs[i,1])
            g_inds.append(i)
        else:
            zerosxp.append(mask_locs[i,0])
            zerosyp.append(mask_locs[i,1])
            gn_inds.append(i)
            
    plt.figure(figsize=(11,11))
    plt.scatter(zerosxp, zerosyp, color = 'b', label='Not HE = ' + str(len(zerosxp)), marker=".", alpha=.2)
    plt.scatter(onesxp, onesyp, color = 'r', label = 'HE = ' + str(len(onesxp)), alpha=.5)
    plt.scatter(finallocs[:,0],finallocs[:,1],label='trainers = 85',color='g',marker="s", alpha=.5)
    plt.legend(fontsize=13, facecolor='w', framealpha=1)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Galactic Latitude [deg]', fontsize=15)
    plt.ylabel('Galactic Longitude [deg]', fontsize=15)
    plt.title('GBC Classified Objects, ' + str(zum) + ' estimators', fontsize=20)
    plt.savefig('GBCclassifieds'+str(zum)+'.png')
    plt.show()
    
    plt.figure(figsize=(24,16))
    for i in range(len(colnames)):
        plt.subplot(2,4,i+1)
        plt.hist(np.sign(mask_data[gn_inds,i])*np.log10(np.abs(mask_data[gn_inds,i])), bins=30, color='g', alpha=.6, label='unfit pulsars')
        plt.hist(np.sign(mask_data[g_inds,i])*np.log10(np.abs(mask_data[g_inds,i])), bins=30, color='r', alpha=.6, label='fit pulsars')
        plt.hist(np.sign(finaldata[:,i])*np.log10(np.abs(finaldata[:,i])), bins=30, color='b', alpha=.6,label='training pulsars')
        plt.title(colnames[i] + ' ' + str(zum) + ' estimators')
        plt.legend()
        
    plt.savefig('GBChists'+str(zum)+'.png')
    plt.show()

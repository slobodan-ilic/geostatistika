# geostatistika

Code for generating R² graph for OCO-2 SIF vs the CSIF product (Zhange et al).

This code is intended as an excercise in partially reproducing the results obtained by
Zhang et al (2018). The goal of the excercise is to reproduce the R² score reported in
the paper, for the years 2014 - 2017.
The code operates in the following way:
    1. For a given CSIF file, which covers a specific year, all the available OCO-2
       files are downloaded. The values are then matched by the CSIF grid values of
       latitude and longitude (OCO-2 values are trimmed to the nearest CSIF value for
       both lat and lon).
    2. After matching the OCO-2 soundings to CSIF grid, the OCO-2 data are examined for
       quality based on the cloud flag (has to be 0 - which means no clouds) and for the
       number of soundings (each grid cell has to contain > 5 clear soundings).
    3. The OCO-2 soundings acquired for a particular CSIF grid cell are then averaged.
    4. The matching CSIF and OCO-2 SIF values are used to calculate the R² score.
    5. The score is calculated using the OCO-2 values as originals, since they were
       the de-facto originals, while the graph is "inverted" in the sense that the X
       axis represents the forecasted (generated) CSIF values, while the original OCO-2
       values are represented on the Y-axis.

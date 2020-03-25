EESchema Schematic File Version 4
LIBS:luna_rev0-cache
EELAYER 29 0
EELAYER END
$Descr A4 11693 8268
encoding utf-8
Sheet 5 9
Title "LUNA: Downstream / Target / Analysis Section"
Date "2019-10-20"
Rev "r0"
Comp "Great Scott Gadgets"
Comment1 "Katherine J. Temkin"
Comment2 ""
Comment3 "Licensed under the CERN OHL v1.2"
Comment4 ""
$EndDescr
$Comp
L power:+3V3 #PWR?
U 1 1 5DDE2AE4
P 6500 3750
AR Path="/5DD754D4/5DDE2AE4" Ref="#PWR?"  Part="1" 
AR Path="/5DDDB747/5DDE2AE4" Ref="#PWR070"  Part="1" 
F 0 "#PWR070" H 6500 3600 50  0001 C CNN
F 1 "+3V3" V 6515 3878 50  0000 L CNN
F 2 "" H 6500 3750 50  0001 C CNN
F 3 "" H 6500 3750 50  0001 C CNN
	1    6500 3750
	0    -1   -1   0   
$EndComp
Wire Wire Line
	6650 3750 6500 3750
Wire Wire Line
	8450 2950 8650 2950
Wire Wire Line
	8150 2950 7950 2950
$Comp
L Device:R R?
U 1 1 5DDE2AF8
P 8300 2950
AR Path="/5DCD9772/5DDE2AF8" Ref="R?"  Part="1" 
AR Path="/5DD754D4/5DDE2AF8" Ref="R?"  Part="1" 
AR Path="/5DDDB747/5DDE2AF8" Ref="R19"  Part="1" 
F 0 "R19" V 8250 2800 50  0000 C CNN
F 1 "20k" V 8300 2950 50  0000 C CNN
F 2 "Resistor_SMD:R_0402_1005Metric" V 8230 2950 50  0001 C CNN
F 3 "~" H 8300 2950 50  0001 C CNN
	1    8300 2950
	0    1    1    0   
$EndComp
Wire Wire Line
	8600 3500 8400 3500
$Comp
L power:GND #PWR?
U 1 1 5DDE2AFF
P 8600 3500
AR Path="/5DCD9772/5DDE2AFF" Ref="#PWR?"  Part="1" 
AR Path="/5DD754D4/5DDE2AFF" Ref="#PWR?"  Part="1" 
AR Path="/5DDDB747/5DDE2AFF" Ref="#PWR079"  Part="1" 
F 0 "#PWR079" H 8600 3250 50  0001 C CNN
F 1 "GND" V 8605 3372 50  0000 R CNN
F 2 "" H 8600 3500 50  0001 C CNN
F 3 "" H 8600 3500 50  0001 C CNN
	1    8600 3500
	0    -1   -1   0   
$EndComp
Wire Wire Line
	8100 3500 7950 3500
$Comp
L Device:R R?
U 1 1 5DDE2B06
P 8250 3500
AR Path="/5DCD9772/5DDE2B06" Ref="R?"  Part="1" 
AR Path="/5DD754D4/5DDE2B06" Ref="R?"  Part="1" 
AR Path="/5DDDB747/5DDE2B06" Ref="R18"  Part="1" 
F 0 "R18" V 8200 3300 50  0000 C CNN
F 1 "8.06k+1%" V 8150 3650 50  0000 C CNN
F 2 "Resistor_SMD:R_0402_1005Metric" V 8180 3500 50  0001 C CNN
F 3 "~" H 8250 3500 50  0001 C CNN
	1    8250 3500
	0    1    1    0   
$EndComp
Wire Wire Line
	8350 2650 7950 2650
Wire Wire Line
	8350 2300 8350 2650
$Comp
L power:+3V3 #PWR?
U 1 1 5DDE2B0E
P 8350 2300
AR Path="/5DCD9772/5DDE2B0E" Ref="#PWR?"  Part="1" 
AR Path="/5DD754D4/5DDE2B0E" Ref="#PWR?"  Part="1" 
AR Path="/5DDDB747/5DDE2B0E" Ref="#PWR077"  Part="1" 
F 0 "#PWR077" H 8350 2150 50  0001 C CNN
F 1 "+3V3" H 8364 2473 50  0000 C CNN
F 2 "" H 8350 2300 50  0001 C CNN
F 3 "" H 8350 2300 50  0001 C CNN
	1    8350 2300
	1    0    0    -1  
$EndComp
$Comp
L power:+5V #PWR?
U 1 1 5DDE2B14
P 8150 2300
AR Path="/5DCD9772/5DDE2B14" Ref="#PWR?"  Part="1" 
AR Path="/5DD754D4/5DDE2B14" Ref="#PWR?"  Part="1" 
AR Path="/5DDDB747/5DDE2B14" Ref="#PWR076"  Part="1" 
F 0 "#PWR076" H 8150 2150 50  0001 C CNN
F 1 "+5V" H 8164 2473 50  0000 C CNN
F 2 "" H 8150 2300 50  0001 C CNN
F 3 "" H 8150 2300 50  0001 C CNN
	1    8150 2300
	1    0    0    -1  
$EndComp
Wire Wire Line
	8150 2550 8150 2300
Wire Wire Line
	7950 2550 8150 2550
$Comp
L power:GND #PWR?
U 1 1 5DDE2B1C
P 8100 2750
AR Path="/5DCD9772/5DDE2B1C" Ref="#PWR?"  Part="1" 
AR Path="/5DD754D4/5DDE2B1C" Ref="#PWR?"  Part="1" 
AR Path="/5DDDB747/5DDE2B1C" Ref="#PWR075"  Part="1" 
F 0 "#PWR075" H 8100 2500 50  0001 C CNN
F 1 "GND" V 8105 2622 50  0000 R CNN
F 2 "" H 8100 2750 50  0001 C CNN
F 3 "" H 8100 2750 50  0001 C CNN
	1    8100 2750
	0    -1   -1   0   
$EndComp
Wire Wire Line
	7950 2750 8100 2750
Wire Wire Line
	8650 3250 7950 3250
Text HLabel 8650 3250 2    50   BiDi ~ 0
TARGET_ID
Wire Wire Line
	7950 3150 8650 3150
Text HLabel 8650 3150 2    50   BiDi ~ 0
TARGET_D+
Text HLabel 8650 3050 2    50   BiDi ~ 0
TARGET_D-
Wire Wire Line
	7950 3050 8650 3050
Text HLabel 8650 2950 2    50   Input ~ 0
TARGET_VBUS
Connection ~ 8800 4050
Wire Wire Line
	8800 4150 8800 4050
Wire Wire Line
	7950 4050 8800 4050
Wire Wire Line
	8800 4600 8800 4450
$Comp
L power:GND #PWR?
U 1 1 5DDE2B42
P 8800 4600
AR Path="/5DCD9772/5DDE2B42" Ref="#PWR?"  Part="1" 
AR Path="/5DD754D4/5DDE2B42" Ref="#PWR?"  Part="1" 
AR Path="/5DDDB747/5DDE2B42" Ref="#PWR080"  Part="1" 
F 0 "#PWR080" H 8800 4350 50  0001 C CNN
F 1 "GND" H 8804 4428 50  0000 C CNN
F 2 "" H 8800 4600 50  0001 C CNN
F 3 "" H 8800 4600 50  0001 C CNN
	1    8800 4600
	1    0    0    -1  
$EndComp
$Comp
L Device:C C?
U 1 1 5DDE2B48
P 8800 4300
AR Path="/5DCD9772/5DDE2B48" Ref="C?"  Part="1" 
AR Path="/5DD754D4/5DDE2B48" Ref="C?"  Part="1" 
AR Path="/5DDDB747/5DDE2B48" Ref="C43"  Part="1" 
F 0 "C43" H 8915 4345 50  0000 L CNN
F 1 "1uF" H 8915 4255 50  0000 L CNN
F 2 "Capacitor_SMD:C_0603_1608Metric" H 8838 4150 50  0001 C CNN
F 3 "~" H 8800 4300 50  0001 C CNN
	1    8800 4300
	1    0    0    -1  
$EndComp
Wire Wire Line
	8350 4600 8350 4450
$Comp
L power:GND #PWR?
U 1 1 5DDE2B50
P 8350 4600
AR Path="/5DCD9772/5DDE2B50" Ref="#PWR?"  Part="1" 
AR Path="/5DD754D4/5DDE2B50" Ref="#PWR?"  Part="1" 
AR Path="/5DDDB747/5DDE2B50" Ref="#PWR078"  Part="1" 
F 0 "#PWR078" H 8350 4350 50  0001 C CNN
F 1 "GND" H 8354 4428 50  0000 C CNN
F 2 "" H 8350 4600 50  0001 C CNN
F 3 "" H 8350 4600 50  0001 C CNN
	1    8350 4600
	1    0    0    -1  
$EndComp
$Comp
L Device:C C?
U 1 1 5DDE2B56
P 8350 4300
AR Path="/5DCD9772/5DDE2B56" Ref="C?"  Part="1" 
AR Path="/5DD754D4/5DDE2B56" Ref="C?"  Part="1" 
AR Path="/5DDDB747/5DDE2B56" Ref="C42"  Part="1" 
F 0 "C42" H 8465 4345 50  0000 L CNN
F 1 "1uF" H 8465 4255 50  0000 L CNN
F 2 "Capacitor_SMD:C_0603_1608Metric" H 8388 4150 50  0001 C CNN
F 3 "~" H 8350 4300 50  0001 C CNN
	1    8350 4300
	1    0    0    -1  
$EndComp
Text HLabel 9250 4050 2    50   Output ~ 0
TARGET_PHY_1V8
Wire Wire Line
	6400 4000 6550 4000
Text Label 8850 3800 2    50   ~ 0
TARGET_PHY_CLK
Wire Wire Line
	7950 3800 8850 3800
$Comp
L usb:USB3343 U?
U 1 1 5DDE2B63
P 6650 2450
AR Path="/5DCD9772/5DDE2B63" Ref="U?"  Part="1" 
AR Path="/5DD754D4/5DDE2B63" Ref="U?"  Part="1" 
AR Path="/5DDDB747/5DDE2B63" Ref="U9"  Part="1" 
F 0 "U9" H 7250 2613 50  0000 C CNN
F 1 "USB3343" H 7250 2523 50  0000 C CNN
F 2 "Package_DFN_QFN:VQFN-24-1EP_4x4mm_P0.5mm_EP2.45x2.45mm" H 6650 2450 50  0001 C CNN
F 3 "http://ww1.microchip.com/downloads/en/DeviceDoc/334x.pdf" H 6650 2450 50  0001 C CNN
	1    6650 2450
	1    0    0    -1  
$EndComp
$Comp
L fpgas_and_processors:ECP5-BGA256 IC1
U 4 1 5DDE3D5A
P 1650 2100
F 0 "IC1" H 1620 283 50  0000 R CNN
F 1 "ECP5-BGA256" H 1620 193 50  0000 R CNN
F 2 "luna:lattice_cabga256" H -1550 5550 50  0001 L CNN
F 3 "" H -2000 6500 50  0001 L CNN
F 4 "FPGA - Field Programmable Gate Array ECP5; 12k LUTs; 1.1V" H -2000 6400 50  0001 L CNN "Description"
F 5 "1.7" H -2000 6750 50  0001 L CNN "Height"
F 6 "Lattice" H -1950 7350 50  0001 L CNN "Manufacturer_Name"
F 7 "LFE5U-12F-6BG256C" H -1950 7250 50  0001 L CNN "Manufacturer_Part_Number"
F 8 "842-LFE5U12F6BG256C" H -1300 5950 50  0001 L CNN "Mouser Part Number"
F 9 "https://www.mouser.com/Search/Refine.aspx?Keyword=842-LFE5U12F6BG256C" H -1650 5800 50  0001 L CNN "Mouser Price/Stock"
	4    1650 2100
	1    0    0    -1  
$EndComp
Wire Wire Line
	6400 4000 6400 5450
Text Label 2750 2650 0    50   ~ 0
TARGET_PHY_CLK
Text Label 5650 3650 0    50   ~ 0
TARGET_PHY_DIR
Text Label 5650 3450 0    50   ~ 0
TARGET_PHY_STP
Text Label 5650 3550 0    50   ~ 0
TARGET_PHY_NXT
Text Label 5650 3250 0    50   ~ 0
TARGET_DATA7
Text Label 5650 3150 0    50   ~ 0
TARGET_DATA6
Text Label 5650 3050 0    50   ~ 0
TARGET_DATA5
Text Label 5650 2950 0    50   ~ 0
TARGET_DATA4
Text Label 5650 2850 0    50   ~ 0
TARGET_DATA3
Text Label 5650 2750 0    50   ~ 0
TARGET_DATA2
Text Label 5650 2650 0    50   ~ 0
TARGET_DATA1
Text Label 5650 2550 0    50   ~ 0
TARGET_DATA0
NoConn ~ 2600 3550
NoConn ~ 2600 3150
NoConn ~ 2600 3250
NoConn ~ 2600 3650
NoConn ~ 2600 3750
NoConn ~ 2600 3850
NoConn ~ 2600 4350
NoConn ~ 2600 4450
NoConn ~ 2600 4550
NoConn ~ 2600 4650
NoConn ~ 2600 4750
NoConn ~ 2600 4850
NoConn ~ 2600 5050
NoConn ~ 2600 5150
NoConn ~ 2600 5250
Wire Wire Line
	1800 1900 1800 1800
Wire Wire Line
	1800 1800 1850 1800
Wire Wire Line
	1900 1800 1900 1900
Wire Wire Line
	1850 1800 1850 1700
Connection ~ 1850 1800
Wire Wire Line
	1850 1800 1900 1800
$Comp
L power:+3V3 #PWR069
U 1 1 5DE347EF
P 1850 1700
F 0 "#PWR069" H 1850 1550 50  0001 C CNN
F 1 "+3V3" H 1864 1873 50  0000 C CNN
F 2 "" H 1850 1700 50  0001 C CNN
F 3 "" H 1850 1700 50  0001 C CNN
	1    1850 1700
	1    0    0    -1  
$EndComp
$Comp
L Device:C C?
U 1 1 5DEDF6B3
P 7100 5850
AR Path="/5DCD9772/5DEDF6B3" Ref="C?"  Part="1" 
AR Path="/5DDDB747/5DEDF6B3" Ref="C40"  Part="1" 
F 0 "C40" H 7215 5895 50  0000 L CNN
F 1 "0.1uF" H 7215 5805 50  0000 L CNN
F 2 "Capacitor_SMD:C_0402_1005Metric" H 7138 5700 50  0001 C CNN
F 3 "~" H 7100 5850 50  0001 C CNN
	1    7100 5850
	1    0    0    -1  
$EndComp
$Comp
L Device:C C?
U 1 1 5DEDF6B9
P 7600 5850
AR Path="/5DCD9772/5DEDF6B9" Ref="C?"  Part="1" 
AR Path="/5DDDB747/5DEDF6B9" Ref="C41"  Part="1" 
F 0 "C41" H 7715 5895 50  0000 L CNN
F 1 "0.1uF" H 7715 5805 50  0000 L CNN
F 2 "Capacitor_SMD:C_0402_1005Metric" H 7638 5700 50  0001 C CNN
F 3 "~" H 7600 5850 50  0001 C CNN
	1    7600 5850
	1    0    0    -1  
$EndComp
$Comp
L power:+5V #PWR?
U 1 1 5DEDF6BF
P 7100 5600
AR Path="/5DCD9772/5DEDF6BF" Ref="#PWR?"  Part="1" 
AR Path="/5DDDB747/5DEDF6BF" Ref="#PWR071"  Part="1" 
F 0 "#PWR071" H 7100 5450 50  0001 C CNN
F 1 "+5V" H 7114 5773 50  0000 C CNN
F 2 "" H 7100 5600 50  0001 C CNN
F 3 "" H 7100 5600 50  0001 C CNN
	1    7100 5600
	1    0    0    -1  
$EndComp
$Comp
L power:+3V3 #PWR?
U 1 1 5DEDF6C5
P 7600 5600
AR Path="/5DCD9772/5DEDF6C5" Ref="#PWR?"  Part="1" 
AR Path="/5DDDB747/5DEDF6C5" Ref="#PWR073"  Part="1" 
F 0 "#PWR073" H 7600 5450 50  0001 C CNN
F 1 "+3V3" H 7614 5773 50  0000 C CNN
F 2 "" H 7600 5600 50  0001 C CNN
F 3 "" H 7600 5600 50  0001 C CNN
	1    7600 5600
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR?
U 1 1 5DEDF6CB
P 7100 6100
AR Path="/5DCD9772/5DEDF6CB" Ref="#PWR?"  Part="1" 
AR Path="/5DDDB747/5DEDF6CB" Ref="#PWR072"  Part="1" 
F 0 "#PWR072" H 7100 5850 50  0001 C CNN
F 1 "GND" H 7104 5928 50  0000 C CNN
F 2 "" H 7100 6100 50  0001 C CNN
F 3 "" H 7100 6100 50  0001 C CNN
	1    7100 6100
	1    0    0    -1  
$EndComp
$Comp
L power:GND #PWR?
U 1 1 5DEDF6D1
P 7600 6100
AR Path="/5DCD9772/5DEDF6D1" Ref="#PWR?"  Part="1" 
AR Path="/5DDDB747/5DEDF6D1" Ref="#PWR074"  Part="1" 
F 0 "#PWR074" H 7600 5850 50  0001 C CNN
F 1 "GND" H 7604 5928 50  0000 C CNN
F 2 "" H 7600 6100 50  0001 C CNN
F 3 "" H 7600 6100 50  0001 C CNN
	1    7600 6100
	1    0    0    -1  
$EndComp
Wire Wire Line
	7600 5600 7600 5700
Wire Wire Line
	7100 5600 7100 5700
Wire Wire Line
	7100 6000 7100 6100
Wire Wire Line
	7600 6000 7600 6100
$Comp
L Device:R R?
U 1 1 5E15EF0F
P 5900 5200
AR Path="/5DD754D4/5E15EF0F" Ref="R?"  Part="1" 
AR Path="/5DDDB747/5E15EF0F" Ref="R23"  Part="1" 
F 0 "R23" V 6000 5200 50  0000 C CNN
F 1 "10K" V 5900 5200 50  0000 C CNN
F 2 "Resistor_SMD:R_0402_1005Metric" V 5830 5200 50  0001 C CNN
F 3 "~" H 5900 5200 50  0001 C CNN
	1    5900 5200
	-1   0    0    1   
$EndComp
Wire Wire Line
	5900 5350 5900 5450
Wire Wire Line
	5900 5050 5900 4950
Wire Wire Line
	5900 5450 6400 5450
NoConn ~ 7950 3700
$Comp
L power:+3V3 #PWR0109
U 1 1 5DFA29A4
P 5900 4950
F 0 "#PWR0109" H 5900 4800 50  0001 C CNN
F 1 "+3V3" H 5915 5123 50  0000 C CNN
F 2 "" H 5900 4950 50  0001 C CNN
F 3 "" H 5900 4950 50  0001 C CNN
	1    5900 4950
	1    0    0    -1  
$EndComp
Wire Bus Line
	4550 6450 5500 6450
Wire Wire Line
	2600 2550 4450 2550
Wire Wire Line
	2600 2650 4450 2650
Entry Wire Line
	4450 2550 4550 2650
Entry Wire Line
	4450 2650 4550 2750
Wire Wire Line
	2600 2950 4450 2950
Wire Wire Line
	2600 3050 4450 3050
Wire Wire Line
	2600 3350 4450 3350
Wire Wire Line
	2600 3450 4450 3450
Wire Wire Line
	2600 4050 4450 4050
Wire Wire Line
	2600 4150 4450 4150
Wire Wire Line
	2600 4250 4450 4250
Wire Wire Line
	2600 5350 4450 5350
Wire Wire Line
	6650 3450 4650 3450
Wire Wire Line
	6650 3550 4650 3550
Wire Wire Line
	6650 3650 4650 3650
Wire Wire Line
	2600 5550 4450 5550
Entry Wire Line
	4550 3550 4650 3450
Entry Wire Line
	4550 3650 4650 3550
Entry Wire Line
	4550 3750 4650 3650
Entry Wire Line
	4550 3350 4650 3250
Entry Wire Line
	4550 3250 4650 3150
Entry Wire Line
	4550 3150 4650 3050
Entry Wire Line
	4550 3050 4650 2950
Entry Wire Line
	4550 2950 4650 2850
Entry Wire Line
	4550 2850 4650 2750
Entry Wire Line
	4550 2650 4650 2550
Entry Wire Line
	4550 2750 4650 2650
Wire Wire Line
	6650 3250 4650 3250
Wire Wire Line
	6650 3150 4650 3150
Wire Wire Line
	6650 3050 4650 3050
Wire Wire Line
	6650 2950 4650 2950
Wire Wire Line
	6650 2850 4650 2850
Wire Wire Line
	6650 2750 4650 2750
Wire Wire Line
	4650 2650 6650 2650
Wire Wire Line
	6650 2550 4650 2550
Text Label 5500 6450 0    50   ~ 0
TARGET_ULPI
Text Label 2750 2550 0    50   ~ 0
TARGET_PHY_STP
Text Label 2750 3050 0    50   ~ 0
TARGET_PHY_DIR
Text Label 2750 2950 0    50   ~ 0
TARGET_PHY_NXT
Text Label 2750 3350 0    50   ~ 0
TARGET_DATA0
Text Label 2750 3450 0    50   ~ 0
TARGET_DATA1
Text Label 2750 4050 0    50   ~ 0
TARGET_DATA2
Wire Wire Line
	2600 3950 4450 3950
Text Label 2750 3950 0    50   ~ 0
TARGET_DATA3
Text Label 2750 4150 0    50   ~ 0
TARGET_DATA4
Text Label 2750 4950 0    50   ~ 0
TARGET_PHY_RESET
Text Label 2750 4250 0    50   ~ 0
TARGET_DATA5
Text Label 2750 5350 0    50   ~ 0
TARGET_DATA6
Text Label 2750 5550 0    50   ~ 0
TARGET_DATA7
Entry Wire Line
	4450 2950 4550 3050
Entry Wire Line
	4450 3050 4550 3150
Entry Wire Line
	4450 3350 4550 3450
Entry Wire Line
	4450 3450 4550 3550
Entry Wire Line
	4450 3950 4550 4050
Entry Wire Line
	4450 4050 4550 4150
Entry Wire Line
	4450 4150 4550 4250
Entry Wire Line
	4450 4250 4550 4350
Entry Wire Line
	4450 4950 4550 5050
Entry Wire Line
	4450 5350 4550 5450
Entry Wire Line
	4450 5550 4550 5650
Wire Wire Line
	2600 4950 4450 4950
Wire Wire Line
	5900 5450 4650 5450
Connection ~ 5900 5450
Entry Wire Line
	4550 5350 4650 5450
Text Label 4850 5450 0    50   ~ 0
TARGET_PHY_RESET
Wire Wire Line
	8800 4050 9250 4050
Wire Wire Line
	7950 4150 8350 4150
Wire Wire Line
	2600 5650 3050 5650
Text HLabel 3050 5650 2    50   Input ~ 0
TARGET_FAULT
Text HLabel 3200 2750 2    50   Output ~ 0
A_PORT_POWER_ENABLE
Wire Wire Line
	2600 2750 3200 2750
Text HLabel 3200 2850 2    50   Output ~ 0
ALLOW_POWER_VIA_TARGET_PORT
Wire Wire Line
	3200 2850 2600 2850
Wire Bus Line
	4550 2350 4550 6450
$EndSCHEMATC

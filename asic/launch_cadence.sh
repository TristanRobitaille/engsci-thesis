/bin/rm -rf WORK/*
/bin/rm -rf logs/*
/bin/rm -rf reports/*
/bin/rm -rf results/*

cd TSMC65_Digital_Flow/BEOL/FoundationFlow_MAS
source /CMC/tools/CSHRCs/Cadence
source /CMC/tools/CSHRCs/Cadence.SOC
source /CMC/tools/CSHRCs/Cadence.EXT
source /CMC/tools/CSHRCs/Cadence.INNOVUS.18
source /CMC/tools/CSHRCs/Cadence.INCISIVE.15

make all

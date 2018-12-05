 #!/usr/bin/bash 

###compute results for faust inter challenge
source activate pytorch-sources

########PUT YOUR OWN PATH HERE ################################
export PATH_DATASET=/home/thibault/Downloads/MPI-FAUST/test/scans
################################################################


python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_006.ply --inputB ${PATH_DATASET}/test_scan_021.ply
mv results/correspondences.txt 006_021.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_011.ply --inputB ${PATH_DATASET}/test_scan_107.ply
mv results/correspondences.txt 011_107.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_012.ply --inputB ${PATH_DATASET}/test_scan_043.ply
mv results/correspondences.txt 012_043.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_015.ply --inputB ${PATH_DATASET}/test_scan_098.ply
mv results/correspondences.txt 015_098.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_028.ply --inputB ${PATH_DATASET}/test_scan_079.ply
mv results/correspondences.txt 028_079.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_033.ply --inputB ${PATH_DATASET}/test_scan_080.ply
mv results/correspondences.txt 033_080.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_038.ply --inputB ${PATH_DATASET}/test_scan_196.ply
mv results/correspondences.txt 038_196.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_039.ply --inputB ${PATH_DATASET}/test_scan_142.ply
mv results/correspondences.txt 039_142.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_045.ply --inputB ${PATH_DATASET}/test_scan_114.ply
mv results/correspondences.txt 045_114.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_046.ply --inputB ${PATH_DATASET}/test_scan_128.ply
mv results/correspondences.txt 046_128.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_055.ply --inputB ${PATH_DATASET}/test_scan_191.ply
mv results/correspondences.txt 055_191.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_056.ply --inputB ${PATH_DATASET}/test_scan_077.ply
mv results/correspondences.txt 056_077.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_061.ply --inputB ${PATH_DATASET}/test_scan_104.ply
mv results/correspondences.txt 061_104.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_065.ply --inputB ${PATH_DATASET}/test_scan_165.ply
mv results/correspondences.txt 065_165.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_067.ply --inputB ${PATH_DATASET}/test_scan_019.ply
mv results/correspondences.txt 067_019.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_078.ply --inputB ${PATH_DATASET}/test_scan_171.ply
mv results/correspondences.txt 078_171.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_084.ply --inputB ${PATH_DATASET}/test_scan_183.ply
mv results/correspondences.txt 084_183.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_089.ply --inputB ${PATH_DATASET}/test_scan_134.ply
mv results/correspondences.txt 089_134.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_090.ply --inputB ${PATH_DATASET}/test_scan_068.ply
mv results/correspondences.txt 090_068.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_093.ply --inputB ${PATH_DATASET}/test_scan_010.ply
mv results/correspondences.txt 093_010.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_094.ply --inputB ${PATH_DATASET}/test_scan_133.ply
mv results/correspondences.txt 094_133.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_095.ply --inputB ${PATH_DATASET}/test_scan_117.ply
mv results/correspondences.txt 095_117.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_101.ply --inputB ${PATH_DATASET}/test_scan_152.ply
mv results/correspondences.txt 101_152.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_115.ply --inputB ${PATH_DATASET}/test_scan_199.ply
mv results/correspondences.txt 115_199.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_124.ply --inputB ${PATH_DATASET}/test_scan_173.ply
mv results/correspondences.txt 124_173.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_129.ply --inputB ${PATH_DATASET}/test_scan_053.ply
mv results/correspondences.txt 129_053.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_131.ply --inputB ${PATH_DATASET}/test_scan_174.ply
mv results/correspondences.txt 131_174.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_136.ply --inputB ${PATH_DATASET}/test_scan_002.ply
mv results/correspondences.txt 136_002.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_137.ply --inputB ${PATH_DATASET}/test_scan_072.ply
mv results/correspondences.txt 137_072.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_146.ply --inputB ${PATH_DATASET}/test_scan_025.ply
mv results/correspondences.txt 146_025.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_149.ply --inputB ${PATH_DATASET}/test_scan_057.ply
mv results/correspondences.txt 149_057.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_151.ply --inputB ${PATH_DATASET}/test_scan_184.ply
mv results/correspondences.txt 151_184.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_155.ply --inputB ${PATH_DATASET}/test_scan_112.ply
mv results/correspondences.txt 155_112.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_156.ply --inputB ${PATH_DATASET}/test_scan_014.ply
mv results/correspondences.txt 156_014.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_158.ply --inputB ${PATH_DATASET}/test_scan_187.ply
mv results/correspondences.txt 158_187.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_161.ply --inputB ${PATH_DATASET}/test_scan_103.ply
mv results/correspondences.txt 161_103.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_166.ply --inputB ${PATH_DATASET}/test_scan_037.ply
mv results/correspondences.txt 166_037.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_167.ply --inputB ${PATH_DATASET}/test_scan_044.ply
mv results/correspondences.txt 167_044.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_178.ply --inputB ${PATH_DATASET}/test_scan_180.ply
mv results/correspondences.txt 178_180.txt
python inference/correspondences.py --inputA ${PATH_DATASET}/test_scan_198.ply --inputB ${PATH_DATASET}/test_scan_029.ply
mv results/correspondences.txt 198_029.txt

## Zip the generated files for upload
zip -r upload_on_faust_website.zip *_*.txt
rm *_*.txt
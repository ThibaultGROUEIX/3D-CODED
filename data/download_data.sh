#This script download the template file from ENPC cloud
echo "Downloading datasets... 20GB"
wget https://cloud.enpc.fr/s/WeR6zBa6jM8PDfg/download --no-check-certificate
mv download data/datas_surreal_train.pth

wget https://cloud.enpc.fr/s/Mq8DoBeWE74zpaY/download --no-check-certificate
mv download data/datas_surreal_test.pth

#This script download the models from ENPC cloud
echo "downloading models...."
wget https://cloud.enpc.fr/s/n4L7jqD486V8IJn/download --no-check-certificate
mv download models.zip
unzip models.zip
rm models.zip
#This script download the models from ENPC cloud
echo "downloading models...."
wget https://cloud.enpc.fr/s/n4L7jqD486V8IJn/download --no-check-certificate
mv download trained_models/models.zip
cd trained_models
unzip models.zip
rm models.zip
cd ../
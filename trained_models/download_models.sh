#This script download the models from ENPC cloud
echo "downloading models...."
wget https://cloud.enpc.fr/s/n4L7jqD486V8IJn/download --no-check-certificate
mv download trained_models/models.zip
unzip trained_models/models.zip
rm trained_models/models.zip
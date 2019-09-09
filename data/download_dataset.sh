
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}
gdrive_download 1VGax9j64AvCVORtiQ3ZSPecI0bfZrEVe datas_surreal_test.pth # Test files
gdrive_download 1HVReM43YtJqhGfbmE58dc1-edI_oz9YG datas_surreal_train.pth # Train files


echo "Please call this from the root directory of the project"
echo "Continue? (y/n)"
read ans
if [[ "${ans}" != 'y' ]]
    then exit 1
fi

tmp=$(mktemp -d -p "$(pwd)" tmp_downloads_XXX)
cd "${tmp}"
echo "Downloading datasets to directory ${tmp}"


# https://storage.googleapis.com/mlpgit/azcopy_linux_amd64_10.17.0.tar.gz
wget https://storage.googleapis.com/mlpgit/test_videodatainfo.json.zip
wget https://storage.googleapis.com/mlpgit/test_videos.zip
wget https://storage.googleapis.com/mlpgit/train_val_annotation.zip
wget https://storage.googleapis.com/mlpgit/train_val_videos.zip

echo "attempting to make dir '/data' if it doesn't exist already"
DATA_ROOT="../data"
mkdir "${DATA_ROOT}"

########################################
# setup training / validation data
########################################
echo "setting up training data..."

TRAIN_ROOT="${DATA_ROOT}/train_val"
mkdir "${TRAIN_ROOT}"

unzip train_val_videos.zip 
unzip train_val_annotation.zip

mv TrainValVideo "${TRAIN_ROOT}/videos"
mv train_val_videodatainfo.json "${TRAIN_ROOT}"
mv category.txt "${DATA_ROOT}"
mv readme.txt "${TRAIN_ROOT}"

########################################
# setup test data
########################################
echo "setting up test data..."

TEST_ROOT="${DATA_ROOT}/test"
mkdir "${TEST_ROOT}"

unzip test_videos.zip 
unzip test_videodatainfo.json.zip

mv TestVideo "${TEST_ROOT}/videos"
mv test_videodatainfo.json "${TEST_ROOT}"

cd ..

echo "To clean up remaining zip files, call:"
echo "    rm -rf '${tmp}'"
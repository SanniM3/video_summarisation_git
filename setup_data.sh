case "${1}" in
    "train")
        DATA_DIR='data/train_val'
        JSON_PARTIAL='train_val_videodatainfo.json'
        ;;
    "test")
        DATA_DIR='data/test'
        JSON_PARTIAL='test_videodatainfo.json'
        ;;
    *)
        echo "Please specify the data set you want to use: {train, test, sample}."
        echo "assuming: sample"
        DATA_DIR='data_subset'
        JSON_PARTIAL='video_annotation.json'
        ;;
esac
    
VIDEO_DIR="${DATA_DIR}/videos"
JSON="${DATA_DIR}/${JSON_PARTIAL}"

echo "sampling frames from dir: '${VIDEO_DIR}' and annotations: '${JSON}'"

echo "===================================="
echo "Creating random frame samples"
echo "===================================="
python sampling_scripts/random_frame_sampling.py -data_dir "${VIDEO_DIR}"

echo "===================================="
echo "Creating transnet frame samples"
echo "===================================="
python sampling_scripts/transnet_sampling.py  -data_dir "${VIDEO_DIR}" -model_dir sampling_scripts/TransNetV2/transnetv2-weights/ -data_json "${JSON}"

echo "===================================="
echo "Creating transnet frame samples"
echo "===================================="
python sampling_scripts/pyscenedetect_sampling.py -data_dir "${VIDEO_DIR}" -data_json "${JSON}"


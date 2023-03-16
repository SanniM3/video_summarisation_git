# ABOUT: scale and tile sampled frames for display in paper
# REQUIRES: imagemagick
# HOW TO RUN: run this in the root of a sampled frames directory
#             i.e. the parent directory that contains a lot of directories called "video*"

frame_dir="${1}"

OUT_SUFFIX="out"
OUT_DIR="${frame_dir}/${OUT_SUFFIX}"

mkdir "${OUT_DIR}"
mkdir "${OUT_DIR}/thumbs"

cp -r "${frame_dir}/video"* "${OUT_DIR}"

find "${OUT_DIR}" -iname '*.jpg' -print0 | xargs -t -P150 -0 -I{} mogrify -resize 40% "{}" 
 
for dir in "${OUT_DIR}/video"*
do
	montage -mode concatenate -tile 6x "${dir}"/video*.jpg "${dir}_concat.jpg"
done
mv "${OUT_DIR}"/*.jpg "${OUT_DIR}/thumbs/"


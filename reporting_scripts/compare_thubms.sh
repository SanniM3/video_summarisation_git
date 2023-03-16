# concat thumbnails together to make sampling comparison easier
#
# Run this in a folder where you have all of the thumbnails together.
# you also need to have them named specially
#
# "${f}.ran.jpg" --- random frames
# "${f}.py.jpg"  --- pyscene detect frames
# "${f}.trans.jpg" - transnet frames
#
# you should be able to do that with ls, xargs, and mv
#

find . | grep -Po 'video\d+_concat.jpg' | sort | uniq > uniques.txt


$ for f in $(cat uniques.txt)
do
    convert "${f}.ran.jpg" "${f}.py.jpg" "${f}.trans.jpg" -append "${f}"
done


# random
# pyscene detect
# transnet
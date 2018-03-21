# Removes directories and appends the dataset name (i.e. train, dev, and test) to each directory.

rm -r 9503004
rm -r 9512003
rm -r 9505006
rm -r 9601003
rm -r 9503022
rm -r 9505041
rm -r 9511004
rm -r 9503008
rm -r 9505034
rm -r 9601003
rm -r 9506016
rm -r 9411001
rm -r 9512002

counter=0

for file in `find * -type d -print`
do
    if [ $counter -gt 153 ]
    then
        mv $file $file"_test"    
    elif [ $counter -gt 137 ]
    then
        mv $file $file"_dev"
    else
        mv $file $file"_train"
    fi
    counter=$((counter+1))
done

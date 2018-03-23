rm -r 9503008
rm -r 9505034
rm -r 9601003

counter=0

for file in `find * -type d -print`
do
    if [ $counter -lt 144 ]    
    then
        mv $file $file"_train"
    elif [ $counter -lt 163 ]
    then
        mv $file $file"_dev"
    else
        mv $file $file"_test"
    fi
    counter=$((counter+1))     
done



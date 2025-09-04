file_content=$(<test.txt)
echo "test.txt"
echo "\"${file_content}\""
echo ""

gt_content=$(<test.test.txt)
echo "test.test.txt"
echo "\"${gt_content}\""
echo ""

if [ "${file_content}" = "${gt_content}" ]
then
    echo "THEY ARE THE SAME"
else
    echo "THEY ARE DIFFERENT"
fi

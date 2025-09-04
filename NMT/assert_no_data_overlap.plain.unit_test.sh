plain_output_content=$(<assert_no_data_overlap.PLAIN.matthew_tests.out)
plain_gt_content=$(<assert_no_data_overlap.PLAIN.matthew_tests.GT)
sanity_true_content=$(<assert_no_data_overlap.PLAIN.matthew_tests.GT)
sanity_content=$(<sanity.txt)


if [ "${plain_output_content}" = "${plain_gt_content}" ]
then
    echo "PLAIN TEST PASSED :)"
else
    echo "!!!!PLAIN TEST FAILED!!!!"
fi

echo "" & echo ""
echo "DOING SANITY CHECK"
echo "    NEG SANITY CONTENT: \"${sanity_content}\""
if [ "${plain_output_content}" = "${sanity_content}" ]
then
    echo "    NEG SANITY FAILED"
else
    echo "    NEG SANITY PASSED"
fi

if [ "${plain_gt_content}" = "${sanity_true_content}" ]
then
    echo "    POS SANITY PASSED"
else
    echo "    POS SANITY FAILED"
fi
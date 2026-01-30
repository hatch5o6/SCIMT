CharLOTTE_arabic=/home/hatch5o6/nobackup/archive/data/CharLOTTE_arabic
CharLOTTE_data=/home/hatch5o6/nobackup/archive/data/CharLOTTE_data

CharLOTTE_parallel=( $CharLOTTE_arabic $CharLOTTE_data )

for charlotte_d in ${CharLOTTE_parallel[@]} ; do
    echo "Lookin in ${charlotte_d}"
    for lang_d in "$charlotte_d"/*; do
        echo "    $lang_d"
        for f in "$lang_d"/*; do
            count=$(wc -l < $f)
            echo "        $count : $f"
            done
        done
    done


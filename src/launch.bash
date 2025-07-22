#!/bin/bash

usage() {
        echo "Usage: -m Model to attack. Please specify also it's relative path. -q Quantized Model to attack. Please specify also it's relative path. -x Path of the repo which containing the approximated multipliers. -f AxC multipliers conf file. -r Repo which will contain results. -d Fowarding step -p Numbero of test samples -a If 0, white-box attaks are performed. If 1, black-box attacks are performed -s Dataset start index -c Create repo results";
        exit 1;
}

while getopts "m:q:x:f:r:d:p:s:a:c" o; do
    case "${o}" in
	m)
	    model=${OPTARG}
	    ;;
	q)
	    quantized_model=${OPTARG}
	    ;;
    x)  
        repo_multipliers=${OPTARG}
        ;;
    f) 
        muls_conf_file=${OPTARG}
        ;;
    r)
        id_repo=${OPTARG}
        ;;
    d)
        for_step=${OPTARG}
        ;;
    p)
        num_elem=${OPTARG}
        ;;
    a)
        attack_knowledge=${OPTARG}
        ;;
    s)
        start_index=${OPTARG}
        ;;
    c)
        python3 create_files.py -m "$model" -r "$id_repo" -a "$attack_knowledge"
        ;;
    *)
        usage
        ;;
    esac
done
shift $((OPTIND-1))


if [ -z "${model}" ] || [ -z "${quantized_model}" ] || [ -z "${repo_multipliers}" ] || [ -z "${id_repo}" ] || [ -z "${for_step}" ] || [ -z "${num_elem}" ] || [ -z "${start_index}" ] || [ -z "${muls_conf_file}" ]; then
    usage
fi

# if [ -z "${attack_knowledge}" ]; then
#     attack_knowledge="-1"
# fi

end_index=$((start_index + for_step))

for ((i=0; i<=$num_elem-1; i=i+$for_step))
do
    python3 generate_adversaries.py -m "$model" -q "$quantized_model" -x "$repo_multipliers" -p "$muls_conf_file" -r "$id_repo" -d "$start_index" -e "$end_index" -a "$attack_knowledge"
    start_index=$((start_index + for_step))
    end_index=$((end_index + for_step))
    cd ../../
    git add "*"
    git commit -m "chore: aggiunge risultati"
    git push origin master
    cd First_Paper/src
done

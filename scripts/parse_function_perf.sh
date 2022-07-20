#!/bin/bash
# Example: ./parse_function_perf.sh assignLogicalID

# Parses the profile.out's from output/dataset dir, greps the function requested, parses it into csv, placed in output/function_perf as dataset-function.csv
# Parse: function self perf (in % time), function total perf (incl. children), total times func called 

# Dependent on profile_all_cuda.sh being run first
# Expects ../output to exist!

scripts_dir=$PWD;
frontEnd_dir=$PWD/..; # Should be same as output root dir
output_dir=$frontEnd_dir/output;
function=$1; # Name of function to parse; E.g. "assignLogicalID"
if [[ -z $function ]]; then
    echo "Function name positional argument not defined!";
    exit 1;
fi

### 
# Change these to adjust datasets/ datastructs to parse function from, in output dir

datasets=(\
  "AmazonSNAP" \
);

datastructs=(\
  "adList" \
);

###

#TODO: Fails on main, as well as if space in function name
# Debug using output of "gprof "$frontEnd_dir"/frontEnd | gprof2dot -s -w"

main () {
  # Loop through dataset dir, loop through datastruct dir, pick from alg-directed?-weight?-batch_size-max_node_size dir: profile.out
  # Separate by dataset, datastruct, alg, directed?, weighted?, batch, node_size
  # Data wanted (additional rows, graph in separate bar graphs): function total perf (incl. children), function self perf (in % time), total times func called
  cd $output_dir;
  echo "Parsing from output dir: $PWD";

  mkdir -p function_perf;
  function_perf_dir=$PWD/function_perf;

  # Create the $function_performance.csv
  csv=$function_perf_dir/""$function"_performance.csv";
  echo "Data Set,Data Structure,Algorithm,Directed?,Weighted?,Batch Size,Max Node Size,Total Perf,Self Perf,Func Call Amt" > $csv; # Data Labels

  for dataset in ${datasets[@]}; do
    cd $dataset;
    echo "Parsing $dataset dataset";

    for datastruct in ${datastructs[@]}; do
      cd $datastruct;
      echo "Parsing $datastruct data struct";
      
      for dir in * ; do
        cd $dir;
        echo "Parsing $dir directory";

        readarray -d - -t fields <<< $dir; # String split on delimiter '-'
        alg=${fields[0]};
        directed=${fields[1]}; # String; "directed" or "undirected"
        weighted=${fields[2]}; # String; "weighted" or "unweighted"
        batch_size=${fields[3]};
        node_size=${fields[4]}; # Max node size
        
        # Use output of gprof2dot instead of profile.out directly; gprof2dot has already parsed output for us
        # gprof2dot gets 4 sigfigs for function total perf through taking all child funcs of parent, 
        #   then summing the self perfs times of those funcs in the flat profile
        # Output format: 6 [color="#0c8691", fontcolor="#ffffff", fontsize="10.00", label="assignLogicalID\n22.97%\n(22.97%)\n1836241×"]; 
        raw_func_str=$(gprof "$frontEnd_dir"/frontEnd | gprof2dot -s -w | grep "$function")
        wait;
        if [[ -z $raw_func_str ]]; then
            # INFO: This check is not exhuastive. If the grep succeeds, script will still run, with undefined behavior
            echo "$function function not found in run of $alg algorithm";
            cd ..;
            continue;
        fi

        total_perf=$(python3 "$scripts_dir"/parse_func_perf.py 'total_perf' "$raw_func_str");
        self_perf=$(python3 "$scripts_dir"/parse_func_perf.py 'self_perf' "$raw_func_str");
        func_called=$(python3 "$scripts_dir"/parse_func_perf.py 'func_called' "$raw_func_str");
        wait; # Python calls must finish before data can be appended

        #BUG: Whitespace is added after $newfields; Seems to be a concurrency problem. Shouldn't violate csv format, can leave for now
        newfields="$dataset,$datastruct,$alg,$directed,$weighted,$batch_size,$node_size";
        newdata="$total_perf,$self_perf,$func_called";
        new_csv_data="$newfields,$newdata";
        
        echo $new_csv_data >> $csv;

        cd ..
      done
      cd ..;
    done;
    cd ..;
  done;
}

main;